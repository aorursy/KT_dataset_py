import torch
torch.cuda.is_available()
import glob
PSEUDO = len(glob.glob("../input/global-wheat-detection/test/*"))>10
!python -m pip install --upgrade ../input/package/pip-20.1.1-py2.py3-none-any.whl
!pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/.
!pip install --no-deps '../input/timm-package/timm-0.1.26-py3-none-any.whl' > /dev/null
!pip install --no-deps '../input/pycocotools/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl' > /dev/null
!pip install "../input/eff-omegaconf/omegaconf-2.0.0-py3-none-any.whl" > /dev/null
!cp -r ../input/train-effdet/* ./
import sys
# sys.path.insert(0, "../input/train-effdet")
# sys.path.insert(0, "../input/omegaconf")
sys.path.insert(0, "../input/weightedboxesfusion")

import ensemble_boxes
import torch
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import gc
from matplotlib import pyplot as plt
from effdet import get_efficientdet_config, EfficientDet, DetBenchEval
from effdet.efficientdet import HeadNet
def get_valid_transforms():
    return A.Compose([
            A.Resize(height=1024, width=1024, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
DATA_ROOT_PATH = '../input/global-wheat-detection/test'

class DatasetRetriever(Dataset):

    def __init__(self, image_ids, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        shape = image.shape
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        return shape,image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
dataset = DatasetRetriever(
    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.jpg')]),
    transforms=get_valid_transforms()
)

def collate_fn(batch):
    return tuple(zip(*batch))

data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    drop_last=False,
    collate_fn=collate_fn
)
import torch
import torch.nn as nn
from effdet.anchors import Anchors, AnchorLabeler, generate_detections, MAX_DETECTION_POINTS
from effdet.bench import _post_process
import torch.nn.functional as F
import math
import numpy as np

class DetBenchEvalMultiScale(nn.Module):
    def __init__(self, model, config,multiscale=[]):
        super(DetBenchEvalMultiScale, self).__init__()
        self.config = config
        self.model = model
        self.multiscale = multiscale
        self.anchors = []
        for i,m in enumerate(multiscale):
            self.anchors.append(Anchors(
                config.min_level, config.max_level,
                config.num_scales, config.aspect_ratios,
                config.anchor_scale, int(config.image_size*m)).cuda())
        self.size = config.image_size

    def forward(self, x, image_scales,scale):
        if scale not in self.multiscale:
            print("scale not in 0.5,0.625,0.75,0.875,1.0,1.125,1.25,1.375,1.5")
            return None
        s = np.where(np.array(self.multiscale)==scale)[0][0]
        class_out, box_out = self.model(x)
        class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)

        batch_detections = []
        # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
        for i in range(x.shape[0]):
            detections = generate_detections(
                class_out[i], box_out[i], self.anchors[s].boxes, indices[i], classes[i], image_scales[i])
            batch_detections.append(detections)
        return torch.stack(batch_detections, dim=0)
def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d7')
    net = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = 1
    config.image_size=1024
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    net = torch.nn.DataParallel(net)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['ema_state_dict'])

    del checkpoint
    gc.collect()

    net = DetBenchEvalMultiScale(net, config,[0.875,1.0,1.125,1.25])
    net.eval();
    return net.cuda()

net = load_net('../input/effd7pth/best-checkpoint-027epoch.bin')
import torch.nn.functional as F
import math
def scale_img(img, ratio=1.0, same_shape=False):  # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        gs = 32  # (pixels) grid size
        h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean
def make_tta_predictions(images, score_threshold=0.4024215973594928):
    with torch.no_grad():
        x = torch.stack(images).float().cuda()
        predictions = []
        img_size = x.shape[-2:]  # height, width
        s = [1.0,1.0,1.0,1.25,1.125]  # scales
        y = []
        all_x =[]
        center = []
        for i, xi in enumerate((x,
                                x.flip(3), # only flip-lr
                                x.flip(2),
                                
                                scale_img(x,s[3]),
#                                 scale_img(x.flip(3),s[4]),
#                                 scale_img(x.flip(2),s[5]),
                                
                                scale_img(x,s[4]),
#                                 scale_img(x.flip(3),s[7]),
#                                 scale_img(x.flip(2),s[8])
                                )):
            y.append(net(xi, torch.tensor([1]*xi.shape[0]).float().cuda(),s[i]))

            
        y[1][..., 0] = img_size[1] - y[1][..., 0] - y[1][..., 2] # flip lr
        y[2][..., 1] = img_size[0] - y[2][..., 1] - y[2][..., 3]  # flip ud
        
        y[3][..., :4]/=s[3]
        y[4][..., :4]/=s[4]
#         y[4][..., 0] = img_size[1] - y[4][..., 0] - y[4][..., 2] # flip lr
        

#         y[5][..., :4]/=s[5]
#         y[5][..., 1] = img_size[0] - y[5][..., 1] - y[5][..., 3]  # flip ud
        
        
        
#         y[6][..., :4]/=s[6]
#         y[7][..., :4]/=s[7]
#         y[7][..., 0] = img_size[1] - y[7][..., 0] - y[7][..., 2] # flip lr
#         y[8][..., :4]/=s[8]
#         y[8][..., 1] = img_size[0] - y[8][..., 1] - y[8][..., 3]  # flip ud
        


        
        
        y = np.array(y)
        boxes_all = []
        scores_all=[]
        for j in range(len(y)):
            boxes = y[j][0].cpu().numpy()[:,:4].copy() 
            scores = y[j][0].cpu().numpy()[:,4].copy()
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            boxes_all.append(boxes/1024)
            scores_all.append(scores[indexes])
    return boxes_all,scores_all,y
def run_wbf(boxes,scores, image_index, image_size=1024, iou_thr=0.40813015995026714, skip_box_thr=0.4133694759009949, weights=[0.6,0.2,0.2],name='nms'):
    labels = [np.ones([row.shape[0]]) for row in scores]
    if name=='wbf':
        boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    else:
        boxes, scores, labels = ensemble_boxes.nms(boxes, scores, labels, weights=None, iou_thr=iou_thr)
#     boxes, scores, labels = ensemble_boxes.soft_nms(boxes, scores, labels, weights=None, iou_thr=iou_thr,thresh=skip_box_thr,sigma=0.5)

    boxes = boxes*(image_size-1)
    return boxes, scores, labels
def norm(img):
    img = img.astype(float)
    img-=img.min()
    img/=img.max()
    return img
import matplotlib.pyplot as plt
plt.figure(figsize=[20,20])
for j, (shape,images, image_ids) in enumerate(data_loader):
    h,w,_ = shape[0]
    w_factor = w/1024
    h_factor = h/1024

    boxes_all,scores_all,y = make_tta_predictions(images,score_threshold= 0.4024215973594928)
    i = 0
    sample = norm(images[i].permute(1,2,0).cpu().numpy())

    boxes, scores, labels = run_wbf(boxes_all,scores_all, image_index=i,name='wbf',iou_thr=0.43312889428044965, skip_box_thr= 0.393358169307333,
                                    weights=[0.4,0.05,0.05,0.25,0.25])
    boxes = boxes.round().clip(min=0, max=1023)
    boxes[:, [0,2]]*=w_factor
    boxes[:, [1,3]]*=h_factor
    boxes = boxes.astype(np.int32)
#     indexes = np.where(scores > 0.1)[0]
#     boxes = boxes[indexes]
    plt.subplot(3,3,j+1)
    for box in boxes:
        cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1, 1, 1), 3)

    plt.imshow(sample)
    if j==8:
        break
plt.show()
def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)
results = []
flag = True
csv_pseudo = []
for shape,images, image_ids in data_loader:
    boxes_all,scores_all,y = make_tta_predictions(images,score_threshold=0.4024215973594928)
    for i, image in enumerate(images):
        h,w,_ = shape[i]
        w_factor = float(w)/1024.0
        h_factor = float(h)/1024.0
        try:
            boxes, scores, labels = run_wbf(boxes_all,scores_all, image_index=i,name='wbf',iou_thr=0.40813015995026714, skip_box_thr=0.4133694759009949,
                                            weights=[0.4,0.05,0.05,0.25,0.25])
        except Exception as e:
            boxes, scores, labels = run_wbf(boxes_all,scores_all, image_index=i,name='wbf',iou_thr=0.395, skip_box_thr=0.393)
            print(e)
#         indexes = np.where(scores > 0.1)[0]
#         boxes = boxes[indexes]
        boxes = boxes.round().clip(min=0, max=1023)
        boxes[:, [0,2]]*=w_factor
        boxes[:, [1,3]]*=h_factor
        image_id = image_ids[i]
        boxes=boxes.astype(np.int32)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        df = pd.DataFrame(boxes,columns=['x','y','w','h'])
        df['image_id'] = image_id
        df['source'] = 'test'
        csv_pseudo.append(df)
df_pseudo = pd.concat(csv_pseudo)[['image_id','x','y','w','h','source']]
df_pseudo.to_csv("/kaggle/working/pseudo.csv",index=False)
df_pseudo.to_csv("/kaggle/working/train_spike_kaggle.csv",index=False)
df_pseudo.head()
# sample = plt.imread("../input/global-wheat-detection/test/2fd875eaa.jpg")
# bbox = df_pseudo[df_pseudo.image_id=='2fd875eaa'][['x','y','w','h']].values
# for box in bbox:
#         cv2.rectangle(sample, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (1, 1, 1), 3)
# plt.imshow(sample)
!mkdir data
!mkdir data/train
!cp -r ../input/global-wheat-detection/train/* ./data/train
!cp -r ../input/global-wheat-detection/test/* ./data/train
!cp -r ../input/global-wheat-detection/train.csv ./
import train_utils
import datasets_utils
import sys
from tqdm.auto import tqdm
import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
#import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
import math
import torchvision
import argparse
import pprint

class Config:
    root_data = '/kaggle/working/data/train/'
    csv = '/kaggle/working/train.csv'
    pseudo_csv = '/kaggle/working/pseudo.csv'
    fold_number = 2
    num_workers = 4
    batch_size = 2
    grad_step = 1
    n_epochs = 2
    optimizer = torch.optim.SGD #torch.optim.AdamW
    lr = 0.002
    SchedulerClass = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_params = dict(
        T_max=800,
        )
    verbose = True
    verbose_step = 1
    TrainMultiScale = [1.0]
    print(TrainMultiScale)
    net_name = 'tf_efficientdet_d7'
    checkpoint_name = '../efficientdet/tf_efficientdet_d7_53-6d1d7a95.pth'
    CocoFormat=False
class data_config:
    real = 0.3
    mosaic = 0.0
    cutmix = 0.00
    stylized = 0.0
    scale = 0.2
    hsv = 0.2

if __name__=="__main__":
    class opt:
        fold=2
        epochs=2
        resume = '../input/effd7pth/best-checkpoint-027epoch.bin'
        train=0 if not PSEUDO else 1
    config = Config()
    config.fold_number = opt.fold
    config.resume = opt.resume
    folder = 'effdet7_pseudo'
    if not os.path.exists(folder):
        os.makedirs(folder)
    config.folder = folder
    config.n_epochs = opt.epochs
    if not config.CocoFormat:
        marking,df_folds,spike = train_utils.get_k_fols(config)
        pseudo_csv = pd.read_csv(config.pseudo_csv)
        index_pseudo = list(set(pseudo_csv.image_id))
        train_csv = pd.concat([marking,pseudo_csv])
        image_ids=list(df_folds[df_folds['fold'] != config.fold_number].index.values)
        image_ids = image_ids+index_pseudo+index_pseudo
        if opt.train==0:
            image_ids = image_ids[0:10]
        
        train_dataset = datasets_utils.train_wheat(image_ids=np.array(image_ids),
                                    marking=train_csv,
                                    data_config = data_config,
                                    transforms=datasets_utils.get_train_transforms(),
                                    test=False,
                                    TRAIN_ROOT_PATH=config.root_data)

        validation_dataset = datasets_utils.DatasetRetrieverTest(image_ids=df_folds[df_folds['fold'] == config.fold_number].index.values,
                                                  marking=marking,
                                                  transforms=datasets_utils.get_valid_transforms(),
                                                  test=True,
                                                  TRAIN_ROOT_PATH=config.root_data)


    train_loader = torch.utils.data.DataLoader(
                                                    train_dataset,
                                                    batch_size=config.batch_size,
                                                    sampler=RandomSampler(train_dataset),
                                                    pin_memory=False,
                                                    drop_last=True,
                                                    num_workers=config.num_workers,
                                                    collate_fn=datasets_utils.collate_fn
                                                   )
    validation_loader = torch.utils.data.DataLoader(
                                                    validation_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=2,
                                                    drop_last=False,
                                                    collate_fn=datasets_utils.collate_fn
                                                    )
    if len(config.TrainMultiScale)==1:
        net = train_utils.get_net(type_net=config.net_name,checkpoint_name=config.checkpoint_name,resume = config.resume)
    else:
        print(config.TrainMultiScale)
        net = train_utils.get_net_multiscle(type_net=config.net_name,checkpoint_name=config.checkpoint_name,resume = config.resume,multiScale=config.TrainMultiScale)
    net.cuda()
    fitter = train_utils.Fitter(model=net, config=config)
    fitter.fit(train_loader, validation_loader)
torch.cuda.empty_cache()
net = load_net('/kaggle/working/effdet7_pseudo/last-checkpoint.bin')
from sklearn.model_selection import StratifiedKFold
from skopt import gp_minimize, forest_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_objective, plot_evaluations, plot_convergence, plot_regret
from skopt.space import Categorical, Integer, Real
import os
def make_tta_predictions_val(net,images, score_threshold=0.0):
    with torch.no_grad():
        x = torch.stack(images).float().cuda()
        predictions = []
        img_size = x.shape[-2:]  # height, width
        s = [0.83, 0.67]  # scales
        y = []
        all_x =[]
        for i, xi in enumerate((x,
                                x.flip(3), # only flip-lr
                                x.flip(2),
#                                 x.flip(2).flip(3),  # only flip-up
                                )):
            xp = xi
            all_x.append(xp)
            y.append(net(xp, torch.tensor([1]*xp.shape[0]).float().cuda(),1.0))
            
        y[1][..., 0] = img_size[1] - y[1][..., 0] - y[1][..., 2] # flip lr
        y[2][..., 1] = img_size[0] - y[2][..., 1] - y[2][..., 3]  # flip ud
#         y[3][..., 1] = img_size[0] - y[3][..., 1] - y[3][..., 3]  # flip ud
#         y[3][..., 0] = img_size[1] - y[3][..., 0] - y[3][..., 2] # flip lr
        
        
        y = np.array(y)
        boxes_all = []
        scores_all=[]
        for j in range(len(y)):
            boxes = y[j][0].cpu().numpy()[:,:4].copy() 
            scores = y[j][0].cpu().numpy()[:,4].copy()
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]

            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            boxes = boxes.round().clip(min=0, max=1023)
            boxes_all.append(boxes/1024)
            scores_all.append(scores[indexes])
    return boxes_all,scores_all,y
if True:
    all_predictions = []
    for images, targets, image_ids in tqdm(validation_loader, total=len(validation_loader)):
        with torch.no_grad():
            fold_predictions = {}
            for fold_number in range(2,3):
#                 all_bbox,all_score,_ = make_tta_predictions(net,images,0.0)
                all_bbox,all_score,_ = make_tta_predictions(images,0.0)

            for i in range(1):
                image_predictions = {
                    'image_id': image_ids[i],
                    'gt_boxes': (targets[i]['boxes'].cpu().numpy()).clip(min=0, max=1023).astype(int),
                }
                for fold_number in range(2,3):
                    image_predictions[f'pred_boxes_fold{fold_number}'] = all_bbox
                    image_predictions[f'scores_fold{fold_number}'] = all_score

                all_predictions.append(image_predictions)
        if not PSEUDO:
            break
from test_utils import *
def calculate_final_score(
    all_predictions,
    iou_thr,
    skip_box_thr=0.0,
    score_threshold = 0.2,
    method='nms',
    sigma=0.5,weights=None
):
    final_scores = []

    for i in tqdm(range(len(all_predictions))):
        gt_boxes = all_predictions[i]['gt_boxes'].copy().astype(float)/1024
        image_id = all_predictions[i]['image_id']
        folds_boxes, folds_scores, folds_labels = [], [], []
        for fold_number in range(2,3):
            folds_boxes = all_predictions[i][f'pred_boxes_fold{fold_number}'].copy()[0:3]
            folds_scores = all_predictions[i][f'scores_fold{fold_number}'].copy()[0:3]
#             folds_labels = [np.ones([row.shape[0]]) for row in folds_scores]
        folds_boxes_new = []
        folds_scores_new = []
        for bb in range(len(folds_boxes)):
            s = folds_scores[bb].copy()
            b = folds_boxes[bb].copy()
            indexes = np.where(s > score_threshold)[0]
            new_bbox = b[indexes]
            new_score = s[indexes]
            folds_boxes_new.append(new_bbox)
            folds_scores_new.append(new_score)
        folds_boxes = folds_boxes_new
        folds_scores = folds_scores_new
        folds_labels = [np.ones([row.shape[0]]) for row in folds_scores_new]
        if method == 'weighted_boxes_fusion':
            boxes, scores, labels = weighted_boxes_fusion(folds_boxes, folds_scores, folds_labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        elif method == 'nms':
            try:
                boxes, scores, labels = nms(folds_boxes, folds_scores, folds_labels, weights=weights, iou_thr=iou_thr)
            except:
                boxes, scores, labels = weighted_boxes_fusion(folds_boxes, folds_scores, folds_labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        elif method == 'soft_nms':
            boxes, scores, labels = soft_nms(folds_boxes, folds_scores, folds_labels, weights=weights, iou_thr=iou_thr, thresh=skip_box_thr, sigma=sigma)
        elif method == 'non_maximum_weighted':
            boxes, scores, labels = non_maximum_weighted(folds_boxes, folds_scores, folds_labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        else:
            raise
        image_precision = calculate_image_precision(gt_boxes, boxes, thresholds=iou_thresholds, form='pascal_voc')
        final_scores.append(image_precision)

    return np.mean(final_scores)
from ensemble_boxes import *
print('[WBF]: ', calculate_final_score(
        all_predictions, 
        iou_thr=0.43312889428044965,
        skip_box_thr=0.393358169307333,
        score_threshold = 0.4024215973594928,
        method='weighted_boxes_fusion',weights=None
    ))
def log(text):
    with open('opt.log', 'a+') as logger:
        logger.write(f'{text}\n')

def optimize(space, all_predictions, method, n_calls=10):
    @use_named_args(space)
    def score(**params):
        print('-'*5 + f'{method}' + '-'*5)
        print(params)
        final_score = calculate_final_score(all_predictions, method=method, **params)
        print(f'final_score = {final_score}')
        print('-'*10)
        return -final_score

    return gp_minimize(func=score, dimensions=space, n_calls=n_calls)
space = [
    Real(0.3, 0.75, name='iou_thr'),
    Real(0.2, 0.5, name='score_threshold'),
    Real(0.3, 0.6, name='skip_box_thr'),
#     Real(0.4, 0.9, name='x1'),
#     Real(0.1, 0.5, name='x2'),
#     Real(0.1, 0.5, name='x3'),
#     Real(0.25, 0.55, name='th1'),
#     Real(0.25, 0.55, name='th2'),
#     Real(0.25, 0.55, name='th3'),
]

if True:
    opt_result = optimize(
        space, 
        all_predictions,
        method='weighted_boxes_fusion',
        n_calls=10 if not PSEUDO else 50,
    )
best_iou_thr = opt_result.x[0]
best_score_thr = opt_result.x[1]
best_skip_box_thr = opt_result.x[2]
import matplotlib.pyplot as plt
plt.figure(figsize=[20,20])
for j, (shape,images, image_ids) in enumerate(data_loader):
    h,w,_ = shape[0]
    w_factor = w/1024
    h_factor = h/1024

    boxes_all,scores_all,y = make_tta_predictions(images,score_threshold=best_score_thr)
    i = 0
    sample = norm(images[i].permute(1,2,0).cpu().numpy())

    boxes, scores, labels = run_wbf(boxes_all,scores_all, image_index=i,name='wbf',iou_thr=best_iou_thr, skip_box_thr=best_skip_box_thr,weights=[0.4,0.05,0.05,0.25,0.25])
    boxes = boxes.round().clip(min=0, max=1023)
    boxes[:, [0,2]]*=w_factor
    boxes[:, [1,3]]*=h_factor
    boxes = boxes.astype(np.int32)
    plt.subplot(3,3,j+1)
    for box in boxes:
        cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1, 1, 1), 3)

    plt.imshow(sample)
    if j==8:
        break
plt.show()

results = []
flag = True
for shape,images, image_ids in data_loader:
    boxes_all,scores_all,y = make_tta_predictions(images,score_threshold=best_score_thr)
    for i, image in enumerate(images):
        h,w,_ = shape[i]
        w_factor = float(w)/1024.0
        h_factor = float(h)/1024.0
        try:
            boxes, scores, labels = run_wbf(boxes_all,scores_all, image_index=i,name='wbf',iou_thr=best_iou_thr, skip_box_thr=best_skip_box_thr,
                                    weights=[0.4,0.05,0.05,0.25,0.25])
        except Exception as e:
            print(e)
            try:
                boxes, scores, labels = run_wbf(boxes_all,scores_all, image_index=i,name='wbf',iou_thr=0.43, skip_box_thr=0.4)
            except Exception as e:
                print(e)
                boxes, scores = boxes[0]*1023, scores[0]
            

        boxes = boxes.round().clip(min=0, max=1023)
        boxes[:, [0,2]]*=w_factor
        boxes[:, [1,3]]*=h_factor
        image_id = image_ids[i]
        boxes=boxes.astype(np.int32)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        if flag:
            plt.figure(figsize=[10,10])
            sample = norm(image.permute(1,2,0).cpu().numpy())
            for box in boxes:
                cv2.rectangle(sample, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (1, 0, 0), 1)
            plt.imshow(sample)
            plt.show()
            flag=False
        
        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }
        results.append(result)
!rm -rf *
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)
test_df.head()

