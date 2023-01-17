!pip install albumentations
!pip install --no-deps '../input/pycocotools/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl' > /dev/null

import sys
sys.path.insert(0,'../input/effdet-pytorch') #add packages to system path to allow import
sys.path.insert(0,'../input/torch-img-model')
sys.path.insert(0,'../input/omegaconf')
import numpy as np 
import pandas as pd 
import torch
import os
from glob import glob
import random
from tqdm.notebook import tqdm
import cv2
import albumentations as A
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import StratifiedKFold
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
TRAIN_PATH = "../input/global-wheat-detection/train/"
IMG_SIZE = 512

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    random.seed(seed)  #set fixed value for python built-in pseudo-random generator
    np.random.seed(seed) # for numpy pseudo-random generator
    torch.manual_seed(seed) # pytorch (both CPU and CUDA)
    
set_seed(2020)
train_csv = pd.read_csv('../input/global-wheat-detection/train.csv')

print("Shape of train csv: ", train_csv.shape)
print("Number of distinct img in data: ", train_csv['image_id'].nunique())
train_csv.head()
# split bbox 
bbox = np.stack(train_csv['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, col in enumerate(['x','y','w','h']):
    train_csv[col] = bbox[:,i]
train_csv.drop(columns=['bbox'], inplace=True)
skf = StratifiedKFold(n_splits=5, shuffle=True)

df_folds = train_csv[['image_id']].copy()
df_folds['bbox_count'] = 1
df_folds = df_folds.groupby('image_id').count() #num bbox for each img_id
df_folds['source'] = train_csv[['image_id', 'source']].groupby('image_id').min()['source'] #get source from each img_id
df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['source'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
)
df_folds.loc[:, 'fold'] = 0

for fold_num, (train_idx, val_idx) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_idx].index,'fold'] = fold_num
class GlobalConfig:
    num_workers = 2
    batch_size = 4
    n_epochs = 2
    lr = 1e-4
    
    verbose = 1
    verbose_step = 1
    
    folder = 'effdet_train'
    
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss
    
    
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min', #lr reduced when monitored quantity stopped decreasing
        factor=0.5,
        patience=1,
        threshold_mode='abs',
        min_lr=1e-8
    )
class WheatData(Dataset):
    def __init__(self, df, img_ids, transform=None, test=False):
        super().__init__()
        self.df = df
        self.img_ids = img_ids
        self.transform = transform
        self.test = test
        
    def __getitem__(self, index:int):
        img_id = self.img_ids[index]
    
        if self.test or random.random() > 0.5:
            image, boxes = self.load_image_and_boxes(index)
        else:
            image, boxes = self.load_cutmix_image_and_boxes(index)
            
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['img_scale'] = torch.tensor([1.])
        target['image_id'] = torch.tensor([index])
        target['img_size'] = torch.tensor([(IMG_SIZE, IMG_SIZE)])
        
        if self.transform:
            for i in range(10):
                sample = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                }
                sample = self.transform(**sample)
                
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    #print(sample['bboxes'])
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    #print(target['boxes'])
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx format that is compatible with the model requirements
                    break
                    
        return image, target, img_id
    
    def __len__(self):
        return self.img_ids.shape[0]
    
    
    def load_image_and_boxes(self, index):
        image_id = self.img_ids[index]
        image = cv2.imread(f'{TRAIN_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.df[self.df['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes
    
    
    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.img_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        return result_image, result_boxes
def train_transforms():
    return A.Compose([
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
                    ],p=0.9),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0)], p=1.0, 
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
        )
    )

def valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        )
    )
fold_number = 0

def collate_fn(batch):
    return tuple(zip(*batch))

# Instantiate dataset class
train_dataset = WheatData(
    img_ids=df_folds[df_folds['fold'] != fold_number].index.values,
    df=train_csv,
    transform=train_transforms(),
    test=False)

valid_dataset = WheatData(
    img_ids=df_folds[df_folds['fold'] == fold_number].index.values,
    df=train_csv,
    transform=valid_transforms(),
    test=True)


# Create dataloader
train_loader = DataLoader(train_dataset,
                         batch_size = GlobalConfig.batch_size,
                         collate_fn = collate_fn,
                         shuffle=True)

valid_loader = DataLoader(valid_dataset,
                         batch_size = GlobalConfig.batch_size,
                         collate_fn = collate_fn)
image, target, image_id = train_dataset[0]
boxes = target['boxes'].cpu().numpy().astype(np.int64)

numpy_image = image.permute(1,2,0).cpu().numpy()

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(numpy_image, (box[1], box[0]), (box[3],  box[2]), (0, 1, 0), 2)
    
ax.set_axis_off()
ax.imshow(numpy_image);
class AverageMeter():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
import effdet
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

def get_net():
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load('../input/efficientdet-model/eff_det_models/tf_efficientdet_d5-ef44aea8.pth') #d3-d7 ('efficientdet_model' folder) 
    net.load_state_dict(checkpoint)
    config.num_classes = 1
    config.image_size = 512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)

net = get_net()
class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.model = model
        self.device = device
        
        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.epoch = 0
        self.best_summary_loss = 10**5
        self.log_path = f'{self.base_dir}/log.txt'

        param_optimizer = list(self.model.named_parameters())

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        
        self.log("Begin training with {}".format(self.device))
    
    
    def fit(self, train_loader, valid_loader):
        for i in range(self.config.n_epochs):
            summary_loss = self.train_epoch(train_loader)
            self.log(f'[TRAINING] Epoch {self.epoch}, Loss : {summary_loss.avg}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            summary_loss = self.validation(valid_loader)
            self.log(f'[VALIDATION] Epoch {self.epoch}, Loss : {summary_loss.avg}')

            if self.best_summary_loss > summary_loss.avg:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(2)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1
            
    
    def validation(self, valid_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        for steps, (images, targets, image_ids) in enumerate(valid_loader):
            with torch.no_grad():
                pred_res = {}
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]
                img_scale = torch.tensor([target['img_scale'].to(self.device) for target in targets])
                img_size = torch.tensor([(IMG_SIZE, IMG_SIZE) for target in targets]).to(self.device).float()
                
                pred_res['bbox'] = boxes
                pred_res['cls'] = labels
                pred_res['img_scale'] = img_scale
                pred_res['img_size'] = img_size

                outputs = self.model(images, pred_res)
                loss = outputs['loss']
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss
    
    
    def train_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        for images, targets, image_ids in tqdm(train_loader):
            target_res = {}
            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]
            img_scale = torch.tensor([target['img_scale'] for target in targets]).to(self.device).float()
            img_size = torch.tensor([(IMG_SIZE, IMG_SIZE) for target in targets]).to(self.device).float()
            
            target_res['bbox'] = boxes
            target_res['cls'] = labels
            target_res['img_scale'] = img_scale
            target_res['img_size'] = img_size

            self.optimizer.zero_grad()
            
            outputs = self.model(images, target_res)
            loss = outputs['loss']
            
            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

        return summary_loss
    
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)
        
        
#     def load(self, path):
#         checkpoint = torch.load(path)
#         self.model.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         self.best_summary_loss = checkpoint['best_summary_loss']
#         self.epoch = checkpoint['epoch'] + 1
        
    
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def training():
    net.to(device)
    
    fitter = Fitter(model=net, device=device, config=GlobalConfig)
    fitter.fit(train_loader, valid_loader)
training()
sys.path.insert(0, "../input/weightedboxfusion")

import gc
from effdet import DetBenchPredict
from ensemble_boxes import *

TEST_PATH = "../input/global-wheat-detection/test/"
class WheatData(Dataset):
    def __init__(self, img_ids, transform=None):
        self.img_ids = img_ids
        self.transform = transform
        
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image = cv2.imread(f'{TEST_PATH}/{img_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image /255.0
        
        if self.transform:
            sample = {'image' : image}
            sample = self.transform(**sample)
            image = sample['image']
        
        target = {}
        target['img_scale'] = torch.tensor([1.])
            
        return image, img_id, target
        
    def __len__(self) -> int: #annotate parameters with their expected type
        return self.img_ids.shape[0]
def valid_transform():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0)], 
            p=1.0)


def collate_fn(batch):
    return tuple(zip(*batch))


test_dataset = WheatData(
    img_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{TEST_PATH}/*.jpg')]),
    transform=valid_transform())

test_loader = DataLoader(test_dataset,
                         batch_size = 4,
                         shuffle = False,
                         drop_last = False,
                         collate_fn = collate_fn) 
def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    
    config.num_classes = 1
    config.image_size = 512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, 
                            norm_kwargs=dict(eps=.001, momentum=.01))
    
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    
    del checkpoint
    gc.collect()
    
    net = DetBenchPredict(net, config)
    net.eval()
    
    return net.cuda()

# load
net = load_net('./effdet_train/best-checkpoint-01epoch.bin')
class BaseWheatTTA:
    """ author: @shonenkov """
    image_size = 512

    def augment(self, image):
        raise NotImplementedError
    
    def batch_augment(self, images):
        raise NotImplementedError
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class TTAHorizontalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)
    
    def batch_augment(self, images):
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return boxes

class TTAVerticalFlip(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return image.flip(2)
    
    def batch_augment(self, images):
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]
        return boxes
    
class TTARotate90(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]
        res_boxes[:, [1,3]] = boxes[:, [2,0]]
        return res_boxes

class TTACompose(BaseWheatTTA):
    """ author: @shonenkov """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image
    
    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images
    
    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes
    
    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)
def run_wbf(predictions, image_index, image_size=512, iou_thr=0.44, 
            skip_box_thr=0.43, weights=None):
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist() for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in predictions]
    boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels
def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], 
                                                             j[1][2], j[1][3]))
    return " ".join(pred_strings)
from itertools import product

tta_transforms = []
for tta_combination in product([TTAHorizontalFlip(), None], 
                               [TTAVerticalFlip(), None],
                               [TTARotate90(), None]):
    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
# WBF over TTA

def predict(images, target, score_thres=0.25):
    with torch.no_grad():
        prediction = []
        images = torch.stack(images).to(device).float()
        img_scale = torch.tensor([target['img_scale'].to(device) for target in targets])
        img_size = torch.tensor([(IMG_SIZE, IMG_SIZE) for target in targets]).to(device)

        '''

        Within the forward function of the DetBenchPredict class, it takes in 3 arguments (image, image_scale, image_size)
        The return object is as follows: 
        detections = torch.cat([boxes, scores, classes.float()], dim=1) 
        where the first 4 col will be the bboxes, 5th col the scores
        Find out more at https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/bench.py

        '''

        for tta_transform in tta_transforms:
            result = []
            det = net(tta_transform.batch_augment(images.clone()),
                      img_scales = img_scale,
                      img_size = img_size)

            for i in range(images.shape[0]):
                boxes = det[i].detach().cpu().numpy()[:,:4]    
                scores = det[i].detach().cpu().numpy()[:,4]
                indexes = np.where(scores > score_thres)[0]
                boxes = boxes[indexes]
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                boxes = tta_transform.deaugment_boxes(boxes.copy())
                result.append({
                    'boxes': boxes,
                    'scores': scores[indexes],
                })

            prediction.append(result)

    return prediction
results = []
for images, image_ids, targets in test_loader:
    predictions = predict(images, targets)
    for i, image in enumerate(images):
        boxes, scores, labels = run_wbf(predictions, image_index=i)
        boxes = (boxes*2).round().astype(np.int32).clip(min=0, max=1023)
        image_id = image_ids[i]
        
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }
        results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)
test_df.head()