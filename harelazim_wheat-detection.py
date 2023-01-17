import cv2

from matplotlib import pyplot as plt

import numpy as np

import albumentations as A

from albumentations.pytorch.transforms import ToTensor,ToTensorV2

import glob

from sklearn.model_selection import train_test_split

import torchvision.models as models

import pandas as pd



#sklearn

from sklearn.model_selection import StratifiedKFold





from torch.utils.data import Dataset, DataLoader

import torch

import torch.nn as nn

import numpy as np

import torch.nn.functional as F
import time

import random

import os



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(42)

df=pd.read_csv("../input/global-wheat-detection/train.csv")

df.head()

df['image_id']=df['image_id']+'.jpg'

bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

for i, column in enumerate(['x', 'y', 'w', 'h']):

    df[column] = bboxs[:,i]

df.drop(columns=['bbox'], inplace=True)

df.head()
BOX_COLOR = (255, 0, 0)

TEXT_COLOR = (255, 255, 255)



def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):

    x_min, y_min, w, h = bbox

    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    class_name = class_idx_to_name[class_id]

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    

    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)

    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)

    return img





def visualize(annotations, category_id_to_name,img2):

    img = annotations['image'].copy()

    for idx, bbox in enumerate(annotations['bboxes']):

      img = visualize_bbox(img, bbox, annotations['labels'][idx], category_id_to_name)

    plt.figure(figsize=(15, 15))

    plt.subplot(1,2,1)

    print(type(img))

    plt.imshow(img)

    plt.subplot(1,2,2)

    plt.imshow(img2)

    plt.show()



def get_aug(aug, min_area=0., min_visibility=0.):

    return Compose(aug, bbox_params=BboxParams(format='coco', min_area=min_area, 

                                               min_visibility=min_visibility, label_fields=['labels']))
aug=[A.VerticalFlip(p=1),A.HorizontalFlip(p=1),A.Rotate(p=1),

     A.RandomSunFlare(p=1,flare_roi=(0.3, 0.3, 0.5, 0.7),

                     num_flare_circles_lower=2, num_flare_circles_upper=5, src_radius=50),

     A.OneOf([

             A.RandomBrightness(p=1,limit=0.3),

           A.RandomBrightnessContrast(p=1,brightness_limit=0.3, contrast_limit=0.2),

           A.RandomContrast(p=1,limit=0.1)

           ],p=1),

      ]



transform=A.Compose(aug, bbox_params=A.BboxParams(format='coco', label_fields=['labels']))



image_ids = df['image_id'].unique()

random.shuffle(image_ids)

valid_ids = image_ids[-500:]

train_ids = image_ids[:-500]



for i in range(0,len(valid_ids),15):

    p=valid_ids[i]

    img=cv2.imread('../input/global-wheat-detection/train/'+p)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #img=image1

    record=df[df['image_id'] == p]

    box=record[['x', 'y', 'w', 'h']].values

    img2=img.copy()



    sample={'image': img,'bboxes': box,'labels': [1]*len(box)}

    category_id_to_name = {0: '0',1:'100%'}

    annotation=transform(**sample)

    box=annotation['bboxes']



    visualize(annotation,category_id_to_name,img2)
#AverageMeter - class for averaging loss,metric

class AverageMeter(object):

    """Computes and stores the average and current value"""

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






class WheatDataset(Dataset):



  def __init__(self, path,df, transforms=None):

    self.path=path

    self.df=df

    self.transforms=transforms



  def __len__(self):

    return len(self.path)



  def __getitem__(self, idx):

    p=self.path[idx]

    

    image1=cv2.imread('../input/global-wheat-detection/train/'+p,cv2.IMREAD_COLOR)

    image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB).astype(np.float32)

    image1 /= 255.0

    



    records = self.df[self.df['image_id'] == p]

    boxes = records[['x', 'y', 'w', 'h']].values

    area = boxes[:,2]*boxes[:,3]

    area = torch.as_tensor(area, dtype=torch.float32)

    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    boxes = torch.as_tensor(boxes, dtype=torch.float32)



    target={}

    labels = torch.ones((len(boxes),), dtype=torch.int64)

    iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

    

    if self.transforms:

      sample = {

        'image': image1,

        'bboxes': boxes,

        'labels': labels

      }

      sample=self.transforms(**sample)

      image1 = sample['image']

      boxes  = sample['bboxes']

      labels = sample['labels']





    target['boxes']=torch.as_tensor(boxes, dtype=torch.float32)

    target["iscrowd"]=torch.as_tensor(iscrowd,dtype=torch.int64 )

    target['labels'] =  torch.as_tensor(labels,dtype=torch.int64 )

    target['area'] = area

    target["image_id"] = torch.tensor([idx])

    image1=torch.as_tensor(image1,dtype=torch.float32)

    return image1,target
category_id_to_name = {0: '0',1:'100%'}



image_ids = df['image_id'].unique()

random.shuffle(image_ids)

valid_ids = image_ids[-500:]

train_ids = image_ids[:-500]



valid_df = df[df['image_id'].isin(image_ids)]

train_df = df[df['image_id'].isin(image_ids)]



#Cutout  RandomShadow CLAHE



aug=[A.VerticalFlip(p=0.5),A.HorizontalFlip(p=0.5),A.Rotate(p=0.5),

          A.RandomSunFlare(p=0.5,flare_roi=(0.3, 0.3, 0.5, 0.7),

                     num_flare_circles_lower=2, num_flare_circles_upper=5, src_radius=50),

     A.OneOf([

             A.RandomBrightness(p=0.5,limit=0.3),

           A.RandomBrightnessContrast(p=0.5,brightness_limit=0.3, contrast_limit=0.2),

       A.RandomContrast(p=0.5,limit=0.1)

           ],p=1),

      ToTensorV2(p=1.0)]



bbox=A.BboxParams(format='pascal_voc',label_fields=['labels'])

transform=A.Compose(aug, bbox_params=bbox,p=1)



transformValid=A.Compose([ ToTensorV2(p=1.0)], bbox_params=bbox,p=1)



def collate_fn(batch):

    return tuple(zip(*batch))



batch_s=4

dataset_train=WheatDataset(train_ids,df,transforms=transform)

train_loader=DataLoader(dataset_train,batch_size=batch_s,shuffle=True,collate_fn=collate_fn)



dataset_val=WheatDataset(valid_ids,df,transforms=transformValid)

val_loader=DataLoader(dataset_val,batch_size=4,collate_fn=collate_fn,shuffle=True)

len(train_ids),len(valid_ids)
from tqdm import tqdm, notebook



def train_detr(epoch_number,model,optim,criterion,train_losses):

  model.train()

  criterion.train()

  train_losses.reset()



  tqdm_loader=tqdm(train_loader_detr)

  for  index,(img,target) in enumerate(tqdm_loader):

    img = list(image.to(device) for image in img)

    target = [{k: v.to(device) for k, v in t.items()} for t in target]





    out=model(img)

    loss_dict = criterion(out, target)

    weight_dict = criterion.weight_dict



    loss1 =sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    

    optim.zero_grad()

    # track the loss

    loss1.backward()

    optim.step()

    train_losses.update(loss1.item(),n=batch_s)



    tqdm_loader.set_description("Epoch {} loss item ={:4} avg={:4} ".format(epoch_number,round(loss1.item(),4),round(train_losses.avg,4)))



def train(epoch_number,model,optim,train_losses):

  model.train()

  train_losses.reset()

  tqdm_loader=tqdm(train_loader)

  for  index,(img,target) in enumerate(tqdm_loader):

    img = list(image.to(device) for image in img)

    target = [{k: v.to(device) for k, v in t.items()} for t in target]



    out=model(img,target)

    loss1 = sum(loss1 for loss1 in out.values())



    # track the loss

    optim.zero_grad()

    loss1.backward()

    optim.step()



    train_losses.update(loss1.item(),n=batch_s)



    tqdm_loader.set_description("Epoch {} loss item ={:4} avg={:4} ".format(epoch_number,round(loss1.item(),4),round(train_losses.avg,4)))
def calculate_iou(gt, pr, form='pascal_voc') -> float:

    if form == 'coco':

        gt = gt.copy()

        pr = pr.copy()



        gt[2] = gt[0] + gt[2]

        gt[3] = gt[1] + gt[3]

        pr[2] = pr[0] + pr[2]

        pr[3] = pr[1] + pr[3]



    # Calculate overlap area

    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1

    

    if dx < 0:

        return 0.0

    

    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1



    if dy < 0:

        return 0.0



    overlap_area = dx * dy



    # Calculate union area

    union_area = (

            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +

            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -

            overlap_area

    )



    return overlap_area / union_area


def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:

    best_match_iou = -np.inf

    best_match_idx = -1



    for gt_idx in range(len(gts)):

        

        if gts[gt_idx][0] < 0:

            # Already matched GT-box

            continue

        

        iou = -1 if ious is None else ious[gt_idx][pred_idx]



        if iou < 0:

            iou = calculate_iou(gts[gt_idx], pred, form=form)

            

            if ious is not None:

                ious[gt_idx][pred_idx] = iou



        if iou < threshold:

            continue



        if iou > best_match_iou:

            best_match_iou = iou

            best_match_idx = gt_idx



    return best_match_idx



def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:



    n = len(preds)

    tp = 0

    fp = 0

    

    # for pred_idx, pred in enumerate(preds_sorted):

    for pred_idx in range(n):



        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,

                                            threshold=threshold, form=form, ious=ious)



        if best_match_gt_idx >= 0:

            # True positive: The predicted box matches a gt box with an IoU above the threshold.

            tp += 1

            # Remove the matched GT box

            gts[best_match_gt_idx] = -1



        else:

            # No match

            # False positive: indicates a predicted box had no associated gt box.

            fp += 1



    # False negative: indicates a gt box had no associated predicted box.

    fn = (gts.sum(axis=1) > 0).sum()



    return tp / (tp + fp + fn)


def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:



    n_threshold = len(thresholds)

    image_precision = 0.0

    

    ious = np.ones((len(gts), len(preds))) * -1

    # ious = None



    for threshold in thresholds:

        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,

                                                     form=form, ious=ious)

        image_precision += precision_at_threshold / n_threshold



    return image_precision
def val(epoch_number,model,train_losses):

    train_losses.reset()

    tqdm_loader=tqdm(val_loader)

    model.eval()

    iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]

    validation_image_precisions = []

    

    with torch.no_grad():

    

        for step, (images, targets) in enumerate(tqdm_loader):

            

            images = list(image.to(device) for image in images)

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



            outputs = model(images)



            for i, image in enumerate(images):

                boxes = outputs[i]['boxes'].data.cpu().numpy()

                scores = outputs[i]['scores'].data.cpu().numpy()

                gt_boxes = targets[i]['boxes'].cpu().numpy()

                preds_sorted_idx = np.argsort(scores)[::-1]

                preds_sorted = boxes[preds_sorted_idx]

                image_precision = calculate_image_precision(preds_sorted, gt_boxes, thresholds=iou_thresholds,form='pascal_voc')

                validation_image_precisions.append(image_precision)

                train_losses.update(image_precision,n=1)

            tqdm_loader.set_description("Val Epoch {}  avg={:4} ".format(epoch_number,round(np.mean(validation_image_precisions),4)))



    valid_prec = np.mean(validation_image_precisions)

  

    
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained pre-trained on COCO

model1 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)





num_classes = 2  # 1 class (person) + background

# get number of input features for the classifier

in_features = model1.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one

model1.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model1=model1.to(device)

adam=torch.optim.Adam(model1.parameters(),lr=0.0001)

train_losses = AverageMeter()
for i in range(3):

    val(i,model1,train_losses)

    train(i,model1,adam,train_losses)

    

  
#val example

index_img=0

#torch.reshape(img,(3,1024,1024)) val_loader train_loader

images, targets = next(iter(val_loader))

images = list(img.to(device) for img in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

boxes = targets[index_img]['boxes'].cpu().numpy().astype(np.float32)

sample2 = images[index_img].permute(1,2,0).cpu().numpy()

#pred val

model1.eval()

cpu_device = torch.device("cpu")

outputs = model1(images)

outputs=outputs[index_img]

outputs = [{k: v.to(cpu_device) for k, v in outputs.items()} ]
#get box&score to plot 

box1=outputs[0]['boxes'].cpu().detach().numpy()

scores=outputs[0]['scores'].cpu().detach().numpy()

image1=images[index_img].cpu().numpy()

print(len(boxes)," ",len(box1),"  ",scores)

print("boxes : ", sum(scores >0.4))
image1=np.reshape(image1,(1024,1024,3))

plt.subplot(1,2,1)

plt.imshow(image1)

plt.subplot(1,2,2)

plt.imshow(sample2)

plt.show()
image1
for i in range(len(boxes)):

  cv2.rectangle(sample2,(int(boxes[i][0]),int(boxes[i][1])),(int(boxes[i][2]),int(boxes[i][3])),

                (0,255,0),2)



def get_box(box11):

  box_sample=[int(box11[0]),int(box11[1]),int(box11[2]),int(box11[3])]

  box_sample[2]=box_sample[2]-box_sample[0]

  box_sample[3]=box_sample[3]-box_sample[1]

  return box_sample







aug=[]

transform=A.Compose(aug, bbox_params=A.BboxParams(format='coco', min_area=0, 

                                               min_visibility=0, label_fields=['labels']),p=1)

list_box=[]

for i in range(len(box1)):

  if scores[i] >0.5:

    list_box.append(get_box(box1[i]))



sample={'image': sample2,'bboxes': list_box,'labels': [1]*len(list_box)}

category_id_to_name = {0: '0',1:'100%'}

annotation=transform(**sample)

plt.figure(figsize=[12,12])

visualize(annotation,category_id_to_name,sample2)


class WheatDatasetTest(Dataset):



  def __init__(self, path,df, transforms=None):

    self.path=path

    self.df=df

    self.transforms=transforms



  def __len__(self):

    return len(self.path)



  def __getitem__(self, idx):

    p=self.path[idx]

    

    image1=cv2.imread('../input/global-wheat-detection/test/'+p,cv2.IMREAD_COLOR)

    image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB).astype(np.float32)

    image1 /= 255.0

    





    target={}

    labels = torch.ones((len(boxes),), dtype=torch.int64)

    iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

    

    if self.transforms:

      sample = {

        'image': image1,

        'labels':labels

      }

      sample=self.transforms(**sample)

      image1 = sample['image']

      labels = sample['labels']





    target["iscrowd"]=torch.as_tensor(iscrowd,dtype=torch.int64 )

    target['labels'] =  torch.as_tensor(labels,dtype=torch.int64 )



    image1=torch.as_tensor(image1,dtype=torch.float32)

    

    return image1,target,p
test_df=pd.read_csv("../input/global-wheat-detection/sample_submission.csv")

list_path=glob.glob("../input/global-wheat-detection/test/*")

valid_ids=[]

for path in list_path:

    valid_ids.append(path.split('/')[-1])

    

test_dataset = WheatDatasetTest(valid_ids,test_df,transforms=transformValid)



test_data_loader = DataLoader(

    test_dataset,

    batch_size=4,

    shuffle=False

)
valid_ids
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
detection_threshold = 0.5

results = []



for images, target,image_ids in test_data_loader:



    images = list(image.to(device) for image in images)

    outputs = model1(images)



    for i, image in enumerate(images):



        boxes = outputs[i]['boxes'].data.cpu().numpy()

        scores = outputs[i]['scores'].data.cpu().numpy()

        

        boxes = boxes[scores >= detection_threshold].astype(np.int32)

        scores = scores[scores >= detection_threshold]

        image_id = image_ids[i]

        

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        print(image_ids)

        

        result = {

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes, scores)

        }



        

        results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.head()
test_df.to_csv('submission.csv', index=False)