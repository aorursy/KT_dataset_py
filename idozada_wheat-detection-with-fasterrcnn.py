import sys

sys.path.insert(0,"../input/weightedboxesfusion")

import numpy as np 

import pandas as pd 

import cv2

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import torch.utils as utils

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn

import torch.nn.functional as F

import torch

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch.optim

import os

import glob

from tqdm.auto import tqdm

from tqdm import tqdm_notebook

import random

import time

from datetime import datetime

import ensemble_boxes

from ensemble_boxes import *

from itertools import product

%matplotlib inline

from albumentations.pytorch.transforms import ToTensorV2

from albumentations import (Blur, MotionBlur, MedianBlur, GaussianBlur,

                            VerticalFlip, HorizontalFlip, IAASharpen,

                            OneOf, Compose , BboxParams, Resize ,RandomSizedCrop,

                            ToGray , Cutout , HueSaturationValue , RandomBrightnessContrast)
DIR_PATH = '/kaggle/input/global-wheat-detection/'

dir = glob.glob(os.path.join(DIR_PATH , '*'))

dir.sort(reverse=True)

train_paths = glob.glob(os.path.join(dir[1] , '*'))

test_paths = glob.glob(os.path.join(dir[2] , '*'))
df = pd.read_csv(dir[0])

bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

for i, column in enumerate(['x', 'y', 'w', 'h']):

    df[column] = bboxs[:,i]

df.drop(columns=['bbox'], inplace=True)

df.head()
def show_image(image, boxes, title):

  fig, ax = plt.subplots(1, 1, figsize=(25, 8))

  boxes = boxes.astype(np.int32)

  for box in boxes:

      cv2.rectangle(image, (box[0], box[1]), (box[2],  box[3]), (1, 1, 0), 3)

  ax.set_title(title) 

  ax.set_axis_off()

  ax.imshow(image);





def load_image_and_boxes(image_path):

  image_id = image_path.split('/')[-1].split('.')[0]

  image = cv2.imread(image_path, cv2.IMREAD_COLOR)

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

  image /= 255.0

  records = df[df['image_id'] == image_id]

  boxes = records[['x', 'y', 'w', 'h']].values

  boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

  boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

  return image, boxes





def images_after_augmentation(original_image, boxes, augmentation):

    aug_image = original_image.copy()

    boxes = boxes.astype(np.int32)

    if isinstance(augmentation , VerticalFlip) or isinstance(augmentation , HorizontalFlip):

      for box in boxes:

          cv2.rectangle(aug_image, (box[0], box[1]), (box[2],  box[3]), (1, 1, 0), 2)

      sample = {'image': aug_image, 'label': 'label'}

      compose = Compose([augmentation], p=1)

      aug_image = compose(**sample)['image']

    else:

      sample = {'image': aug_image, 'label': 'label'}

      compose = Compose([augmentation], p=1)

      aug_image = compose(**sample)['image']

      for box in boxes:

          cv2.rectangle(aug_image, (box[0], box[1]), (box[2],  box[3]), (1, 1, 0), 2)

    plt.figure(figsize=[12, 12])

    for i in range(len([original_image, aug_image])):

            image = [original_image, aug_image][i]

            plt.subplot(1, 2, i+1)

            plt.title(['Original Image', 'After Augmentaion'][i])

            plt.axis("off")

            plt.imshow(image)

    plt.show()





# Functions to visualize bounding boxes and class labels on an image. 

# Based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py

def visualize(**images):

    """PLot images in one row."""

    n = len(images)

    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(images.items()):

        plt.subplot(1, n, i + 1)

        plt.xticks([])

        plt.yticks([])

        plt.title(' '.join(name.split('_')).title())

        plt.imshow(image)

    plt.show()





BOX_COLOR = (255, 255, 0)





def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=3):

    xmin, ymin, xmax, ymax = bbox

    xmin, ymin, xmax, ymax =  int(xmin), int(ymin), int(xmax), int(ymax)

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=BOX_COLOR, thickness=thickness)

    return img



def visualizeTarget(image, target , visualize_data_loader = True):

  boxes = target['boxes']

  if visualize_data_loader:

    if not type(boxes).__module__ == np.__name__:

      boxes = boxes.numpy()

    image = image.numpy()

    image = np.transpose(image,(1,2,0))

  img = image.copy()

  for idx, bbox in enumerate(boxes):

      img = visualize_bbox(img, bbox)

  return img
image_id = '8425a537b.jpg'

image_path = glob.glob(os.path.join(dir[1] , image_id))

image , boxes  = load_image_and_boxes(image_path[0])

show_image(image, boxes, "Image without bounding box")
image_id = 'b3c96d5ad.jpg'

image_path = glob.glob(os.path.join(dir[1] , image_id))

image , boxes  = load_image_and_boxes(image_path[0])

show_image(image, boxes, "Image with bounding box")
images_after_augmentation(image, boxes, Blur(blur_limit=7 ,p=1))
images_after_augmentation(image, boxes, VerticalFlip(p=1))
images_after_augmentation(image, boxes, HorizontalFlip(p=1))
images_after_augmentation(image, boxes, Cutout(num_holes=8, max_h_size=128, max_w_size=128, fill_value=0, p=1.0))
class WheatDataset(Dataset):

    def __init__(self , paths , dataframe ,  transforms = None):

        super().__init__()

        self.paths = paths

        self.dataframe = dataframe

        self.transforms = transforms

    def __getitem__(self, index):

        path_image = self.paths[index]

        image, boxes , area = self.load_image_and_boxes(index)

       

        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

 

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        target['image_id'] = torch.tensor([index])

        target['area'] = area



        if self.transforms:

            sample = self.transforms(**{

                  'image': image,

                  'bboxes': target['boxes'],

                  'labels': labels

              })

            image = sample["image"]

            target['bboxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)

            target['bboxes'] =  target['bboxes'].reshape(-1, 4)

            target["boxes"] = target["bboxes"]

            del target['bboxes']

        else:

            target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)

            target['boxes'] =  target['boxes'].reshape(-1, 4)

        image = np.transpose(image,(2,0,1))

        image = torch.from_numpy(image)

        return image, target, path_image

    

    def __len__(self):

        return len(self.paths)



    def load_image_and_boxes(self, index):

        path_image = self.paths[index]

        image_id = path_image.split('/')[-1].split('.')[0]

        image = cv2.imread(path_image, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        records = self.dataframe[self.dataframe['image_id'] == image_id]

        boxes = records[['x', 'y', 'w', 'h']].to_numpy()

        area = (boxes[:, 2] * boxes[:, 3])

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = torch.as_tensor(area, dtype=torch.float32)

        return image, boxes , area

    

    def load_cutmix_image_and_boxes(self, index, imsize=1024):

        """ 

        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 

        Refactoring and adaptation: https://www.kaggle.com/shonenkov

        """

        w, h = imsize, imsize

        s = imsize // 2

    

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y

        indexes = [index] + [random.randint(0, len(self.paths) - 1) for _ in range(3)]



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

train_path , valid_path = train_test_split(train_paths,test_size=0.2)
transforms_train = Compose([IAASharpen(p = 0.5),RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),ToGray(p=0.01),

                            Cutout(num_holes=8, max_h_size=128, max_w_size=128, fill_value=0, p=0.5),

                            OneOf([HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),

                                    RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2, p=0.9)]),

                            OneOf([Blur(blur_limit=3), MotionBlur(blur_limit=3), MedianBlur(blur_limit=3)]),

                            OneOf([VerticalFlip(), HorizontalFlip()])

                            ],p = 0.9,bbox_params=BboxParams(format='pascal_voc', min_area=0, 

                                               min_visibility=0, label_fields=['labels']))
def collate_fn(batch):

  return tuple(zip(*batch))
bs = 8

num_workers = 8

train_set = WheatDataset(train_path,df , transforms=transforms_train)

valid_set = WheatDataset(valid_path,df)



train_loader = DataLoader(train_set,batch_size = bs ,shuffle = True,collate_fn=collate_fn , num_workers=num_workers)

valid_loader = DataLoader(valid_set,batch_size = bs ,shuffle = True,collate_fn=collate_fn , num_workers=num_workers)



images , targets , path_images = next(iter(train_loader))

img = visualizeTarget(images[0],targets[0])

visualize(Example_one_image_from_dataloader = img)
class Calculator(object):

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
class Training:

    

    def __init__(self, model, device, config):

        self.config = config

        self.epoch = 0



        self.base_dir = f'{config.folder}'

        if not os.path.exists(self.base_dir):

            os.makedirs(self.base_dir)

        

        self.best_calc_loss = 10**5



        self.model = model

        self.device = device



        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)





    def train_loop(self, train_loader, validation_loader):

        for e in range(self.config.n_epochs):

            if self.config.verbose:

                lr = self.optimizer.param_groups[0]['lr']

                timestamp = datetime.utcnow().isoformat()

                print(f'\n{timestamp}\nLR: {lr}')



            t = time.time()

            calc_loss = self.train_one_epoch(train_loader)



            print(f'Train. Epoch: {self.epoch}, Loss: {calc_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            self.save_model(f'{self.base_dir}/last-epoch.bin')



            t = time.time()

            calc_loss = self.valid_one_epoch(validation_loader)



            print(f'Val. Epoch: {self.epoch}, Loss: {calc_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            if calc_loss.avg < self.best_calc_loss:

                self.best_calc_loss = calc_loss.avg

                self.model.eval()

                self.save_model(f'{self.base_dir}/best-model-{str(self.epoch).zfill(2)}epoch.bin')



            if self.config.validation_scheduler:

                self.scheduler.step(metrics=calc_loss.avg)



            self.epoch += 1





    def train_one_epoch(self, train_loader):

      self.model.train()

      calc_loss = Calculator()

      t = time.time()

      for images, targets, image_ids in tqdm_notebook(train_loader):

        batch_size = len(images)

        images = list(image.to(device).float() for image in images)

        target_res = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, target_res)

        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        self.optimizer.zero_grad()

        losses.backward()

        self.optimizer.step()

        calc_loss.update(loss_value, batch_size)



        if self.config.step_scheduler:

          self.scheduler.step()

      

      return calc_loss





    def valid_one_epoch(self, val_loader):

        self.model.train()

        calc_loss = Calculator()

        t = time.time()

        for images, targets, image_ids in tqdm_notebook(val_loader):

            with torch.no_grad():

                batch_size = len(images)

                images = list(image.to(device).float() for image in images)

                target_res = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images , target_res)

                losses = sum(loss for loss in loss_dict.values())

                loss_value = losses.item()

                calc_loss.update(loss_value, batch_size)

        return calc_loss





    def save_model(self, path):

        self.model.eval()

        torch.save({

            'model_state_dict': self.model.state_dict(),

            'optimizer_state_dict': self.optimizer.state_dict(),

            'scheduler_state_dict': self.scheduler.state_dict(),

            'best_calc_loss': self.best_calc_loss,

            'epoch': self.epoch,

        }, path)





    def load_model(self, path):

        model = torch.load(path)

        self.model.load_state_dict(model['model_state_dict'])

        self.optimizer.load_state_dict(model['optimizer_state_dict'])

        self.scheduler.load_state_dict(model['scheduler_state_dict'])

        self.best_calc_loss = model['best_calc_loss']

        self.epoch = model['epoch'] + 1
def get_model(num_classes = 2):

  fpn_resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False , pretrained_backbone = False)

  in_features = fpn_resnet.roi_heads.box_predictor.cls_score.in_features

  fpn_resnet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  fpn_resnet._name_ = "fpn_resnet"

  return fpn_resnet
class GlobalParametersTrain:

    lr = 0.0003

    n_epochs = 40 



    folder = '/kaggle/input/modelfasterrcnn'



    verbose = True

    verbose_step = 10



    step_scheduler = False 

    validation_scheduler = True 

    

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau

    scheduler_params = dict(mode='min',factor=0.5,patience=1,verbose=False, threshold=0.0001,threshold_mode='abs',cooldown=0, min_lr=1e-8,eps=1e-08)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training = Training(model=get_model().to(device), device=device, config=GlobalParametersTrain)



images, targets, image_ids = next(iter(valid_loader))

images = list(img.to(device) for img in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)

sample = images[0].permute(1,2,0).cpu().numpy()



training.load_model('../input/fasterrcnnmodel049640/best-model-36epoch.bin')

model = training.model

model.eval()

cpu_device = torch.device("cpu")



outputs = model(images)

outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]



fig, ax = plt.subplots(1, 1, figsize=(16, 8))



threshold = 0.95

for i , score in enumerate(outputs[0]['scores']):

  if score.item() > threshold:

    for box in outputs[0]['boxes']:

        cv2.rectangle(sample,(box[0], box[1]),(box[2] , box[3]),(255, 255, 0), 3)

    

ax.set_axis_off()

ax.imshow(sample)
def run_wbf(predictions,weights=None,image_size=1024,iou_thr=0.5,skip_box_thr=0.43):

    boxes_list = [(pred["boxes"] / (image_size-1)).tolist() for pred in predictions]

    scores_list = [pred["scores"].tolist() for pred in predictions]

    labels_list = [np.ones(len(score)).astype(int).tolist() for score in scores_list]

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)



    return boxes, scores, labels
training = Training(model=get_model(2).to(device), device=device, config=GlobalParametersTrain)

training.load_model('../input/fasterrcnnmodel049640/best-model-36epoch.bin')

model = training.model
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
tta_transforms = []

for tta_combination in product([TTAHorizontalFlip(), None], [TTAVerticalFlip(), None],[TTARotate90(), None]):

    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
def display_tta(path):

    iou_thr = 0.4

    detection_threshold = 0.5

    model.cuda()

    model.eval()



    img = plt.imread(path).astype(np.float32)

    img /= 255.0

    

    t = Compose([Resize(width=512,height=512),ToTensorV2()])

    data = { "image": img}

    data = t(**data)

    

    img_tensor = data["image"]

    img_tensor = img_tensor.squeeze(0)

    selected_tta = random.choice(tta_transforms)

    

    tta_image = selected_tta.augment(img_tensor)

    outputs = model(tta_image.unsqueeze(0).cuda())

    boxes = outputs[0]['boxes'].data.detach().cpu().numpy()

    scores = outputs[0]['scores'].data.detach().cpu().numpy() 

    

    boxes = boxes[scores >= detection_threshold]

    scores = scores[scores >= detection_threshold]

    original_boxes  = selected_tta.deaugment_boxes(boxes.copy())

    

    img_tta = tta_image.permute(1,2,0).detach().cpu().numpy()

    img = img_tensor.permute(1,2,0).detach().cpu().numpy()



    return img , img_tta , boxes , original_boxes
for image_path in test_paths:

    img , tta_img , boxes , original_boxes = display_tta(image_path)

    original_with_boxes = visualizeTarget(img,{"boxes": original_boxes},visualize_data_loader = False)

    tta_with_boxes = visualizeTarget(tta_img,{"boxes": boxes},visualize_data_loader = False)

    visualize(input_image = img, image_with_tta_bounding_boxes = tta_with_boxes , original_with_bounding_boxes = original_with_boxes)
def format_prediction_string(boxes, scores):

    pred_strings = []

    for row in zip(scores, boxes):

        score , x_min , y_min , x_max,y_max =row[0], row[1][0], row[1][1], row[1][2], row[1][3]

        x = round(x_min)

        y = round(y_min)

        h = round(x_max-x_min)

        w = round(y_max-y_min)

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(score, x , y , h , w))

    return " ".join(pred_strings)
model.cuda()

model.eval()



iou_thr = 0.4

detection_threshold = 0.5

n = 20



submission = []



for image_path in tqdm(test_paths):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) 

    image /= 255.0

    t = Compose([Resize(width=512,height=512),ToTensorV2()])

    data = { "image": image}

    data = t(**data)

    image = data["image"]

    image = image.squeeze(0)

   



    predictions = []

    

    for i in range(n):

        selected_tta = random.choice(tta_transforms)

        tta_image = selected_tta.augment(image) ## need to be random

        outputs = model(tta_image.unsqueeze(0).cuda())

        boxes = outputs[0]['boxes'].data.detach().cpu().numpy()

        scores = outputs[0]['scores'].data.detach().cpu().numpy()

        boxes = boxes[scores >= detection_threshold]

        scores = scores[scores >= detection_threshold]

        original_boxes  = selected_tta.deaugment_boxes(boxes)

        predictions.append({"boxes"  : original_boxes,'scores': scores})

    

    boxes, scores, labels = run_wbf(predictions,iou_thr=iou_thr,image_size=512)

    boxes = boxes * 1024.0

        

    image = image.permute(1,2,0).detach().cpu().numpy()

    image = cv2.resize(image,(1024,1024))

    original_with_boxes = visualizeTarget(image,{"boxes": boxes} , visualize_data_loader = False)

    

    visualize(image_test=original_with_boxes)



    prediction_string = format_prediction_string(boxes, scores)

    

    image_name = image_path.split("/")[-1].split(".")[0]

    submission.append([image_name, prediction_string])
SUBMISSION_PATH = '/kaggle/working'

submission_id = 'submission'

submission_path = os.path.join(SUBMISSION_PATH, '{}.csv'.format(submission_id))

sample_submission = pd.DataFrame(submission, columns=["image_id","PredictionString"])

sample_submission.to_csv(submission_path, index=False)

submission_df = pd.read_csv(submission_path)

submission_df