import torch
print(torch.__version__)
# !pip3 install torch==1.6.0
# !pip3 install torchvision==0.7.0
# !pip3 install albumentations==0.4.5
import pandas as pd
import numpy as np
import cv2
import os
import re
import random

from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize, ShiftScaleRotate

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from ast import literal_eval
from torch.cuda.amp import GradScaler, autocast


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

#input_dir = 
DIR_train = '/kaggle/input/global-wheat-detection/train/'
DIR_test = '/kaggle/input/global-wheat-detection/test/'
train = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")
print(train.shape)
train.head()
def converter(x):
    return literal_eval(x)
train["bbox"] = train["bbox"].apply(converter)
train["x"] = -1
train["y"] = -1
train["w"] = -1
train["h"] = -1
train[['x', 'y', 'w', 'h']] = np.stack(train["bbox"])
train['x'] = train['x'].astype(np.float)
train['y'] = train['y'].astype(np.float)
train['w'] = train['w'].astype(np.float)
train['h'] = train['h'].astype(np.float)
type(train["x"][0])
train.drop("bbox",inplace=True,axis=1)
image_ids = train['image_id'].unique()
print(f'Total Number of Images: {len(image_ids)}')
valid_ids = image_ids[-665:]
train_ids = image_ids[:-665]

print(f'Number of Train Images: {len(train_ids)}')
print(f'Number of Validation Images: {len(valid_ids)}')
valid = train[train['image_id'].isin(valid_ids)]
train = train[train['image_id'].isin(train_ids)]
valid.head()
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

def get_all_bboxes(df, image_id):
    image_bboxes = df[df.image_id == image_id]
    
    bboxes = []
    for _,row in image_bboxes.iterrows():
        bboxes.append((row.x, row.y, row.w, row.h))
        
    return bboxes

def plot_image_examples(df, rows=3, cols=3, title='Image examples'):
    fig, axs = plt.subplots(rows, cols, figsize=(10,10))
    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(len(df), size=1)[0]
            img_id = df.iloc[idx].image_id
            
            img = cv2.imread(DIR_train + img_id + '.jpg',)
            img = cv2.cvtColor(img, cv2.INTER_CUBIC)
            axs[row, col].imshow(img)
            
            bboxes = get_all_bboxes(df, img_id)
            
            for bbox in bboxes:
                rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
                axs[row, col].add_patch(rect)
            
            axs[row, col].axis('off')
            
    plt.suptitle(title)
plot_image_examples(train)
class WheatDataset(Dataset):
    
    def __init__(self, df, IMG_DIR, transforms=None):
        super().__init__()
        self.image_ids = df["image_id"].unique()
        self.df = df
        self.IMG_DIR = IMG_DIR
        self.transforms = transforms
        
    def __getitem__(self, index:int):
        image_id = self.image_ids[index]
        row = self.df[self.df["image_id"]==image_id]
        
        """
        Reading and processing the image using CV2
        cv2.IMREAD_COLOR: It specifies to load a color image. 
        Any transparency of image will be neglected. It is the default flag. Alternatively, we can pass integer value 1 for this flag.
        """
        image = cv2.imread(f'{self.IMG_DIR}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        #image/=255.0
        boxes = row[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((row.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((row.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.tensor(sample['bboxes']).float()
            
        return image, target, image_id
    
    
    def __len__(self) -> int:
        return self.image_ids.shape[0]
# Albumentations
def get_train_transform():
    return A.Compose([A.Flip(0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ShiftScaleRotate(),
        ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = WheatDataset(train, DIR_train, get_train_transform())
valid_dataset = WheatDataset(valid, DIR_train, get_valid_transform())


# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
images, targets, image_ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#targets[0]['boxes'][0][0]
# boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)
# sample = images[2].permute(1,2,0).cpu().numpy()
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

num_epochs = 1
%%time
loss_hist = Averager()
itr = 1
scaler = GradScaler()
for epoch in range(num_epochs):
    loss_hist.reset()
    with autocast():
        for images, targets, image_ids in train_data_loader:
            
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1
    
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")
%%time
loss_hist = Averager()
itr = 1

for epoch in range(num_epochs):
    loss_hist.reset()

    for images, targets, image_ids in train_data_loader:


        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1
    
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")
