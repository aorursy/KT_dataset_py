# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

DIR_INPUT = '/kaggle/input/global-wheat-detection'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
train_df.shape
train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float32)
train_df['y'] = train_df['y'].astype(np.float32)
train_df['w'] = train_df['w'].astype(np.float32)
train_df['h'] = train_df['h'].astype(np.float32)
image_ids = train_df['image_id'].unique()
# valid_ids = image_ids[-665:]
train_ids = image_ids
# valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]
# valid_df.shape, train_df.shape
train_df.shape
class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
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
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# def get_valid_transform():
#     return A.Compose([
#         ToTensorV2(p=1.0)
#     ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
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
def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform())
# valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())


# split the dataset in train and test set
# indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# valid_data_loader = DataLoader(
#     valid_dataset,
#     batch_size=8,
#     shuffle=False,
#     num_workers=4,
#     collate_fn=collate_fn
# )
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
images, targets, image_ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

targets
# boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)
# sample = images[2].permute(1,2,0).cpu().numpy()
# fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# for box in boxes:
#     cv2.rectangle(sample,
#                   (box[0], box[1]),
#                   (box[2], box[3]),
#                   (220, 0, 0), 3)
    
# ax.set_axis_off()
# ax.imshow(sample)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

num_epochs = 3
loss_hist = Averager()
itr = 1

for epoch in range(num_epochs):
    loss_hist.reset()
    
    for images, targets, image_ids in train_data_loader:
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.long().to(device) for k, v in t.items()} for t in targets]
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
# images, targets, image_ids = next(iter(valid_data_loader))
# images = list(img.to(device) for img in images)
# targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# sample = images[0].permute(1,2,0).cpu().numpy()

# model.eval()
# cpu_device = torch.device("cpu")

# outputs = model(images)
# outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
# boxes = outputs[0]['boxes'].cpu().detach().numpy().astype(np.int32)
# fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# for box in boxes:
#     cv2.rectangle(sample,
#                   (box[0], box[1]),
#                   (box[2], box[3]),
#                   (220, 0, 0), 3)
    
# ax.set_axis_off()
# ax.imshow(sample)
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
# outputs


cpu_device = torch.device("cpu")

def get_test_img():
    test_img = []
    for image_id in os.listdir(DIR_TEST):
        image = cv2.imread(f'{DIR_TEST}/{image_id}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
            
        test_img.append(image)
    
    test_img = list(torch.from_numpy(img.transpose(2, 0, 1)).to(device) for img in test_img)
    return test_img
test_img=get_test_img()
test_img
model.eval()

preds = outputs = model(test_img)
preds
sample = test_img[0].permute(1,2,0).cpu().numpy()

preds = [{k: v.to(cpu_device) for k, v in t.items()} for t in preds]
boxes = preds[0]['boxes'].cpu().detach().numpy().astype(np.int32)
score = preds[0]['scores'].cpu().detach().numpy().astype(np.float32)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for index,box in enumerate(boxes):
    if score[index]> 0.05:
        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (220, 0, 0), 3)
    
ax.set_axis_off()
ax.imshow(sample)
from collections import defaultdict
sub = pd.read_csv('/kaggle/input/global-wheat-detection/sample_submission.csv')

for i in range(10):
    current=preds[i]
    current['boxes'] = current['boxes'].cpu().detach().numpy()
    current['scores'] = current['scores'].cpu().detach().numpy()
    
    im_id = os.listdir(DIR_TEST)
    im_id = im_id[i][:-4]
    
    s=""
    for i in range(len(current['boxes'])):
        s+=" "
        s+=str(current['scores'][i])
        s+=" "
        s+=str(int(current['boxes'][i][0]))
        s+=" "
        s+=str(int(current['boxes'][i][1]))
        s+=" "
        s+=str(int(current['boxes'][i][2] - current['boxes'][i][0]))
        s+=" "
        s+=str(int(current['boxes'][i][3]-current['boxes'][i][1]))
    s = s[1:]
    sub.loc[sub['image_id']==im_id,'PredictionString'] = s
        
# sub.loc[sub['image_id']=='aac893a91','PredictionString'] = "0 0 0 0 "
sub.to_csv('submission.csv',index=False)
