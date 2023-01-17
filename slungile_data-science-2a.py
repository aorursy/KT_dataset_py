# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd 
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import SubsetRandomSampler
import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
import cv2
from PIL import Image
from tqdm.notebook import tqdm
import torch.nn as nn
train_path = '/kaggle/input/global-wheat-detection/train.csv'
train_img = '/kaggle/input/global-wheat-detection/train/'
train_df = pd.read_csv(train_path)
train_df.head()
def unique_col_values(df):
    for column in df:
        print("{} | {} | {}".format(df[column].name, len(df[column].unique()), 
                                    df[column].dtype))
unique_col_values(train_df)
train_df['source'].unique()
train_df['source'].value_counts()
print('Images with width less than 1024: ', train_df[train_df['width'] < 1024])
print('Images with width greater than 1024: ', train_df[train_df['width'] > 1024])
print('Images with height less than 1024: ', train_df[train_df['height'] < 1024])
print('Images with height greater than 1024: ', train_df[train_df['height'] > 1024])
xmin, ymin, width, height = [], [], [], []
bboxes = {}
for i, image_id in enumerate(train_df['image_id']):
    bbox = train_df['bbox'][i][1:-1]
    bbox = bbox.split(',')
    xmin = float(bbox[0])
    ymin = float(bbox[1])
    width = float(bbox[2])
    height = float(bbox[3])
    
    if image_id not in bboxes:
        bboxes[image_id] = []
        bboxes[image_id].append([xmin, ymin, width, height])
    else:
        bboxes[image_id].append([xmin, ymin, width, height])
for i, image_id in enumerate(bboxes):
    image_path = train_img + image_id + '.jpg'
    img = np.array(Image.open(image_path), dtype=np.uint8)
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    
    for xmin, ymin, width, height in bboxes[image_id]:
        rect = patches.Rectangle( (xmin ,ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    if i == 10:
        break
#append .jpg to image ids for easier handling
train_df['image_id'] = train_df['image_id'].apply(lambda x: str(x) + '.jpg')
# Number of unique training images
train_df['image_id'].nunique()
#separating x,y,w,h into separate columns
bboxes = np.stack(train_df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep = ',')))
for i, col in enumerate(['x_min', 'y_min', 'w', 'h']):
    train_df[col] = bboxes[:,i]

train_df.drop(columns = ['bbox'], inplace = True)
train_df.head()
image_ids = train_df['image_id'].unique()
validation_ids = image_ids[-665:]
training_ids = image_ids[:-665]
validation_df = train_df[train_df['image_id'].isin(validation_ids)]
training_df = train_df[train_df['image_id'].isin(training_ids)]
training_df.shape, validation_df.shape
class WheatDataset(Dataset):
    def __init__(self, df, image_dir,transform = None):
        super().__init__()
        self.df = df
        self.image_ids = df['image_id'].unique()
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        
        image = cv2.imread(os.path.join(self.image_dir, image_id), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image/255.0
        
        boxes = records[['x_min', 'y_min', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype = torch.float32)
        
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int32)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor(index)
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': target['labels']
            }
            sample = self.transform(**sample)
            image = sample['image']
            
            if len(sample['bboxes']) > 0:
                target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
            else:
                target['boxes'] = torch.linspace(0,3, steps = 4, dtype = torch.float32)
                target['boxes'] = target['boxes'].reshape(-1,4)
            
        return image, target, image_id
def get_training_transform():
    return alb.Compose([
    alb.VerticalFlip(p = 0.5),
    alb.HorizontalFlip(p = 0.5),
    ToTensorV2(p = 1.0)
], p=1.0, bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_validation_transforms():
    return alb.Compose([ToTensorV2(p = 1.0)], p = 1.0, bbox_params = alb.BboxParams(format='pascal_voc', label_fields=['labels']))
# load a pre-trained model for classification and return
# only the features
dense_net = torchvision.models.densenet169(pretrained=True)
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
modules = list(dense_net.children())[:-1]
backbone = nn.Sequential(*modules)
backbone.out_channels = 1664

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
def collate_fn(batch):
    return tuple(zip(*batch))
training_dataset = WheatDataset(training_df, train_img, get_training_transform())
validation_dataset = WheatDataset(validation_df, train_img, get_validation_transforms())

# split the dataset in train and test set
indices = torch.randperm(len(training_dataset)).tolist()

train_dataloader = DataLoader(
        training_dataset, batch_size=2, shuffle= True, num_workers=4,
        collate_fn= collate_fn)

valid_dataloader = DataLoader(
        validation_dataset, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=collate_fn)
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
device
# move model to the right device
# move model to the right device
model.to(device)
    
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9,dampening=0, weight_decay=0, nesterov=False)
    
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10



total_train_loss = []
total_test_loss = []

for epoch in range(num_epochs):
    model.train()
    
    print('Epoch: ', epoch + 1)
    train_loss = []
    
    for images, targets, image_ids in tqdm(train_dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)  
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss.append(loss_value)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        #if itr%50 == 0:
            #print('Iteration: ' + str(itr) + '\n' + 'Loss: '+ str(loss_value))
            
            
             #itr += 1
        
    epoch_loss = np.mean(train_loss)
    print('Epoch Loss is: ' , epoch_loss)
    total_train_loss.append(epoch_loss)
    
    with torch.no_grad():
        test_losses = []
        for images, targets, image_ids in tqdm(valid_dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            test_loss = losses.item()
            test_losses.append(test_loss)
            
            
    test_losses_epoch = np.mean(test_losses)
    print('Test Loss: ' ,test_losses_epoch)
    total_test_loss.append(test_losses_epoch)
    
    if lr_scheduler is not None:
        lr_scheduler.step(test_losses_epoch)
        
torch.save(model.state_dict(), 'fasterrcnn.pth')
model.eval()
images, targets, image_ids = next(iter(valid_dataloader))
images = list(img.to(device) for img in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
sample = images[1].permute(1,2,0).cpu().numpy()
predictions = model(images)
#predictions
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 3)
    
ax.set_axis_off()
ax.imshow(sample)
test_img='../input/global-wheat-detection/test'
test_df = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')
test_df.shape
test_df['image_id'] = test_df['image_id'].apply(lambda x: str(x) + '.jpg')
test_df.head()
class WheatDataset(Dataset):
    def __init__(self, df, image_dir,transform = None):
        super().__init__()
        
        self.image_ids = df['image_id'].unique()
        self.df=df
        self.image_dir = image_dir
        self.transform = transform
        
    
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        
        image = cv2.imread(self.image_dir, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image/255.0
        

        if self.transform:
            sample = {
                'image': image,
            }
            sample = self.transform(**sample)
            image = sample['image']
            
            
        return image, image_id
        
    def __len__(self) ->int:
        return self.image_ids.shape[0]
def get_training_transform():
    return alb.Compose([
    alb.VerticalFlip(p = 0.5),
    alb.HorizontalFlip(p = 0.5),
    ToTensorV2(p = 1.0)
], p=1.0, bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_validation_transforms():
    return alb.Compose([ToTensorV2(p = 1.0)], p = 1.0, bbox_params = alb.BboxParams(format='pascal_voc', label_fields=['labels']))
    
def collate_fn(batch):
    return tuple(zip(*batch))

test_dataset = test_df['image_id'].unique()
test_dataset = WheatDataset(test_df, test_img, get_training_transform())

test_dataloader = DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=collate_fn)
def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)
detection_threshold = 0.5
results = []

for images, image_ids in test_dataloader:

    images = list(image.to(device) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):

        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        result = {'image_id': image_id,'PredictionString': format_prediction_string(boxes, scores)}

        
        results.append(result)





