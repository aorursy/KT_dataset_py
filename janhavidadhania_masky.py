import pandas as pd
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from albumentations.pytorch.transforms import ToTensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
import torch.nn as nn
df = pd.read_csv('/kaggle/input/face-mask-detection-dataset/train.csv')
df.shape
len(df.classname.value_counts())
df.head()
#columns are ulta-pulta :p
df.rename(columns = {'x2' : 'y1', 'y1' : 'x2'}, inplace = True)
df.head()
# total 20 classes 1 to 20 labels :) ...
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(df['classname'])
df['classname']=(le.transform(df['classname'])+1)
df.head()
df.isnull().sum()
image_ids = df['name'].unique()
image_ids.sort()
valid_ids=image_ids[:0] 
print(valid_ids)
#full train :) ... 
train_ids=image_ids[:]
valid_df = df[df['name'].isin(valid_ids)]
train_df = df[df['name'].isin(train_ids)]
#see what you did - no validation data - all training data
valid_df.shape, train_df.shape
df.classname.unique()
#all those 20 classes 
#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
#refer above link to prepare dataset
class prepare_data(Dataset):

    def __init__(self, dataframe, transforms=None):
        super().__init__()

        self.image_ids = dataframe['name'].unique()
        self.df = dataframe
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['name'] == image_id]

        image = cv2.imread('../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'+f'{image_id}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = torch.as_tensor(records[['x1', 'y1', 'x2', 'y2']].values, dtype=torch.float32)
        # there are 21 classes
        labels = torch.as_tensor(records.classname.values,dtype=torch.int64)
        

        keep = (boxes[:, 3]>boxes[:, 1]) & (boxes[:, 2]>boxes[:, 0]) ## To Handle NAN LOSS Cases 
        boxes = boxes[keep]
        labels = labels[keep]

        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        # target['area'] = area

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
        ToTensor()
    ])
        

def get_valid_transform():
    return A.Compose([
        ToTensor()
    ])
## pytorch Faster-RCNN Resnt50 Pretrained Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 21  # 20 class (masks) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = prepare_data(train_df,get_train_transform())
valid_dataset = prepare_data(valid_df, get_valid_transform()) 

print(type(train_dataset))
print(type(train_df))
train_data_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
#     num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
images, targets, image_ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)
sample = images[2].permute(1,2,0).cpu().numpy()
model.to(device)

#Retriving all trainable parameters from model (for optimizer)
params = [p for p in model.parameters() if p.requires_grad]
print(params)
#Defininig Optimizer
optimizer = torch.optim.Adam(params, lr = 0.0001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.00017, div_factor=2 ,steps_per_epoch=len(train_data_loader), epochs=5)

num_epochs = 5
loss_list = []

for epoch in range(num_epochs):
    
    z=tqdm(train_data_loader)
    print(z)

    for itr,(images, targets, image_ids) in enumerate(z):
        torch.cuda.empty_cache()
        
        images = list(image.to(device).float() for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        ## Returns losses and detections 
        # refer to this link - https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        loss_dict = model(images, targets)
        print(loss_dict, end=" ")
        print(len(loss_dict), end=" ")

        losses = sum(loss for loss in loss_dict.values())
        #losses.item() is basically losses[0]...
        #refer this link "https://github.com/pytorch/tnt/issues/108"
        loss_value = losses.item()

        loss_list.append(loss_value)
        z.set_description(f'Epoch {epoch+1}/{num_epochs}, LR: %6f, Loss: %.6f'%(optimizer.state_dict()['param_groups'][0]['lr'],loss_value))
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step() ## Since We are using 1-Cycle LR Policy, LR update step has to be taken after every batch


    print(f"Epoch #{epoch+1} loss: {loss_hist.value}")
#     torch.save(model.state_dict(), f'/content/drive/My Drive/internshala round 1/model-epoch{epoch+1}.pth') 
    print()
    print('Saving Model.......')
    print()
class MaskTestDataset(Dataset):

    def __init__(self, dataframe, transforms=None):
        super().__init__()

        self.image_ids = dataframe['name'].unique()
        self.df = dataframe
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['name'] == image_id]

        image = cv2.imread('../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'+f'{image_id}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
def get_test_transform():
    return A.Compose([
        ToTensor()
    ])
test_df=pd.read_csv('../input/face-mask-detection-dataset/train.csv')
test_df.head()
def collate_fn(batch):
    return tuple(zip(*batch))

test_dataset = MaskTestDataset(test_df[:40], get_test_transform())

test_data_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    collate_fn=collate_fn
)
torch.cuda.empty_cache()
%%time

detection_threshold = 0.60
results = []
model.eval()
for images, image_ids in test_data_loader:
    torch.cuda.empty_cache()

    images = list(image.to(device) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):

        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        labels = outputs[i]['labels'].data.cpu().numpy()

        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        
        
        result = {
            'image_id': image_id,
            'labels': labels,
            'scores': scores,
            'boxes': boxes
        }

        
        results.append(result)
## Using Dictionary is Fastest Way to Create SUBMISSION DATASET.
new=pd.DataFrame(columns=['image_id', 'boxes', 'label'])
rows=[]
for j in range(len(results)):
    for i in range(len(results[j]['boxes'])):
        dict1 = {}
        dict1={"image_id" : results[j]['image_id'],
                  'x1': results[j]['boxes'][i,0],
                  'x2': results[j]['boxes'][i,2],
                  'y1': results[j]['boxes'][i,1],
                  'y2': results[j]['boxes'][i,3],
                  'classname':results[j]['labels'][i].item()}
        rows.append(dict1)


sub=pd.DataFrame(rows)
sub['classname']=le.inverse_transform(sub.classname.values - 1) ## Converting Back Labels To Original Names 
sub.head()
sample = images[1].permute(1,2,0).cpu().numpy()
boxes = outputs[1]['boxes'].data.cpu().numpy()
scores = outputs[1]['scores'].data.cpu().numpy()
boxes = boxes[scores >= 0.6].astype(np.int32)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 2)
    
ax.set_axis_off()
ax.imshow(sample)