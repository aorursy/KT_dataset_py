import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import shutil
import torch.nn as nn
from skimage import io
import torchvision
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from albumentations.pytorch import ToTensor
from torchvision import utils
from albumentations import (HorizontalFlip, ShiftScaleRotate, VerticalFlip, Normalize,Flip,Compose, GaussNoise)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

csv_path = '../input/apples/Apple_data/train.csv'
train_dir = '../input/apples/Apple_data/Apple_images'
df = pd.read_csv(csv_path)
df.head
print(f'Total number of train images is {len(os.listdir(train_dir))}')
print(f'Shape of Data Frame is {df.shape}')
print(f'Number of images in Data Frame is {len(np.unique(df["image_name"]))}')
print(f'Number of images with no bounding boxes is {len(os.listdir(train_dir)) - len(np.unique(df["image_name"]))}')
# this function will take the dataframe and vertically stack the image ids 
# with no bounding boxes
def process_bbox(df):
    
    df['x'] = df['x_min'].astype(np.int)
    df['y'] = df['y_min'].astype(np.int)
    df['w'] = df['x_max'].astype(np.int)
    df['h'] = df['y_max'].astype(np.int)

    df.drop(columns = ['x_min', 'y_min', 'x_max', 'y_max','image_type'], inplace = True)
    return df
df_new = process_bbox(df)
print(f'New shape of dataframe {df_new.shape}')
df_new.tail
#Split Train and Validation Data
image_ids = df_new['image_name'].unique()
train_ids = image_ids[0:int(0.8*len(image_ids))]
val_ids = image_ids[int(0.8*len(image_ids)) :]
train_ids.shape, val_ids.shape
print(f'Total images {len(image_ids)}')
print(f'Number of train images {len(train_ids)}')
print(f'Number of validation images {len(val_ids)}')
train_df = df_new[df_new['image_name'].isin(train_ids)]
val_df = df_new[df_new['image_name'].isin(val_ids)]
train_df.shape, val_df.shape, train_df
def get_transforms(phase):
            list_transforms = []
            if phase == 'train':
                list_transforms.extend([
                       Flip(p=0.5)
                         ])
            list_transforms.extend(
                    [
            ToTensor(),
                    ])
            list_trfms = Compose(list_transforms,
                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
            return list_trfms
class Wheatset(Dataset) :
    
    def __init__(self, data_frame, image_dir, phase = 'train') :
        super().__init__()
        self.df = data_frame
        self.image_dir = image_dir
        self.images = data_frame['image_name'].unique()
        self.transforms = get_transforms(phase)
        
    def __len__(self) :
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx] + '.jpg'
        
        image_arr = cv2.imread(os.path.join(self.image_dir, image), cv2.IMREAD_COLOR)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_arr /= 255.0
        image_id = str(image.split('.')[0])
        point = self.df[self.df['image_name'] == image_id]
        boxes = point[['x', 'y', 'w', 'h']].values
        #boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        #boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:, 2] - boxes[:,0])
        area = torch.as_tensor(area, dtype = torch.float32)
        
        #there is only one class
        labels = torch.ones((point.shape[0],), dtype = torch.int64)
        
        #suppose all instances are not crowd
        iscrowd = torch.zeros((point.shape[0],), dtype = torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor(idx)
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        if self.transforms:
            sample = {
                'image': image_arr,
                'bboxes': target['boxes'],
                'labels': target['labels']
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
        target['boxes'] = torch.stack(tuple(map(torch.tensor, 
                                                zip(*sample['bboxes'])))).permute(1, 0).float()


        
        return image, target, image_id     
#Loading training and validation data through the Wheat class
train_data = Wheatset(train_df, train_dir, phase = 'train')
val_data = Wheatset(val_df, train_dir, phase = 'validation')

print(f'Length of train data {len(train_data)}')
print(f'Length of validation data {len(val_data)}')
# batching
def collate_fn(batch):
    return tuple(zip(*batch))

train_data_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    val_data,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)
def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    image = (image * 255).astype(np.uint8)
    return image

def plot_img(data,idx):
    out = data.__getitem__(idx)
    image = image_convert(out[0])
    image = np.ascontiguousarray(image)
    bb = out[1]['boxes'].numpy()
    for i in bb:
        cv2.rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,255,0), thickness=2)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
plot_img(train_data, 100)
plot_img(train_data, 22)
plot_img(val_data,7)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
images, targets, ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# optimizer = torch.optim.Adam(params, lr=0.001)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 12
train_loss_min = 0.9
total_train_loss = []




for epoch in range(num_epochs):
    print(f'Epoch :{epoch + 1}')
    start_time = time.time()
    train_loss = []
    model.train()
    itr = 1
    for images, targets, image_ids in train_data_loader:
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        train_loss.append(losses.item())        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(f'Iteration : {itr}')
        itr = itr + 1
    #train_loss/len(train_data_loader.dataset)
    epoch_train_loss = np.mean(train_loss)
    total_train_loss.append(epoch_train_loss)
    print(f'Epoch train loss is {epoch_train_loss}')
    
#     if lr_scheduler is not None:
#         lr_scheduler.step()
    
    # create checkpoint variable and add important data
  
    
   
    
    time_elapsed = time.time() - start_time
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# PLOT ACCURACIES
plt.figure(figsize=(15,5))
plt.plot(total_train_loss)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
axes = plt.gca()
axes.set_ylim([0.1,0.3])
plt.show()
images, targets, image_ids = next(iter(valid_data_loader))

images = list(img.to(device) for img in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
images
i = 7
boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
sample = images[i].permute(1,2,0).cpu().numpy()
model.eval()
cpu_device = torch.device("cpu")

outputs = model(images)
outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 3)
    
ax.set_axis_off()
ax.imshow(sample)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
boxes = outputs[i]['boxes']
sample = images[i].permute(1,2,0).cpu().numpy()

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 3)
    
ax.set_axis_off()
ax.imshow(sample)


torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
