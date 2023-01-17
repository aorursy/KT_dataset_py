# Import required packages
import numpy as np
import os
import pandas as pd
import re
from PIL import Image
from PIL import ImageDraw
import gc
import warnings
import time
warnings.filterwarnings('ignore')

# PyTorch libraries
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

from torch.utils.data import DataLoader, Dataset

# OpenCV
import cv2

# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
sns.set_style('whitegrid')
sns.set_palette('muted')
pd.set_option('display.max_columns', 500)
# Define the root directory
root_dir = '../input/global-wheat-detection'
# Load training data csv file
train_data = pd.read_csv(f'{root_dir}/train.csv')

print(train_data.shape)
train_data.head()
# Get bounding box coordinates and make new columns in dataframe
def bbox_coordinates(x):
    coords = np.array(re.findall("([0-9]+\.?[0-9]*)", x))
    coords = list(map(float, coords))
    if len(x) == 0:
        coords = [-1, -1, -1, -1]
    return coords

# Sanity check
x = '[834.0, 222.0, 56.0, 36.0]'
bbox_coordinates(x)
# Add columns for bbox coordinates
train_data['xmin'] = -1
train_data['ymin'] = -1
train_data['width'] = -1
train_data['height'] = -1

# join lists along `axis=0` 
print(train_data['bbox'].apply(lambda x: bbox_coordinates(x)).shape)
coordinates = np.stack(train_data['bbox'].apply(lambda x: bbox_coordinates(x)))
print(coordinates.shape)

# Updating the newly created columns
train_data[['xmin', 'ymin', 'width', 'height']] = coordinates

# dropping the bbox column
train_data.drop(columns='bbox', inplace=True)
train_data.head()
# Checking the data types of dataframe for categorical variables
train_data.dtypes
# Checking the image distribution based on `source`
plt.figure(figsize=(8,6))
sns.countplot(x='source', data=train_data, alpha=0.6)
plt.xlabel("Image source", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.title("Distribution of images \nbased on the source", fontsize=18);
# Number of bounding boxes per image
train_data.groupby('image_id').image_id.count()
# Put bounding boxes on images
def images_with_bbox(df, image_id):
    boxes = df[df['image_id'] == image_id].loc[:, ['xmin', 'ymin', 'width', 'height']].values
    image = os.path.join(root_dir, 'train', image_id) + '.jpg'
    ### OpenCV giving TypeError: an integer is required (got type tuple)
    ### Switching to PIL
    # color = (255, 0, 0)
    # frame = cv2.imread(image)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
    # image = Image.fromarray(img.mul(255).permute(1,2,0).byte().numpy())
    frame = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(frame)
    boxes = boxes.astype(int)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    for box in boxes:
        coord1 = (box[0], box[1])
        coord2 = (box[2], box[3])
        draw.rectangle([coord1, coord2], outline=(220,20,60), width=10)
    return frame
# Checking out a few images
def view_images(root_dir, image_id, df=None, show_bbox=False):
    ncols= 4
    nrows = min(len(image_id)//ncols, 4)
    
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 14))
    
    ax = ax.flatten()
    
    for i, img_id in enumerate(image_id):
        img_name = img_id + '.jpg'
        path = os.path.join(root_dir, 'train', img_name)
        if show_bbox and (df is not None):
            image = images_with_bbox(df, img_id)
            image = np.array(image)
        else:
            image = Image.open(path).convert("RGB")
        ax[i].set_axis_off()
        ax[i].imshow(image)
# Showing random images from the data
images = train_data.sample(n=8)['image_id'].values
view_images(root_dir, images)
# Showing the images with bounding boxes
view_images(root_dir, images, df=train_data, show_bbox=True)
# Creating the dataset
class WheatDataset(Dataset):
    def __init__(self, data_dir, dataframe, transforms=None):
        super().__init__()
        self.data = data_dir
        self.df = dataframe
        self.transforms = transforms
        # load all the images and sort them
        self.images = sorted(os.listdir(os.path.join(data_dir)))
        
    def __getitem__(self, idx:int):
        # getting the image id for the given image
        image_id = self.images[idx].split('.')[0]
        
        # get the bounding box info for the given image
        boxes_info = (self.df[self.df['image_id'] == image_id])
        boxes = boxes_info[['xmin', 'ymin', 'width', 'height']].values
        
        # get the xmax and ymax from width and height
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]    # xmax
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]    # ymax
        
        # load the image
        path = os.path.join(self.data, image_id) + '.jpg'
        image = Image.open(path).convert("RGB")
        
        # convert boxes to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        # create labels; there's only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        
        # iscrowd; for background
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # area of bounding boxes
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # image label
        image_label = torch.tensor([idx])
        
         # storing all the attributes in a dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_label
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target, image_id
    
    def __len__(self):
        return len(self.images)
# Sanity check to see if dataset class is working
data_dir = os.path.join(root_dir, 'train')
dataset = WheatDataset(data_dir=data_dir, dataframe=train_data)
img, target, img_id = dataset.__getitem__(3)
target
def get_model(num_classes):
    # load the object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get the input features in the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the input features of pretrained head with the num_classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
# helper class to keep track of loss and loss per iteration
# source: https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train
class Averager:
    def __init__(self):
        self.current_total = 0
        self.iterations = 0
        
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
        self.current_total = 0
        self.iterations = 0
# creating some transforms
# source: https://github.com/pytorch/vision/blob/master/references/detection/transforms.py
# source: https://github.com/microsoft/computervision-recipes/blob/master/utils_cv/detection/dataset.py
import random
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    """
    Wrapper for torchvision's HorizontalFlip
    """
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]  # image must be a ndarray or torch tensor (PIL.Image has no attribute `.shape`)
            image = image.flip(-1)
            bbox = target['boxes']    # bbox coordinates MUST be of form [xmin, ymin, xmax, ymax]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target['boxes'] = bbox
        return image, target

class ColorJitterTransform(object):
    """
    Wrapper for torchvision's ColorJitter
    """
    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, image, target):
        image = ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue
        )(image)
        return image, target
        
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)   # normalizes the image and converts PIL image to torch.tensor
        return image, target
# merge all the images in a batch
def collate_fn(batch):
    return tuple(zip(*batch))

# add some image augmentation operations
def get_transform(train):
    transforms = []
    
    if train:
        # needs the image to be a PIL image
        transforms.append(ColorJitterTransform(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.05))
        
    # converts a PIL image to pytorch tensor
    transforms.append(ToTensor())
    
    if train:
        # randomly flip the images, bboxes and ground truth (only during training)
        transforms.append(RandomHorizontalFlip(0.5))  # this operation needs image to be a torch tensor
    
    return Compose(transforms)
# Use the dataset class and create train and test dataloaders
dataset_train = WheatDataset(data_dir=data_dir, dataframe=train_data, 
                             transforms=get_transform(train=True))
dataset_valid = WheatDataset(data_dir=data_dir, dataframe=train_data, 
                             transforms=get_transform(train=False))

# split the dataset in train and valid
indices = torch.randperm(len(dataset_train)).tolist()
dataset_train = torch.utils.data.Subset(dataset_train, indices[:-700])
dataset_valid = torch.utils.data.Subset(dataset_valid, indices[-700:])

# creating dataloaders
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, 
                                   num_workers=4, collate_fn=collate_fn)

dataloader_valid = DataLoader(dataset_valid, batch_size=4, shuffle=False, 
                                  num_workers=4, collate_fn=collate_fn)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def show_bbox_single_image(images, targets):
    image = Image.fromarray(images[0].mul(255).permute(1,2,0).cpu().byte().numpy())
    boxes = targets[0]['boxes'].cpu().numpy().astype(np.int64)

    draw = ImageDraw.Draw(image)

    for box in boxes:
        coord1 = (box[0], box[1])
        coord2 = (box[2], box[3])
        draw.rectangle([coord1, coord2], outline=(220,20,60), width=3)
    return image
# sample in training set 
images, targets, image_ids = next(iter(dataloader_train))
image = show_bbox_single_image(images, targets)
image
# sample in validation set
images, targets, image_id = next(iter(dataloader_valid))
image = show_bbox_single_image(images, targets)
image
# our dataset has only two classes: wheat heads and background
num_classes = 2

# get the model using helper function
model = get_model(num_classes)

# move the model to device
model.to(device)

# creating the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# learning rate scheduler; reduce lr by 0.5 after every 3 epochs if loss reaches plateau or doesn't decrease
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          patience=3, verbose=True, 
                                                          factor=0.5)
## Source: https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train
loss_hist = Averager()
itr=1
min_loss = np.Inf
num_epochs = 20

for epoch in range(1, num_epochs+1):
    # keep track of time to run each epoch
    epoch_start_time = time.time()
    # reset the loss history (fresh start for each epoch)
    loss_hist.reset()
    # put the model in train mode
    model.train()
    for images, targets, image_ids in dataloader_train:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        loss_hist.send(loss_value)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if itr % 500 == 0:
            print(f"Iteration #{itr},  loss: {loss_value:.6f}")
        
        itr += 1
    # update learning rate
    if lr_scheduler is not None:
        lr_scheduler.step(loss_hist.value)
        
    print(f"Epoch #{epoch},  time taken: {(time.time() - epoch_start_time):.2f} seconds,  loss: {loss_hist.value:.6f}")
    if loss_hist.value <= min_loss:
        print(f"Loss decreased: {min_loss:.6f} --> {loss_hist.value:.6f}. Saving model ...")
        torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
        min_loss = loss_hist.value
# Load the saved model
saved_model_path = 'fasterrcnn_resnet50_fpn.pth'
model.load_state_dict(torch.load(saved_model_path))
# test the model in validation set
images, targets, label_id = next(iter(dataloader_valid))
images = list(img.to(device) for img in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
# setting the model to evaluation mode
model.eval()
cpu_device = torch.device("cpu")
outputs = model(images)
outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
# Showing inference on model output
def show_inferences(images, targets, num_images):
    ncols = 2
    nrows = min(num_images//ncols, 2)
    
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 14))
    
    ax = ax.flatten()  
    for idx in range(num_images):
        image = Image.fromarray(images[idx].mul(255).permute(1,2,0).cpu().detach().byte().numpy())
        boxes = targets[idx]['boxes'].cpu().detach().numpy().astype(np.int64)

        draw = ImageDraw.Draw(image)

        for box in boxes:
            coord1 = (box[0], box[1])
            coord2 = (box[2], box[3])
            draw.rectangle([coord1, coord2], outline=(220,20,60), width=10)
        ax[idx].set_axis_off()
        ax[idx].imshow(image)
# checking the model output
show_inferences(images, outputs, 4)
# saving the model
# torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')