import numpy as np
import pandas as pd
import os
import imageio
from skimage import draw, measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
from torchvision import models
from torchvision.models.detection import FasterRCNN #, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from tqdm import tqdm

%matplotlib inline
plt.rcParams['figure.figsize'] = (15, 10);
data_dir = '../input/global-wheat-detection'
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
# np.where(train_df['height'] != 1024, 1, 0).sum()
train_df
def extract_box(string):
    string = string.replace('[', '').replace(']', '').replace(' ', '')
    box = []
    for n in string.split(','):
        if '.' in n:
            box.append(int(n.split('.')[0]))
        else:
            box.append(int(n))
    return box
def plot_sample(img_path, df):
    img = imageio.imread(img_path)
    boxes = [y['bbox'] for x, y in df.iterrows() 
             if y['image_id'] == img_path.split('/')[-1].replace('.jpg', '')]
    boxes = [extract_box(i) for i in boxes]
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for box in boxes:
        #box = [int(x.strip(' ')) for x in box.strip('\'').strip('[').strip(']').split(',')]
        #print(box)
        rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        #break
    plt.show(ax)
    #return boxes

# plot_sample(data_dir+'/train/010c93b99.jpg', train_df)
plot_sample(data_dir+'/train/0172359d2.jpg', train_df)
def mask_to_boxes(boxes, size=1024):
    all_boxes = []
    for box in boxes:
        new_box = [0, 0, 0, 0]
        new_box[1] = box[1]
        new_box[0] = box[0]
        if box[1]+box[3] < (size-1):
            new_box[3] = box[1]+box[3]
        else:
            new_box[3] = size-1
        if box[0]+box[2] < (size-1):
            new_box[2] = box[0]+box[2]
        else:
            new_box[2] = size-1
        all_boxes.append(new_box)
    return all_boxes

def bbox_to_mask(boxes, mask_arr=None, 
                 size=1024, return_boxes=False):
    if mask_arr is None:
        mask_arr = np.zeros((size, size))
    if return_boxes:
        for box in mask_to_boxes(boxes):
            mask_arr[box[1], box[0]:box[2]] = 1
            mask_arr[box[3], box[0]:box[2]] = 1
            mask_arr[box[1]:box[3], box[0]] = 1
            mask_arr[box[1]:box[3], box[2]] = 1
    else:
        for box in boxes:
            mask_arr[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = 1
    return mask_arr

def plot_mask(img_path, df, size=1024):
    img = imageio.imread(img_path)
    boxes = [y['bbox'] for x, y in df.iterrows() 
             if y['image_id'] == img_path.split('/')[-1].replace('.jpg', '')]
    boxes = [extract_box(i) for i in boxes]
    fig, ax = plt.subplots(1)
    img = bbox_to_mask(boxes)
    ax.imshow(img, cmap='gray')
    
# plot_mask(data_dir+'/train/010c93b99.jpg', train_df)
plot_mask(data_dir+'/train/0172359d2.jpg', train_df)
def process_frame(frame, test=False):
    frame = frame.groupby('image_id')['bbox'].apply(list)
    #index = range(0, len(frame))
    df = pd.DataFrame()
    df['image_id'] = frame.index
    df['index'] = range(0, len(frame))
    if test:
        return df.set_index('index')
    df['bbox'] = frame.tolist()
    return df.set_index('index')

train_df = process_frame(train_df)
#test_df = process_frame(test_df, test=True)
# !wget https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth
class WheatDataset(Dataset):
    
    def __init__(self, frame, device='cpu', img_format='.jpg', size=1024, test=False):
        super(WheatDataset, self).__init__()
        self.img_format = img_format
        self.frame = frame
        self.device = device
        self.size = size
        self.test = test
        
    def __len__(self):
        return self.frame.shape[0]
    
    def __getitem__(self, ind):
        if self.test:
            img_path = os.path.join(data_dir, 'test', 
                                    self.frame.loc[ind, 'image_id']+self.img_format)
        else:
            img_path = os.path.join(data_dir, 'train', 
                                    self.frame.loc[ind, 'image_id']+self.img_format)
        self.img = torch.from_numpy(np.array(imageio.imread(img_path))/255)
        if not self.test:
            self.target = {}
            self.target['masks'] = torch.from_numpy(self.get_mask(ind)[-1])
            self.target['labels'] = torch.from_numpy(self.get_mask(ind)[1])
            self.target['boxes'] = torch.from_numpy(self.get_mask(ind)[0])
            return self.img, self.target
        return self.img
        
    def get_mask(self, ind):
        boxes = self.frame.loc[ind, 'bbox']
        boxes = np.array([extract_box(i) for i in boxes])
        mask = bbox_to_mask(boxes, None, self.size, True)
        return boxes, np.array([1]), mask
        
def to_device(array):
    if array is None:
        return None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return torch.tensor(array, device=device)
def dice_loss(input, target):
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
class WheatModel(nn.Module):
    
    def __init__(self, size=1024, 
                 in_channels=3, 
                 out_channels=1,
                 device='cuda'):
        super(WheatModel, self).__init__()
        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
        num_classes = 2
        in_features = self.net.roi_heads.box_predictor.cls_score.in_features
        self.net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        #n_features = self.network.fc.out_features
        #backbone.out_channels = n_features
        
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)
        self.model = FasterRCNN(backbone,
                   num_classes=num_classes)
                   #rpn_anchor_generator=anchor_generator,
                   #box_roi_pool=roi_pooler)
        
        
        
    def forward(self, x, y):
        #print(x.shape)
        #x = x.view(-1, self.in_channels, self.size, self.size).float()
        #x = x.float()
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.pool1(x))
        #x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        #x = F.relu(self.pool2(x))
        #print(x.shape)
        
        return self.model(x, y) #, masks=masks) #x.view(-1, self.out_channels, self.out_channels)
    
    def training_step(self, batch):
        #print(batch)
        #images, masks = batch 
        out = self(batch)
        loss = dice_loss(out, masks)
        return loss
    
    def validation_step(self, batch):
        images, masks = batch 
        out = self(batch)
        loss = dice_loss(out, masks)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))
train_set = WheatDataset(train_df)
train_size = int(0.8*len(train_set))
train_set, val_set = random_split(train_set, [train_size, len(train_set) - train_size])
epochs = 3
batch_size = 16
train_loader = DataLoader(train_set, batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_set = WheatDataset(submission, 'cpu', '.jpg', 1024, True)
test_set
def fit():
    model = WheatModel()
    optimizer = Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        total_loss = []
        model.train()
        #for sample in tqdm(train_loader):
        for sample in tqdm(train_set):
            #print(sample[1]['boxes'].shape)
            model.zero_grad()
            out = model(sample[0], sample[1]) #sample[0].view(-1, 3, 1024, 1024), sample[1])
            loss = dice_loss(out, sample[1].view(-1, 1, 1024, 1024))
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            optimizer.zero_grad()
        print(np.array(total_loss).mean())
    torch.save(model.state_dict(), 'model_weights.pth')
    return model
fit()
test_set.frame
@torch.no_grad()
def make_predictions(dataset, weights_path='../working/model_weights.pth', validation=True):
    model = WheatModel()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    total_loss = []
    preds = []
    i=0
    for sample in tqdm(dataset):
        out = model(sample[0].view(-1, 3, 1024, 1024))
        preds.append(out.view(-1, 1024, 1024))
        if validation:
            loss = dice_loss(out, sample[1].view(-1, 1, 1024, 1024))
            total_loss.append(loss.item())
        break
    if validation:
        print(np.array(total_loss).mean())
    return preds
        
preds = make_predictions(val_set) #, weights_path='../input/wheat-model-weights/model_weights.pth')
img = preds[0].view(1, 1024, 1024).detach().numpy().reshape(1024, 1024)
plt.imshow(img, cmap='gray')
plt.show()
