batch_size = 128
no_epochs = 4
!pip install -U git+git://github.com/lilohuang/PyTurboJPEG.git
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY

!conda install -c conda-forge gdcm -y
!pip install torch_optimizer

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import cv2
import os
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader,Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.color import gray2rgb
import functools
import torch
from torch import Tensor
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
import torch_optimizer as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from tqdm.auto import tqdm
from matplotlib import animation, rc
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pydicom as dcm
import gdcm
import time
torch.backends.cudnn.benchmark = True
train_path = '../input/rsna-str-pulmonary-embolism-detection/train.csv'
test_path = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")
jpeg_path = '../input/rsna-str-pe-detection-jpeg-256/train-jpegs'
files = glob.glob('../input/rsna-str-pulmonary-embolism-detection/train/*/*/*.dcm')
train_df=pd.read_csv(train_path)
df_main=train_df.copy()
df_main.head()
df_main.SOPInstanceUID.nunique()
df_main.shape
list_of_jpgs= [i for i in range(1,10)]
columns_only_for_info = ["qa_motion","qa_contrast","flow_artifact","true_filling_defect_not_pe"]
train_df.iloc[list_of_jpgs,:]
target_columns = ['StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID','pe_present_on_image', 'negative_exam_for_pe', 'rv_lv_ratio_gte_1', 
                  'rv_lv_ratio_lt_1','leftsided_pe', 'chronic_pe','rightsided_pe', 
                  'acute_and_chronic_pe', 'central_pe', 'indeterminate']
train_df[target_columns]
class jpg_dataset(Dataset):
    def __init__(self, root_dir, df,  transforms = None):
        
        super().__init__()
        self.df = df[target_columns]
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ndx):
        row = self.df.iloc[ndx,:]
        in_file = open(glob.glob(f"{root_dir}/{row[0]}/{row[1]}/*{row[2]}.jpg")[0], 'rb')
        img = jpeg.decode(in_file.read())
        in_file.close()
        label = row[3:].astype(int)
        label[2:] = label[2:] if label[0]==1 else 0
        #img /= 255.0
        
        #Retriving class label

        
        #Applying transforms on image
        if self.transforms:
            img = self.transforms(image=img)['image']
        

        return (img,label.values)
root_dir=jpeg_path
jpeg = TurboJPEG()
row = train_df.iloc[0,:]
row

jpg_data = jpg_dataset(root_dir=jpeg_path,df=train_df)
dataloader = DataLoader(jpg_data, batch_size=4,
                        shuffle=True, num_workers=0)
dataloader
for i_batch, sample_batched in enumerate(dataloader):
    image=sample_batched[0]
    print(i_batch, image.size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        plt.imshow(image[0])
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
StudyInstanceUID = list(set(train_df['StudyInstanceUID']))
print(len(StudyInstanceUID))
t_df = train_df[train_df['StudyInstanceUID'].isin(StudyInstanceUID[0:6500])]
v_df = train_df[train_df['StudyInstanceUID'].isin(StudyInstanceUID[6500:])]
t_df.head()
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': A.Compose(
    [
        #A.SmallestMaxSize(max_size=160),
        #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        #A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
),
    'val': A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        A.CenterCrop(height=128, width=128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {}
image_datasets['train'] = jpg_dataset(root_dir=jpeg_path,df=t_df, transforms = data_transforms['train'])
image_datasets['val'] = jpg_dataset(root_dir=jpeg_path,df=v_df, transforms = data_transforms['val'])

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True) for x in ['train', 'val']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
dataset_sizes['train']
image_datasets['train'][0]
!pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet

class EfficientNetEncoderHead(nn.Module):
    def __init__(self, depth, num_classes):
        super(EfficientNetEncoderHead, self).__init__()
        self.depth = depth
        self.base = EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.output_filter = self.base._fc.in_features
        self.classifier = nn.Linear(self.output_filter, num_classes)
    def forward(self, x):
        x = self.base.extract_features(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x
model = EfficientNetEncoderHead(depth=0, num_classes=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
def train_model(model, criterion, optimizer, scheduler, no_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(no_epochs):
        print('Epoch {}/{}'.format(epoch, no_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            current_loss_mean = 0.0
            running_corrects = 0

            # Iterate over data.
            tqdm_loader = tqdm(dataloaders_dict[phase])

            for batch_idx, (inputs,labels) in enumerate(tqdm_loader):
                #inputs = inputs.to(device)
                #labels = labels.to(device)
                
                inputs, labels = inputs.cuda().float(), labels.cuda().float() 

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history only if in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)
                tqdm_loader.set_description('loss: {:.4}'.format(
                    current_loss_mean))
                running_corrects += torch.sum(outputs == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            score = 1-current_loss_mean
            print('metric {}'.format(score))
            
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),f'model{k}.bin')
    return model
def radam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    if isinstance(betas, str):
        betas = eval(betas)
    return optim.RAdam(parameters,
                      lr=lr,
                      betas=betas,
                      eps=eps,
                      weight_decay=weight_decay)


criterion = torch.nn.BCEWithLogitsLoss()

# Observe that all parameters are being optimized

optimizer = radam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-3, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloaders_dict['train'])*no_epochs, eta_min=1e-6) # dataloader should be train one - change later

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
import copy
train_model(model, criterion, optimizer, scheduler, no_epochs=no_epochs)
