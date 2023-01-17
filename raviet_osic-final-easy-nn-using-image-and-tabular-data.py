!conda install -c conda-forge gdcm -y
from IPython.display import display_html
def restartkernel() :
    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
restartkernel()
import gdcm
import plotly.graph_objs as go
import pydicom as dicom
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pandas as pd
import numpy as np
import gc
import os
import time
import datetime
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
folder_path = '../input/osic-pulmonary-fibrosis-progression'
train_csv = folder_path + '/train.csv'
test_csv = folder_path+ '/test.csv'
sample_csv = folder_path + '/sample_submission.csv'

train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)
sample = pd.read_csv(sample_csv)

print(train_data.shape)
print(test_data.shape)
print(sample.shape)

train_data.head()

test_data.head()
grouped=train_data.groupby('Patient')
pid=[]
for name,group in grouped:
    if np.var(group.FVC)>100000:
        pid.append(name)
        print(name)
        print(np.var(group.FVC),np.std(group.FVC))
        
fig =go.Figure()

for patient in pid:
    df = train_data[train_data["Patient"] == patient]
    fig.add_trace(go.Scatter(x=df.Weeks,y=df.FVC,
                            mode='lines',
                            name=str(patient)))
fig.show()
train=train_data
for patient in pid:
    train=train[train["Patient"] != patient]
grouped=train_data.groupby('Patient')
pid=[]
for name,group in grouped:
    pid.append(name)
        
fig =go.Figure()

for patient in pid:
    df = train_data[train_data["Patient"] == patient]
    fig.add_trace(go.Scatter(x=df.Weeks,y=df.FVC,
                            mode='lines',
                            name=str(patient)))
fig.show()

train.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
chunk=test_data

print("add infos")
sub =sample
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
print(sub.index.size)
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(test_data.drop('Weeks', axis=1), on="Patient")
sub.head()
tr=train
tr['WHERE'] = 'train'
chunk['WHERE'] = 'val'
sub['WHERE'] = 'test'
data = tr.append([chunk, sub])
data['min_week'] = data['Weeks']
data.loc[data.WHERE=='test','min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
data.loc[data.Weeks == data.min_week]
base = data.loc[data.Weeks == data.min_week]
base = base[['Patient','FVC']].copy()
base.columns = ['Patient','min_FVC']
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb==1]
base.drop('nb', axis=1, inplace=True)
base[base.Patient=='ID00419637202311204720264']
data = data.merge(base, on='Patient', how='left')
data['base_week'] = data['Weeks'] - data['min_week']
#del base
data['diff_fvc_prev']=data['FVC'].diff(1)/data['FVC'].shift(1)
data[data.Patient=='ID00007637202177411956430']
COLS = ['Sex','SmokingStatus'] #,'Age'
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)
data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )
data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )
data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )
data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )

FE += ['age','percent','week','BASE']
data[FE]
data=data.fillna(0)

data.head()
tr = data.loc[data.WHERE=='train']
chunk = data.loc[data.WHERE=='val']
sub = data.loc[data.WHERE=='test']
count=0
for dirname, _, filenames in os.walk('../input/osic-pulmonary-fibrosis-progression/test'):
    for filename in filenames:
        count=count+1
print("Count of total files in test data:",count)
patients = train['Patient'].unique()
data_dir = '../input/osic-pulmonary-fibrosis-progression' + '/train/'

for patient in patients[0:5]:
    #patient='ID00026637202179561894768'
    #label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient

    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    slices = [dicom.dcmread(path + '/' + s) for s in os.listdir(path)]
    try:
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    except:
        print(patient,slices.ImagePositionPatient[2])
        break
    print(patient,slices[0].pixel_array.shape, len(slices))
samp=['ID00009637202177434476278','ID00014637202177757139317']
IMG_PX_SIZE = 150

for patient in patients[:1]:
    
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    fig = plt.figure()
    for num,each_slice in enumerate(slices[:12]):
        y = fig.add_subplot(3,4,num+1)
        new_img = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE))
        y.imshow(new_img,cmap='gray')
    plt.show()
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)



def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


PIXEL_MEAN = 0.25
def zero_center(image):
    image = image - PIXEL_MEAN
    return image
from PIL import Image
def process_data(patient,data_dir,img_px_size=64, HM_SLICES=10, visualize=False,):
    def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
   
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def mean(l):
        return sum(l) / len(l)
    #patient='ID00422637202311677017371'
    
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    try:
        print(patient)
        # sorting the ct scan of a patient based on the 3D position of the scan
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    except:
        print(patient,"No Image position patient")
        return []
        
    new_slices = []
    # 1-Converting pixel data to HU
    slices= get_pixels_hu(slices)
    # 2-Chunking the overall images in to 10 images for each patient
    chunk_sizes = round(len(slices) / HM_SLICES)
    if(len(slices)%HM_SLICES!=0):
        a=math.floor((len(slices) / HM_SLICES))
        b=math.ceil(len(slices) / HM_SLICES)
        x=abs((len(slices)-(b*HM_SLICES))/(b-a))
        y=HM_SLICES - x         

        split=int(x*a)

        for slice_chunk in chunks(slices[:split], a):
            slice_chunk = list(map(mean, zip(*slice_chunk)))
            new_slices.append(slice_chunk)
        for slice_chunk in chunks(slices[split:], b):
            slice_chunk = list(map(mean, zip(*slice_chunk)))
            new_slices.append(slice_chunk)
    else:
        chunk_sizes = round(len(slices) / HM_SLICES)
        for slice_chunk in chunks(slices, chunk_sizes):
            slice_chunk = list(map(mean, zip(*slice_chunk)))
            #3-Normalising the values of images(slices)
            b=normalize(np.array(slice_chunk))
            #4-Zero centering the above pixels
            slice_chunk=zero_center(b)
            new_slices.append(slice_chunk)
    real_slices = []
    fig = plt.figure()
    for num,each_slice in enumerate(new_slices):
        #5- Resizing the Images
        each_slice = cv2.resize(np.array(each_slice),(IMG_PX_SIZE,IMG_PX_SIZE))
        real_slices.append(each_slice)
        #print(np.array(each_slice).shape)
        y = fig.add_subplot(4,5,num+1)
        y.imshow(each_slice, cmap='gray')
    plt.show()
    return np.array(real_slices),

IMG_PX_SIZE = 64
HM_SLICES = 64

data_dir = '../input/osic-pulmonary-fibrosis-progression' + '/train/'
much_data = []


    
for num,patient in enumerate(patients[:2]):
    if num % 100 == 0:
        print(num)
    try:
        
        img_data = process_data(patient,data_dir)
        #print(img_data.shape,label)
        much_data.append([img_data])
    except KeyError as e:
        print('This is unlabeled data!')

#np.save('../input/osic-pulmonary-fibrosis-progression/images/muchdata-{}-{}-{}.npy'.format(64,64,64), much_data)

tr.head()
tr.columns
columns=['Weeks','Percent','Age','WHERE','SmokingStatus','Sex']
tr=tr.drop(columns,axis=1)
tr.head()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CtscanDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True,  meta_features = None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age
            
        """
        self.df = df
        self.imfolder = imfolder        
        self.train = train
        self.meta_features = meta_features
        
    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['Patient'] + '.dcm')
        patient=self.df.iloc[index]['Patient']
        x = process_data(patient,self.imfolder)
        
        try:
            meta = np.array(self.df.iloc[index,1:].values, dtype=np.float32)
            meta=torch.tensor(meta,dtype=torch.float32)
            x=torch.tensor(x,dtype=torch.float32)
            if self.train:
                y = self.df.iloc[index]['FVC']
                y=torch.tensor(y,dtype=torch.float32)
                return (x, meta), y
            else:
                return (x, meta)
        except:
            print('error')
    
    def __len__(self):
        return len(self.df)
train_1 = CtscanDataset(df=tr.loc[tr['Patient'].isin(samp)],
                       imfolder='../input/osic-pulmonary-fibrosis-progression/train/', 
                       train=True,  )
train_1[0][0][0].shape
class Model(nn.Module):

    def __init__(self):

        
        super(Model, self).__init__()

        self.conv_layer1 = self._make_conv_layer(1, 32)
        self.conv_layer2 = self._make_conv_layer(32, 64)
        self.conv_layer4 = self._make_conv_layer(64, 256)
        self.conv_layer5=nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=0)
        
        self.fc5 = nn.Linear(4096, 256)
        self.relu = nn.LeakyReLU()
        self.batch0=nn.BatchNorm1d(256)
        self.drop=nn.Dropout(p=0.15)        
        self.fc6 = nn.Linear(256, 124)
        self.relu = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(124)
        
        self.drop=nn.Dropout(p=0.15)
        self.fc7 = nn.Linear(124, 128)
        
        
        self.layer1 = nn.Linear(16,128)
        self.relu1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(128,128)
        self.relu2 = nn.LeakyReLU()
        self.out1 = nn.Linear(256,3)
        self.relu3 = nn.ReLU()
        self.out2 = nn.Linear(256,3)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(2, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(2, 3, 3), padding=1),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, inputs):
        #print(x.size())
        x,meta=inputs
        meta = self.relu1(self.layer1(meta))
        meta = self.relu2(self.layer2(meta))
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        x = self.conv_layer4(x)
        #print(x.size())
        x=self.conv_layer5(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.batch1(x)
        x = self.drop(x)
        x = self.fc7(x)
        feat = torch.cat((x, meta), dim=1)
        o1 = self.out1(feat)
        o2 = F.relu(self.out2(feat))
        return o1 + torch.cumsum(o2,dim=1)


def score(outputs,target):
    confidence = outputs[:,2] - outputs[:,0]
    clip = torch.clamp(confidence,min=70)
    target=torch.reshape(target,outputs[:,1].shape)
    delta = torch.abs(outputs[:,1] - target)
    delta = torch.clamp(delta,max=1000)
    sqrt_2 = torch.sqrt(torch.tensor([2.])).to(device)
    metrics = (delta*sqrt_2/clip) + torch.log(clip*sqrt_2)
    return torch.mean(metrics)

def qloss(outputs,target):
    qs = [0.25,0.5,0.75]
    qs = torch.tensor(qs,dtype=torch.float).to(device)
    e =  target - outputs
    e.to(device)
    v = torch.max(qs*e,(qs-1)*e)
    v = torch.sum(v,dim=1)
    return torch.mean(v)

def loss_fn(outputs,target,l):
    return l * qloss(outputs,target) + (1- l) * score(outputs,target)

def train_loop(train_loader,model,loss_fn,device,optimizer,lr_scheduler=None):
    model.train()
    losses = list()
    metrics = list()
    for i, (data, labels) in enumerate(train_loader):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):           
            outputs = model(data)                 
            metric = score(outputs,labels)

            loss = loss_fn(outputs,labels,0.8)
            metrics.append(metric.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())

            loss.backward()

            optimizer.step()
            if lr_scheduler != None:
                lr_scheduler.step()

    return losses,metrics

def valid_loop(valid_loader,model,loss_fn,device):
    model.eval()
    losses = list()
    metrics = list()
    for i, (inputs, labels) in enumerate(valid_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)                 
        metric = score(outputs,labels)

        loss = loss_fn(outputs,labels,0.8)
        metrics.append(metric.cpu().detach().numpy())
        losses.append(loss.cpu().detach().numpy())

    return losses,metrics    

batch_size =16
train_1 = CtscanDataset(df=tr.loc[tr['Patient'].isin(samp)][:3],
                       imfolder='../input/osic-pulmonary-fibrosis-progression/train/', 
                       train=True,  )
train_loader = torch.utils.data.DataLoader(dataset=train_1,
                                           batch_size=batch_size, shuffle=True)

import torch.optim as optim
n_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model.to(device)
optimizer = optim.Adam(model.parameters(),lr=0.1)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

for epoch in range(n_epochs):
    print(epoch)
    train_loop(train_loader,model,loss_fn,device,optimizer,lr_scheduler)
    #evaluate(train_loader)