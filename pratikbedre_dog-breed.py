import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import torch

from torch.utils.data import Dataset, DataLoader



from torch.nn import Conv2d as conv

from torch.nn import MaxPool2d as maxpool

from torch.nn import MaxUnpool2d as unpool

from torch.nn import ReLU as relu

from torch.nn import Softmax as softmax

from torch.nn import BatchNorm2d as bn

from torch.nn import Dropout as drop

from torch.nn import Dropout2d as drop2d

from torch.nn import Linear as fc

from torch.nn import ConvTranspose2d as convT

from torch.nn import Identity

from torch.nn import MSELoss  as MSE

from torch.nn import Sequential

import albumentations as albu

import torch.optim as optim

from torch.optim import lr_scheduler, Adam

from PIL import Image

from tqdm import tqdm

from albumentations import (HorizontalFlip,VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise,RandomRotate90,Transpose,RandomBrightnessContrast,RandomCrop)

from albumentations.pytorch import ToTensor

from sklearn.preprocessing import LabelEncoder
train_dir = '/kaggle/input/dog-breed-identification/train/'

test_dir = '/kaggle/input/dog-breed-identification/test/'

data = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv' )

# LE = LabelEncoder()

# train_df['breed'] = LE.fit_transform(train_df['breed'])

data.head()
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(data, test_size = 0.2, shuffle = True, random_state = 0)

train_df.head(), val_df.head()

train_encod = pd.get_dummies(train_df['breed'])

val_encod = pd.get_dummies(val_df['breed'])

train_df = pd.concat([train_df, train_encod], axis = 1).drop('breed', axis =1)

val_df = pd.concat([val_df, val_encod], axis = 1).drop('breed', axis =1)

test_df = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')
train_df.head(), val_df.head(), test_df.head()
class Data(Dataset):

    def __init__(self, csv:pd.DataFrame, root_dir,  mode='train' ):

        """

        -csv: pd dataframe

        -root_dir: root_dir for images

        """

        super(Data, self).__init__()

        self.data = csv

        self.root_dir = root_dir

        if mode== 'train':

            self.trans = albu.Compose([

                                        albu.SmallestMaxSize(256),

                                        albu.RandomCrop(256,256),

                                        albu.HorizontalFlip(p=0.5),

                                        albu.Cutout(),

                                        albu.RGBShift(),

                                        albu.HueSaturationValue(),

                                        albu.RandomContrast(),

                                        albu.GaussNoise(),

                                        albu.GaussianBlur(),

                                        albu.RandomBrightnessContrast(),

                                        albu.Rotate(limit=(-90,90)),

                                        albu.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

            ])

        else:

            self.trans = albu.Compose([

                                        albu.Resize(256,256),

                                        albu.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

        ])

            

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        """

        -idx: index of data example



        Returns a dictionary containing

        'rgb' : color image(output)

        'gray' : grayscale image(input)

        """

        

        if torch.is_tensor(idx):

            idx = idx.tolist()

        img_name = self.root_dir+self.data.iloc[idx, 0] +  '.jpg'

        img = Image.open(img_name)

        label = None

        img = self.trans(image=np.array(img))['image']

        img = np.transpose(img,(2,0,1)).astype(np.float32)

        img = torch.tensor(img, dtype = torch.float)

        label =int( np.argmax(self.data.iloc[idx,  1:].to_numpy(), axis = -1))

        out = { 'img' : img, 'label': label}

        

        return out
train_data = Data(train_df, train_dir , mode= 'train')

train_loader = DataLoader(train_data, batch_size = 128)

val_data = Data(val_df, train_dir, mode = 'val') 

val_loader = DataLoader(val_data, batch_size = 128)

test_data = Data(test_df, test_dir, mode = 'test')

test_loader =  DataLoader(test_data, batch_size = 128, shuffle = False)
def grad(model):

    for param in model.parameters():

        param.requires_grad =False
import torch

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)

model = model.apply(grad)
model.fc = Sequential(fc(2048, 1024), 

                      relu(), 

                      fc(1024, 512), 

                      relu(), 

                      fc(512, 120)) 
params = [ p for p in  model.parameters() if p.requires_grad]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay = 0.001)

infnum_epoch = 15

running_loss = 0

train_loss_list = []

val_loss_list = []

prev_loss = float('inf')

num_train = len(train_df)

num_val = len(val_df)
import matplotlib.pyplot as plt
#######################################################################################TRAINING###################################################################################################

for i in tqdm(range(num_epoch)):

    os.system('clear')

    print('*'*75, 'TRAIN PHASE', '*'*75)

    train_acc = 0

    val_acc = 0

    running_train_loss = 0

    val_running_loss = 0

    for j, batch in enumerate(train_loader):

        inputs = batch['img'].to(device)

        output = batch['label'].to(device)

        optimizer.zero_grad()

        y_hat = model(inputs)

        train_loss = criterion(y_hat, output)

        train_loss.backward()

        optimizer.step()

        train_loss_list.append(train_loss.item())

        running_train_loss += train_loss

        label_ = torch.argmax(y_hat, dim = 1)

        train_acc += torch.mean((label_ == output).type('torch.FloatTensor')*1)/num_train

        if j%10 == 9:

            print('[ Epoch: %3d, Mini_batch: %5d  ------------>>>>>>>>>>>>>>    train_loss: %.5f]'%(i+1, (j+1)//10, train_loss))

    print('*'*72, 'VALIDATION PHASE', '*'*72)

    for j, batch in enumerate(val_loader):

        inputs = batch['img'].to(device)

        output = batch['label'].to(device)

        y_hat = model(inputs)

        val_loss = criterion(y_hat, output)

        

        val_running_loss += val_loss.item()

        label_ = torch.argmax(y_hat, dim = 1)

        val_acc += torch.mean((label_ == output).type('torch.FloatTensor')*1)/num_train

        val_loss_list.append(val_loss)

        if j%10 == 9:

            print('[ Epoch: %3d, Mini_batch: %5d------------>>>>>>>>>>>>>>    val_loss: %.5f]'%(i+1, (j+1)//10, val_loss))

    print(' Train Accuracy: {%.3f}, Val Accuracy: {%.3f}'%(train_acc, val_acc))

    if val_running_loss<prev_loss:

        prev_loss = val_running_loss

        torch.save(model.state_dict(), 'model.pth')

    fig,a =  plt.subplots(1,2)

    a[0].set_title('Train Loss')

    a[1].set_title('VAL Loss')

    a[0].plot( 1 + np.arange(len(train_loss_list)), train_loss_list)

    a[1].plot( 1 + np.arange(len(val_loss_list)), val_loss_list)

    plt.show()

    #after calculating error per epoch
    fig,a =  plt.subplots(1,2)

    a[0].set_title('Train Loss')

    a[0].set_title('VAL Loss')

    a[0].plot( 1 + np.arange(len(train_loss_list)), train_loss_list)

    a[0].plot( 1 + np.arange(len(val_loss_list)), val_loss_list)

    plt.show()
torch.save(model.state_dict(), 'model.pth')
ans= []

from torch.nn.functional import softmax

for i in test_loader:

    out = model(i['img'].to(device))

    out = softmax(out, dim = -1)

    out = out.view(-1, 120).cpu().detach().numpy()

    ans.append(out)
len(ans)
ans = np.array(ans)

np.argmax(submission.iloc[0, 1:])
model.load_state_dict(torch.load('/kaggle/working/model.pth'))

from scipy.ndimage.filters import gaussian_filter1d



loss_smoothed = gaussian_filter1d(loss_list, sigma=2)

plt.plot(np.arange(len(loss_list))+1, loss_smoothed)



plt.show()

test_df = pd.read_csv('/kaggle/input/dog-breed-identification/sample_submission.csv')

test_data = Data(test_df, test_dir, train_ds,  mode='test' )

test_loader = DataLoader(test_data, shuffle = False)
sample = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')

sample['breed'].unique()
submission = pd.read_csv('/kaggle/input/dog-breed-identification/sample_submission.csv')
from torch.nn.functional import  *

for i, slic in enumerate(test_loader):

    out_squeeze = np.zeros((1, 120))

    for j in range(num_crops):

        img = test_data[i]['img'][j].to(device)

        out_squeeze += softmax(model(img)).cpu().detach().numpy()

    submission.iloc[i*128 : i*128 + sic.shape[0], 1:] = out_squeeze/num_crops
num_crops =5
submission.to_csv('submission.csv', float_format = '%.6f' , header = submission.columns,  index = False)
import torchvision

dataset = torchvision.datasets.ImageFolder('/kaggle/input/dog-breed-identification/')
test_loader = DataLoader()
labels = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
labels.head()