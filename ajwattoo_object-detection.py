# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
cd /kaggle/input/visiolab

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import torch
import pandas as pd
images = glob.glob('/kaggle/input/visiolab/train/images/*.jpg')
X_train=[]
for im in images:
    img= cv2.imread(im)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(int(512),int(512)))
    img = img/255
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    X_train.append(img)
X=(np.array(X_train))
print(X.shape)
plt.imshow(X[12],'gray')
plt.show()
print(img.shape)
labels= glob.glob('/kaggle/input/visiolab/train/labels/*.txt')
target=[]
for file in labels:
    with open(file, 'r') as new_file:
        infoFile =pd.read_csv(new_file) #Reading all the lines from File
        for line in infoFile:#Reading line-by-line
            words = line.split(" ")
            cx=(float(words[1]))
            cy=(float(words[2]))
            w=(float(words[3]))
            h=(float(words[4]))
            target.append([cx,cy,w,h])
target =np.array(target)
print(target.shape)
size=512
target=size*target
print(target[0])
final=[]
for i,t in zip(X,target):
    cx=t[0]
    cy=t[1]  
    w=t[2]  
    h=t[3]
    x1,y1 = (cx - w/2),(cy - h/2)
    img = cv2.rectangle(i, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)
    final.append(img)
final=np.array(final)
print(final.shape)
plt.imshow(final[12])
plt.show()
print(img.shape)
import time
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile
# fix bugs with loading png files
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset, random_split
def iou(X,Y, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    bs,_ = X.shape
    res = []
    for i in range(bs):
        a,b = X[i],Y[i]
        # COORDINATES OF THE INTERSECTION BOX
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        # AREA OF OVERLAP - Area where the boxes intersect
        width = (x2 - x1)
        height = (y2 - y1)
        # handle case where there is NO overlap
        if (width<0).any() or (height <0).any():
            res.append(0.0)
        area_overlap = width * height

        # COMBINED AREA
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        area_combined = area_a + area_b - area_overlap

        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = abs(area_overlap / (area_combined+epsilon))
        res.append(iou)
        
    return np.mean(np.abs(np.array(res)))
final= final.reshape(233, 3,512,512)
#X_train  = torch.tensor(X_train)

target= target.astype(int);
#y_train= torch.tensor(y_train)
print(target.shape)
from torch.utils.data import TensorDataset
train_dataset = TensorDataset(torch.from_numpy(final),torch.from_numpy(target))
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=4,pin_memory=True, drop_last=True)
type(train_loader)
class SimpleNet(nn.Module):   
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=1),
            nn.Conv2d(32,32,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(128,256 , kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256,256 , kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256,512 , kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512,512 , kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(512* 15* 15, 4),
            nn.Sigmoid()
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        #print(x.shape)

        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
torch.manual_seed(42)

#ts = LocTransform()

#ds = PennFudanDataset(ts)

#loader = DataLoader(ds, batch_size=4, pin_memory=True, drop_last=True) 
epochs = 50

device = torch.device("cuda")
model =SimpleNet()
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
st = time.time()
model.train()
for e in range(epochs):
    t = 0
    loss_avg = 0
    iou_avg = 0
    for i,(img, target) in enumerate(train_loader):
        img=img.to(device,dtype=torch.float)
        target = target.to(device,dtype=torch.float)
        
        # before each iteration we clear all gradients calculated before 
        optimizer.zero_grad()
        output = model(img)
        loss = F.mse_loss(output, target)
        # calculating gradients
        loss.backward()
        # optimizing
        optimizer.step()
        # .item() is simple way to get value from scalar
        loss_avg += loss.item()
        # detach - returns new tensor detached from computation graph
        # to get data from gpu first we need to move it to cpu
        iou_avg += iou(target.cpu().detach().numpy(), output.cpu().detach().numpy())
        t += 1
    print(f'Epoch: {e}/{epochs} Loss:{loss_avg/t} IoU:{iou_avg/t}')
    # saving state of the model
    torch.save(model.state_dict(), 'loc.pth')
print('Training time:', time.time()-st)
import imageio
!pip install imageio-ffmpeg
def test(model_path, model_class, input, num=30,size=(512,512), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    state = torch.load(model_path)
    model = model_class()
    model.load_state_dict(state)
    # don't forget about this in other case Dropouts, BatchNorms, etc will not 
    # work correctly
    model.eval()
    img_ts = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(), # converts to [0,1] interval
        transforms.Normalize(mean=mean, std=std)
    ])  
    t = 0
    iou_avg = 0
    if isinstance(input, Image.Image):
        input = [[input, None]]
    for img,target in input:
        #img=np.array(img)
        #print(img.shape)
        #img = torch.tensor(a)
        img_t = img_ts(img)
        #img_t= img_t.reshape(3,512,512)
        img_t = img_t.view((1,)+img_t.shape)
        #print(img_t.shape)
        st = time.time()
        res = model(img_t)
        res=res*512/2
        (xmin, ymin, xmax, ymax) = res.detach().numpy()[0]
        xmin=int(xmin)
        ymin=int(ymin)
        xmax=int(xmax)
        ymax=int(ymax)
    d = ImageDraw.Draw(img)
    d.rectangle([(xmin),(ymin),(xmax),(ymax)], outline=(0,0,255), width=3)
    #a=cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),2)
    plt.imshow(img)
    plt.show()
test('loc.pth',SimpleNet,Image.open('/kaggle/input/visiolab/test/images/img_41.jpg') )
reader = imageio.get_reader('/kaggle/input/visiolab/videos/20200313-200602.avi')
fps = reader.get_meta_data()['fps']
Writer = imageio.get_writer('new.avi',fps=fps)
for i,frame in enumerate(reader):
    frame = test("loc.pth",SimpleNet,frame)
    #Writer.append_data(imageio.imread(frame))
    Writer.append_data(frame)
    print(i)
Writer.close()

ls /kaggle/input/visiolab/test/images/img_0.jpg
