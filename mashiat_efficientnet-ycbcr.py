! git clone https://github.com/dwgoon/jpegio
# Once downloaded install the package
!pip install jpegio/.
import jpegio as jpio
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#   for filename in filenames:
#       print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import jpegio as jpio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms,datasets

from tqdm.notebook import tqdm
import sys

REBUILD_DATA= True
def JPEGdecompressYCbCr(jpegStruct):
    
    nb_colors=len(jpegStruct.coef_arrays)           #nb_colors=3=number of channels
    
    [Col,Row] = np.meshgrid( range(8) , range(8) )
    T = 0.5 * np.cos(np.pi * (2*Col + 1) * Row / (2 * 8))
    T[0,:] = T[0,:] / np.sqrt(2)    #T is of shape [8 , 8] 
    
    sz = np.array(jpegStruct.coef_arrays[0].shape)    #shape of coef_arrays[0] is 512,512. So sz is 2D list --512 512
    imDecompressYCbCr = np.zeros([sz[0], sz[1], nb_colors]);    #imgdecompressYCbCr is a 3D array of zeros of size 512,512,3
    szDct = (sz/8).astype('int')     # 2D list --- sz/8 ---- [64 64]

    
    
    
    for ColorChannel in range(nb_colors):
        tmpPixels = np.zeros(sz) #zero 2D vector of size (512, 512)
    
        DCTcoefs = jpegStruct.coef_arrays[ColorChannel]; #DCT Co-efficients of the image's color channel...size [512,512]
        if ColorChannel==0:
            QM = jpegStruct.quant_tables[ColorChannel];
        else:
            QM = jpegStruct.quant_tables[1];     #quantization table maybe
        
        for idxRow in range(szDct[0]):     #range---64
            for idxCol in range(szDct[1]):  #range----64
                D = DCTcoefs[idxRow*8:(idxRow+1)*8 , idxCol*8:(idxCol+1)*8]     #size of D [8, 8]- D takes 8*8 blocks from DCTcoefs
                tmpPixels[idxRow*8:(idxRow+1)*8 , idxCol*8:(idxCol+1)*8] = np.dot( np.transpose(T) , np.dot( QM * D , T ) )
        imDecompressYCbCr[:,:,ColorChannel] = tmpPixels;
    return imDecompressYCbCr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
BASE_PATH = "/kaggle/input/alaska2-image-steganalysis"
train_imageids = pd.Series(os.listdir(BASE_PATH + '/Cover')).sort_values(ascending=True).reset_index(drop=True)
test_imageids = pd.Series(os.listdir(BASE_PATH + '/Test')).sort_values(ascending=True).reset_index(drop=True)
sub = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')
print(test_imageids.head(3))
train_imageids.head()
#https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus
def append_path(pre):
    return np.vectorize(lambda file: os.path.join(BASE_PATH, pre, file))
train_filenames = np.array(os.listdir("/kaggle/input/alaska2-image-steganalysis/Cover/"))
print(len(train_filenames))
train_filenames
#https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus
np.random.seed(0)
positives = train_filenames.copy()
negatives = train_filenames.copy()
np.random.shuffle(positives)
np.random.shuffle(negatives)

jmipod = append_path('JMiPOD')(positives[:10000])
juniward = append_path('JUNIWARD')(positives[10000:20000])
uerd = append_path('UERD')(positives[20000:30000])

pos_paths = np.concatenate([jmipod, juniward, uerd])
test_paths = append_path('Test')(sub.Id.values)
neg_paths = append_path('Cover')(negatives[:30000])
train_paths = np.concatenate([pos_paths, neg_paths])
train_labels = np.array([1] * len(pos_paths) + [0] * len(neg_paths))
print(train_paths)
print(test_paths)
test_labels = np.array( [0] * len(test_paths))
print(len(test_paths))
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
n_splits = 10 # Number of K-fold Splits

splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=4590).split(train_paths, train_labels))
#https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus
'''train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=2020)'''
'''l=np.array([train_paths,train_labels])
traindataset = pd.DataFrame({ 'images': list(train_paths), 'label': train_labels},columns=['images','label'])
testdataset = pd.DataFrame({ 'images': list(test_paths), 'label': test_labels},columns=['images','label'])'''
'''val_l=np.array([valid_paths,valid_labels])
validdataset=dataset = pd.DataFrame({ 'images': list(valid_paths), 'label': valid_labels},columns=['images','label'])'''
'''from PIL import Image, ImageFile
import scipy                        # for cosine similarity
from scipy import fftpack'''
# Image preprocessing modules
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize(512, 512),
    transforms.ToTensor()])
test_transform=transforms.Compose([
    transforms.Resize(512, 512),
    transforms.ToTensor()])
# add image augmen tation
class train_images(torch.utils.data.Dataset):

    def __init__(self, csv_file):

        self.data = csv_file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print(idx)
        img_name =  self.data.loc[idx][0]
        jpegStruct = jpio.read(img_name)
        YCbCr_img=JPEGdecompressYCbCr(jpegStruct)
        #image = Image.open(img_name)
        label = self.data.loc[idx][1] #torch.tensor(self.data.loc[idx, 'label'])
       

# ## https://pytorch.org/docs/stable/torchvision/transforms.html
# transforms.Compose([
# transforms.CenterCrop(10),
# transforms.ToTensor(),
# ])
        
#         return {'image': transforms.ToTensor()(image), # ORIG
        return {'image': YCbCr_img,
            'label': label
            }
# add image augmen tation
class test_images(torch.utils.data.Dataset):

    def __init__(self, csv_file):

        self.data = csv_file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print(idx)
        img_name =  self.data.loc[idx][0]
        jpegStruct = jpio.read(img_name)
        YCbCr_img=JPEGdecompressYCbCr(jpegStruct)
        #image = Image.open(img_name)
       

# ## https://pytorch.org/docs/stable/torchvision/transforms.html
# transforms.Compose([
# transforms.CenterCrop(10),
# transforms.ToTensor(),
# ])
        
#         return {'image': transforms.ToTensor()(image), # ORIG
        return {'image': YCbCr_img
   }
'''train_dataset = train_images(traindataset)
valid_dataset = test_images(validdataset)
print(type(train_dataset))
len(train_dataset)'''
batch_size = 32
epochs = 2
learning_rate = 0.001
# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle=True, num_workers=4)'''
print(os.listdir("../input/"))
package_path = '../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/'
sys.path.append(package_path)
!pip install --upgrade efficientnet-pytorch
from efficientnet_pytorch import EfficientNet
num_classes = 1
model = (EfficientNet.from_name('efficientnet-b0')).to(device)
model.load_state_dict(torch.load('../input/efficientnet-pytorch/efficientnet-b0-08094119.pth'))
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, num_classes)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(model)
for i, (train_idx, valid_idx) in enumerate(splits):
    valid_preds = []
    x_train = np.array(train_paths)
    y_train = np.array(train_labels)   
    #x_train , y_train = augment(x_train , y_train)
    
    x_train_fold = x_train[train_idx.astype(int)]
    y_train_fold = y_train[train_idx.astype(int)]
    
    x_val_fold = x_train[valid_idx.astype(int)]
    y_val_fold = y_train[valid_idx.astype(int)]
    
    
    traindataset = pd.DataFrame({ 'images': list(x_train_fold), 'label': y_train_fold},columns=['images','label'])
    validdataset = pd.DataFrame({ 'images': list(x_val_fold), 'label': y_val_fold},columns=['images','label'])
    traindataset = train_images(traindataset)
    validdataset = test_images(validdataset)
    
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size = batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(validdataset, batch_size = batch_size, shuffle=True, num_workers=4)
    
    total_step = len(train_loader)
    curr_lr = learning_rate
    
    for epoch in range(epochs):

        tk0 = tqdm(train_loader, total=int(len(train_loader)))
        counter = 0
        epoch_loss = 0
        for bi, d in enumerate(tk0):
                inputs = d["image"]
                labels = d["label"].view(-1, 1)
                print(inputs.shape)

                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                #labels = labels.squeeze(1)

                inputs = inputs.view(-1,3,512,512)
                #print(inputs.shape)

                # Forward pass
                outputs = model(inputs)
                loss  = criterion(outputs,labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss =epoch_loss + loss.item()

        epoch_loss = epoch_loss / (len(train_loader)/batch_size)
        print ("Epoch [{}/{}] Loss: {:.4f}".format(epoch+1, epochs, epoch_loss))

        # Decay learning rate
        curr_lr /= 3  
        update_lr(optimizer, curr_lr)
        
        torch.save(model.state_dict(), 'effnet.ckpt')
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            tk0 = tqdm(valid_loader, total=int(len(valid_loader)))
            counter = 0
            for bi, d in enumerate(tk0):
                    inputs = d["image"]
                    labels = d["label"].view(-1, 1)

                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)
                    inputs = inputs.view(-1,3,512,512)

                    outputs = model(inputs)
                    print(outputs.shape)
                    total += labels.size(0)
                    correct += (outputs == labels).sum().item()
                    
            print("correct",correct)
            print("total", total)
            print('Accuracy of the model on the valid images: {} %'.format(100 * correct / total))
        
        
    






total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(epochs):

    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    counter = 0
    epoch_loss = 0
    for bi, d in enumerate(tk0):
            inputs = d["image"]
            labels = d["label"].view(-1, 1)
    
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            #labels = labels.squeeze(1)
        
            inputs = inputs.view(-1,3,512,512)
            #print(inputs.shape)
            
            # Forward pass
            outputs = model(inputs)
            loss  = criterion(outputs, torch.max(labels, 1)[1])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss =epoch_loss + loss.item()
            
    epoch_loss = epoch_loss / (len(train_loader)/batch_size)
    print ("Epoch [{}/{}] Loss: {:.4f}".format(epoch+1, epochs, epoch_loss))

    # Decay learning rate
    curr_lr /= 3  
    update_lr(optimizer, curr_lr)
torch.save(model.state_dict(), 'effnet.ckpt')
model = torch.load('effnet.ckpt')

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    tk0 = tqdm(valid_loader, total=int(len(valid_loader)))
    counter = 0
    for bi, d in enumerate(tk0):
            inputs = d["image"]
            labels = d["label"].view(-1, 1)
    
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            inputs = inputs.view(-1,3,512,512)
   
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted.shape)
            total += labels.size(0)
            correct += (predicted[0] == labels).sum().item()
            for i,val in enumerate(predicted):
                print(val)
    print("correct",correct)
    print("total", total)
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))  
del train_dataset
del train_loader
test_dataset = test_images(testdataset)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=4,
                                          shuffle=False,
                                          drop_last=False)
test_df = pd.DataFrame({'ImageFileName': list(
    test_paths)}, columns=['ImageFileName'])

test_df.head()
model.eval()

preds = []
tk0 = tqdm(test_loader)
with torch.no_grad():
    for i, im in enumerate(tk0):
        inputs = im["image"]
        inputs = inputs.to(device, dtype=torch.float)
        inputs = inputs.view(-1,3,512,512)
        # flip vertical
        im = inputs.flip(2)
        outputs = model(im)
        for i,val in enumerate(outputs):
            print(val)
        # fliplr
        im = inputs.flip(3)
        outputs = (0.25*outputs + 0.25*model(im))
        outputs = (outputs + 0.5*model(inputs))        
        preds.extend(F.softmax(outputs, 1).cpu().numpy())

preds = np.array(preds)
labels = preds.argmax(1)
new_preds = np.zeros((len(preds),))
new_preds[labels != 0] = preds[labels != 0, 1:].sum(1)
new_preds[labels == 0] = 1 - preds[labels == 0, 0]

test_df['Id'] = test_df['ImageFileName'].apply(lambda x: x.split(os.sep)[-1])
test_df['Label'] = new_preds

test_df = test_df.drop('ImageFileName', axis=1)
test_df.to_csv('submission_eb0.csv', index=False)
print(test_df.head())
