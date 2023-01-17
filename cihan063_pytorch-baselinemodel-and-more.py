import torch

import torch.nn as nn

from PIL import Image

from pathlib import Path

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import math

import cv2

from scipy import signal


import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
image1 = np.array(Image.open('../input/100-bird-species/train/AFRICAN FIREFINCH/001.jpg').convert('L'))  #convert image to black and white

plt.imshow(image1, cmap='gray')
np.array(image1).shape
image1_1 = np.array(Image.open('../input/100-bird-species/train/AFRICAN FIREFINCH/002.jpg').convert('L'))  #convert image to black and white

image1_1 = cv2.Canny(image1_1,224,224)

plt.imshow(image1_1, cmap='gray')
kernel = np.array([[ 0, 1, 0],

                   [ 1,-4, 1],

                   [ 0, 1, 0],]) 



grad = signal.convolve2d(image1_1, kernel, mode='same', boundary='symm')

print('GRADIENT MAGNITUDE - Feature map')



fig, aux = plt.subplots(figsize=(10, 8))

aux.imshow(np.absolute(grad), cmap='gray');
df=pd.DataFrame(np.array(image1))

df.style.set_properties().background_gradient("Greys")
df=pd.DataFrame(np.array(image1_1))

df.style.set_properties().background_gradient("Greys")
PATH=Path('../input/100-bird-species')
list(PATH.iterdir())
Path.ls = lambda x: list(x.iterdir())
(PATH/"train").ls()
(PATH/"train/AFRICAN FIREFINCH").ls()
Image.open((PATH/"train/AFRICAN FIREFINCH").ls()[0])
img1=torch.tensor(np.array(Image.open('../input/100-bird-species/train/AFRICAN FIREFINCH/001.jpg').convert('L')), dtype = torch.float32)

plt.figure(figsize=(7,7))

plt.imshow(img1);
albatross=[torch.tensor(np.array(Image.open(img).convert('L')), dtype = torch.float32) for img in (PATH/"train/ALBATROSS").ls()]

antbird=[torch.tensor(np.array(Image.open(img).convert('L')), dtype = torch.float32) for img in (PATH/"train/ANTBIRD").ls()]
albatross_1=[torch.tensor(cv2.Canny(np.array(Image.open(img).convert('L')),224,224), dtype = torch.float32) for img in (PATH/"train/ALBATROSS").ls()]

antbird_1=[torch.tensor(cv2.Canny(np.array(Image.open(img).convert('L')),224,224), dtype = torch.float32) for img in (PATH/"train/ANTBIRD").ls()]
plt.imshow(albatross[1])
plt.imshow(albatross_1[1])
plt.imshow(antbird[0])
plt.imshow(antbird_1[0])
albatross_stacked = torch.stack(albatross)/255
albatross_stacked.shape
albatross_1_stacked =torch.stack(albatross_1)/255
albatross_1_stacked.shape
antbird_stacked = torch.stack(antbird)/255
antbird_stacked.shape
antbird_1_stacked = torch.stack(antbird_1)/255
antbird_1_stacked.shape
valid_albatross=[torch.tensor(np.array(Image.open(img).convert('L')), dtype = torch.float32) for img in (PATH/"valid/ALBATROSS").ls()]

valid_antbird=[torch.tensor(np.array(Image.open(img).convert('L')), dtype = torch.float32) for img in (PATH/"valid/ANTBIRD").ls()]
avr_albatross= albatross_stacked.mean(0)
avr_antbird= antbird_stacked.mean(0)
plt.imshow(avr_albatross, cmap='gray')
plt.imshow(avr_antbird, cmap='gray');
avr_1_albatross= albatross_1_stacked.mean(0)
avr_1_antbird= antbird_1_stacked.mean(0)
plt.imshow(avr_1_albatross, cmap='gray')
plt.imshow(avr_1_antbird, cmap='gray');
sample_albatross= albatross_stacked[1]
sample_albatross_1 = albatross_1_stacked[1]
plt.imshow(sample_albatross, cmap = "gray");
dist_to_albatross = ((sample_albatross - avr_albatross)**2).mean().sqrt()
dist_to_antbird = ((sample_albatross - avr_antbird)**2).mean().sqrt()
dist_to_albatross_1 = ((sample_albatross - avr_1_albatross)**2).mean().sqrt()
dist_to_antbird_1 = ((sample_albatross - avr_1_antbird)**2).mean().sqrt()
print(dist_to_albatross.item(),dist_to_albatross_1.item())
print(dist_to_antbird.item(),dist_to_antbird_1.item())
dist_to_albatross_1_1 = ((sample_albatross_1 - avr_albatross)**2).mean().sqrt()
dist_to_albatross_1_2 = ((sample_albatross_1 - avr_1_albatross)**2).mean().sqrt()
print(dist_to_albatross_1_1.item(),dist_to_albatross_1_2.item())
dist_to_antbird_1_1 = ((sample_albatross_1 - avr_antbird)**2).mean().sqrt()
dist_to_antbird_1_2 = ((sample_albatross_1 - avr_1_antbird)**2).mean().sqrt()
print(dist_to_antbird_1_1.item(),dist_to_antbird_1_2.item())
def distance(a, b):

    return ((a - b)**2).mean((-1,-2)).sqrt()
distance(sample_albatross, avr_albatross)
valid_dist_albatross= distance(albatross_stacked,avr_albatross)

valid_dist_albatross, valid_dist_albatross.shape
def is_albatross(x):

    return distance(x, avr_albatross) < distance(x, avr_antbird)
distance(sample_albatross,avr_albatross)
distance(sample_albatross,avr_antbird)
is_albatross(sample_albatross)
type(sample_albatross)
for i in range(len(valid_albatross)):

    print(is_albatross(valid_albatross[i]))
for i in range(len(valid_antbird)):

    print(is_albatross(valid_antbird[i]))
distance(valid_antbird[4],avr_albatross)
distance(valid_antbird[4],avr_antbird)
arr1=[]

for i in range(len(valid_albatross)):

    arr1.append(is_albatross(valid_albatross[i].mean()*0.01/2))  

arr1    
arr2=[]

for i in range(len(valid_antbird)):

    arr2.append(is_albatross(valid_antbird[i].mean()*0.01/2))

arr2   
avr_albatross.mean()
avr_antbird.mean()
valid_antbird[1].mean()*0.01/2
print(valid_albatross[0].mean())

print((valid_albatross[0].mean() * 0.01 / 2))
def accuracy_1(a,b):

    i=0

    true_num=0

    false_num=0

    first_loop=0

    second_loop=0

    

    a= np.array(a)

    for i in range(len(a)):

        if a[i] == True:

            true_num = true_num + 1

    first_loop= true_num / len(a)

    

    

    b= np.array(b)

    for i in range(len(b)):

        if b[i]==False:

            false_num = false_num + 1

    second_loop= false_num / len(b)

    

    

    res = (first_loop+second_loop)/2

    return res

    
baseline1_accuracy = accuracy_1(arr1,arr2)

baseline1_accuracy
def is_albatross_1(x):

    return distance(x, avr_1_albatross) < distance(x, avr_1_antbird)
avr_1_albatross.mean()
avr_1_antbird.mean()
valid_albatross[i].mean()*0.01/2
arr11=[]

for i in range(len(valid_albatross)):

    arr11.append(is_albatross(valid_albatross[i]*0.01/2))  

arr11  
arr12=[]

for i in range(len(valid_albatross)):

    arr12.append(is_albatross(valid_antbird[i]*0.01/2))  

arr12  
baseline2_accuracy = accuracy_1(arr11,arr12)

baseline2_accuracy
x = torch.tensor(2.).requires_grad_()
x
def f(x):

    return x**2
grad = f(x)

grad
x.grad
labels = {1:"Albatross", 0:"Antbird"}
train_x = torch.cat([albatross_stacked, antbird_stacked]).view(-1, 224*224)
train_y = torch.tensor([1] * len(albatross) + [0] * len(antbird))
train_x
train_y
train_x.shape, train_y.shape
train_y.unsqueeze_(-1)
train_y.shape
valid_albatross_stacked = torch.stack(valid_albatross)
valid_antbird_stacked = torch.stack(valid_antbird)
valid_albatross_stacked.shape
train_x.shape
valid_x = torch.cat([valid_albatross_stacked, valid_antbird_stacked]).view(-1, 224*224)
valid_x.shape
valid_y = torch.tensor([1] * len(valid_albatross_stacked) + [0] * len(valid_antbird_stacked))  



# 1 = Albatross 

# 0 = Antbird
valid_y.shape
valid_y.unsqueeze(0)
valid_y.unsqueeze(0).shape
valid_y.unsqueeze(1)
valid_y.unsqueeze(1).shape
valid_y = valid_y.unsqueeze(1)
ds_train = list(zip(train_x, train_y))

ds_valid = list(zip(valid_x, valid_y))
ds_train[0]
plt.imshow(ds_train[0][0].view(224,224), cmap="gray");
torch.randn(5)
torch.randn(5,2)
def init(size):

    return torch.randn(size, dtype=torch.float32).requires_grad_()
w = init((224*224,1))
w.shape
b = init(1)
train_x[0].shape
w.shape
(train_x[0] * w.T).sum() + b
def linear_layer(xb):

    return xb @ w + b
preds = linear_layer(train_x)
preds.shape
def accuracy(preds, actuals):

    return ((preds > 0.0).float() == actuals).float().mean().item()
accuracy(preds, train_y)
w[0] = w[0] * 1.0001
preds = linear_layer(train_x)
accuracy(preds, train_y)
def loss(preds, targets):

    return torch.where(targets==1, 1-preds, preds).mean()
def sigmoid(x):

    return 1/(1+torch.exp(-x))
def loss_func(preds, targets):

    preds = preds.sigmoid()

    return torch.where(targets==1, 1-preds, preds).mean()
def f(x):

    for i in range(x):

        yield i
t1=f(5)
next(t1)
w = init((224*224,1))
b = init(1)
class DataLoader():

    def __init__(self, ds, bs): 

        self.ds, self.bs = ds, bs

    def __iter__(self):

        n = len(self.ds)

        l = torch.randperm(n)



        

        for i in range(0, n, self.bs): 

            idxs_l = l[i:i+self.bs]

            yield self.ds[idxs_l]
train_dl = DataLoader(ds_train, bs = 20)