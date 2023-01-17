# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
!pip install imutils
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torchvision import models
from PIL import Image 
# url='https://i.pinimg.com/originals/99/0d/66/990d660230e267c11d9d29d913ef696a.jpg'
url='https://m.media-amazon.com/images/M/MV5BMjI0MTg3MzI0M15BMl5BanBnXkFtZTcwMzQyODU2Mw@@._V1_UY1200_CR130,0,630,1200_AL_.jpg'


style_img_url='https://orabart.weebly.com/uploads/8/6/9/2/8692977/one-point-perspective-example-7.jpg'

from PIL import Image
import urllib.request as urllib
import io

fd = urllib.urlopen(url)
image_file = io.BytesIO(fd.read())
img = Image.open(image_file)

fd = urllib.urlopen(style_img_url)
image_file = io.BytesIO(fd.read())
style_img = Image.open(image_file)

plt.imshow(img)
plt.imshow(style_img)
vgg=models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
def transformation(img):
    transform=transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
        
    ])
    img=transform(img)
    return img.unsqueeze(0)
img=transformation(img).to(device)
style_img=transformation(style_img).to(device)
def tensor_to_image(tensor):
    #copy the tensor and remove it from gradient calculation
    img=tensor.clone().detach()
    #squeeze removes the extra dimensions
    img=img.cpu().numpy().squeeze()
    #to change the color channel to be last, after height and width
    img=img.transpose(1,2,0)
    #restoring the original image from standardized form
    img*=np.array(std)+np.array(mean)
    img=img.clip(0,1)
#     img=Image.fromarray(img).convert('RGB')
    return img
tensor_to_image(img).shape
plt.imshow(tensor_to_image(img))
#Using the first and last convolution layers to extract granular and complex details respectively
LAYERS_OF_INTREST={
    '0':'conv1_1',
    '5':'conv2_1',
    '10':'conv3_1',
    '19':'conv4_1',
    '21':'conv4_2',
    '28':'conv5_1'
}
def apply_model_and_extract_features(image,model):
    x=image
    
    features={}
    for name,layer in model._modules.items():
        x=layer(x)
        if name in LAYERS_OF_INTREST:
            features[LAYERS_OF_INTREST[name]]=x
    
    return features
content_img_features=apply_model_and_extract_features(img,vgg)
style_img_features=apply_model_and_extract_features(style_img,vgg)
def calculate_gram_matrix(tensor):
    _,channels,height,width=tensor.size()
    tensor=tensor.view(channels,height*width)
    gram_matrix=torch.mm(tensor,tensor.t())
    gram_matrix=gram_matrix.div(channels*height*width)
    return gram_matrix
style_feature_gram_matrix={layer: calculate_gram_matrix(style_img_features[layer]) for layer in style_img_features}
style_feature_gram_matrix
weights={'conv1_1':1,'conv2_1':0.75,'conv3_1':0.35,'conv4_1':0.25,'conv5_1':0.15}
target=img.clone().requires_grad_(True).to(device)
optimizer=torch.optim.Adam([target],lr=0.03)
for i in range(1,2000):
    target_features=apply_model_and_extract_features(target,vgg)
    
    content_loss=F.mse_loss(target_features['conv4_2'],content_img_features['conv4_2'])
    style_loss=0
    
    for layer in weights:
        target_feature=target_features[layer]
        
        target_gram_matrix=calculate_gram_matrix(target_feature)
        style_gram_matrix=style_feature_gram_matrix[layer]
        
        layer_loss=F.mse_loss(target_gram_matrix,style_gram_matrix)
        layer_loss*=weights[layer]
        _,channels,height,width=target_feature.shape
        
        style_loss+=layer_loss
        
    total_loss=1000000*style_loss +content_loss
    
    if i % 50 ==0:
        print(f'Epoch {i} ,Style_loss : {style_loss} , Content loss:{content_loss}')
              
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
ax1.imshow(tensor_to_image(img))
ax2.imshow(tensor_to_image(target))
