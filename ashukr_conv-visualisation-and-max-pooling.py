import cv2
import matplotlib.pyplot as plt
path =  '../input/udacity_sdc.png'
%matplotlib inline
image = cv2.imread(path)
plt.imshow(image,cmap='gray')
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray_image = gray_image.astype('float32')/255
plt.imshow(gray_image,cmap = 'gray')
import numpy as np
filter_vals = np.array([[-1,-1,1,1],[-1,-1,1,1],[-1,-1,1,1],[-1,-1,1,1]])
filter_vals.shape
filter1  = filter_vals
filter2 = -filter_vals
filter3 = filter_vals.T
filter4 = -filter_vals.T
filters = np.array([filter1,filter2,filter3,filter4])
print(filters.shape)
plt.imshow(filter1,cmap = 'gray')
fig = plt.figure(figsize=(10,5))
for i in range(4):
    ax = fig.add_subplot(1,4,i+1,xticks=[],yticks=[])
    ax.imshow(filters[i],cmap = 'gray')
    print(filters[i])
    width,height=filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]),xy=(y,x))
import torch
import torch.nn as nn
import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self,weight):
        super(Network,self).__init__()
        k_height,k_weight = weight.shape[2:]
        self.conv = nn.Conv2d(1,4,kernel_size = (k_height,k_weight),bias = False)
        self.conv.weight = torch.nn.Parameter(weight)
        
    def forward(self,x):
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)
        return conv_x,activated_x
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
weight.shape
model = Network(weight)
model
# helper function for visualizing the output of a given layer
# default number of filters is 4
def viz_layer(layer, n_filters= 4):
    fig = plt.figure(figsize=(20, 20))
    
    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i+1))
plt.imshow(gray_image,cmap = 'gray')

fig = plt.figure(figsize = (12,6))
fig.subplots_adjust(left=0, right=1.5, bottom=0.8,top=1,hspace=0.05,wspace = 0.05)
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    

gray_image_tensor = torch.from_numpy(gray_image).unsqueeze(0).unsqueeze(1)
gray_image_tensor.shape
convlayer, activated_layer = model(gray_image_tensor)
viz_layer(convlayer)
viz_layer(activated_layer)
class Network(nn.Module):
    def __init__(self,weight):
        super(Network,self).__init__()
        k_height,k_weight = weight.shape[2:]
        self.conv = nn.Conv2d(1,4,kernel_size = (k_height,k_weight),bias = False)
        self.conv.weight = torch.nn.Parameter(weight)
        #adding the maxpooling layer
        self.pool = nn.MaxPool2d(7,3)
        
    def forward(self,x):
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)
        #applying the pooling layer
        pool_x = self.pool(activated_x)
        return conv_x,activated_x,pool_x
model = Network(weight)
convlayer, activated_layer,max_pool  = model(gray_image_tensor)
viz_layer(convlayer)
viz_layer(activated_layer)
viz_layer(max_pool)
max_pool.shape
convlayer.shape
gray_image.shape
activated_layer.shape

