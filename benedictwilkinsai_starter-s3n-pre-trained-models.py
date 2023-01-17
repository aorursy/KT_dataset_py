# imports

import math

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F



import matplotlib.pyplot as plt



import os

import yaml

import h5py

from pprint import pprint
# Model code

def as_shape(shape):

    if isinstance(shape, tuple):

        return shape

    elif isinstance(shape, list):

        return tuple(shape)

    elif isinstance(shape, int):

        return (shape,)

    else:

        raise ValueError("Invalid shape argument: {0}".format(str(shape)))



def conv_output_shape(input_shape, out_channels, kernel_size=1, stride=1, pad=0, dilation=1):

    '''

        Get the output shape of a convolution given the input_shape.

    '''

    input_shape = as_shape(input_shape)

    h,w = input_shape[-2:]

    if type(kernel_size) is not tuple:

        kernel_size = (kernel_size, kernel_size)

    h = math.floor(((h + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)

    w = math.floor(((w + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)

    return out_channels, h, w



class CNet(nn.Module):

    '''

        A convolutional network that takes as input an image of dimension input_shape = (C,H,W).

    '''

    

    def __init__(self, input_shape, device='cpu'):

        super(CNet, self).__init__() 

        self.input_shape = as_shape(input_shape)

        s1 = conv_output_shape(input_shape, 16, kernel_size=4, stride=2)

        s2 = conv_output_shape(s1, 32, kernel_size=4, stride=1)

        s3 = conv_output_shape(s2, 64, kernel_size=4, stride=1)

    

        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=4, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=1)



        self.output_shape = as_shape(int(np.prod(s3)))



        self.device = device

        self.to(device)

    

    def to(self, device):

        self.device = device

        return super(CNet, self).to(device)



    def forward(self, x_):

        x_ = x_.to(self.device)

        y_ = F.leaky_relu(self.conv1(x_))

        y_ = F.leaky_relu(self.conv2(y_))

        y_ = F.leaky_relu(self.conv3(y_)).view(x_.shape[0], -1)

        return y_

    

# This simple model architecture is used in the original paper.

class CNet2(CNet):

    '''

        A convolutional network based on CNet with a fully connected output layer of given dimension.

    '''

    

    def __init__(self, input_shape, output_shape, output_activation=nn.Identity()):

        super(CNet2, self).__init__(input_shape)

        output_shape = as_shape(output_shape)

        self.out_layer = nn.Linear(self.output_shape[0], output_shape[0])

        self.output_shape = output_shape

        self.output_activation = output_activation



    def forward(self, x_):

        x_ = super(CNet2, self).forward(x_)

        y_ = self.output_activation(self.out_layer(x_))

        return y_
# Load Model and Atari Anomaly Dataset trajectory



path = "/kaggle/input/s3n-pretrained-models/models/dryrun-sssn-Breakout-2-20200209131829"

#path = "/kaggle/input/s3n-pretrained-models/models/dryrun-sssn-Pong-64-20200429154006"

config_path = os.path.join(path, "config.yaml")

model_path = os.path.join(path, "model.pt")



#load config

with open(config_path) as file: 

    config = {k:v['value'] for k,v in yaml.full_load(file).items() if "wandb" not in k}

    # fix issue with Breakout-2 (old config version) to be compatible with new config

    if 'state' not in config:

        config['state'] = {'shape':config['state_shape'], 'dtype':'float32', 'format':['CHW', 'RGB']}

        del config['state_shape']

    

pprint(config)

    

#load model

model = CNet2(config['state']['shape'], config['latent_shape'])

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))



#load example data

episode = 86 # CHANGE ME - to visualise different episodes (and different kinds of anomaly)

path = "/kaggle/input/atari-anomaly-dataset-aad/AAD/anomaly/BreakoutNoFrameskip-v4/episode({0}).hdf5".format(str(episode))

file = h5py.File(path, 'r')

state = file['state'][...] #for use in visualisation

label = file['label'][...].astype(np.uint8)
# Encode the trajectory

def encode(model, state, batch_size=64):

    model.eval()

    Z = [] #prevent memory errors if not logged in

    for i in range(0,state.shape[0],batch_size): 

        Z.append(model(state[i:i+batch_size]).detach().numpy())

    return np.concatenate(Z, axis=0)



_state = torch.from_numpy(state.transpose((0,3,1,2)).astype(np.float32) / 255.) #convert to torch tensor

z = encode(model, _state)
# Interactive visualisation of 2D embedding



import cv2

import plotly.graph_objs as go

from ipywidgets import Image, Layout, HBox



x, y = z[:,0], z[:,1]



scatter_colour = np.array(['blue','red'])[label]

fig = go.FigureWidget(data=[dict(type='scattergl',x=x, y=y,

            mode='lines+markers',

            marker=dict(color=scatter_colour),

            line=dict(color='#b9d1fa'))])

fig.update_layout(autosize=False, width=500, height=500, margin=dict(l=5,b=5,r=5,t=5))



def to_bytes(image):

    _, image = cv2.imencode(".png", image)

    return image.tobytes()



#convert images to png format

scale = 2

image_width = '{0}px'.format(int(state.shape[2] * scale))

image_height = '{0}px'.format(int(state.shape[1] * scale))

print(image_width, image_height)

images = [to_bytes(image) for image in state]



image_widget = Image(value=images[0], layout=Layout(height=image_height, width=image_width))

    

def hover_fn(trace, points, state):

    ind = points.point_inds[0]

    image_widget.value = images[ind]



fig.data[0].on_hover(hover_fn)



display(HBox([fig, image_widget]))