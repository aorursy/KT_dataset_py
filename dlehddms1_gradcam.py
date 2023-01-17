# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install pretrainedmodels
import torch

import pretrainedmodels

from torch.autograd import Function

from torchvision import models

from torchvision import utils

import cv2

import sys

import numpy as np

import argparse

import os

i=0



from torch import nn





image = []

class FeatureExtractor():

#     """ Class for extracting activations and

#     registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):

        self.model = model

        self.target_layers = target_layers

        self.gradients = []

    def save_gradient(self, grad):

        self.gradients.append(grad)

    def __call__(self, x):

        outputs = []

        self.gradients = []

        for name, module in self.model._modules.items():

            print('name=',name)

            print('x.size()=',x.size())

            x = module(x)

            if name in self.target_layers:

                x.register_hook(self.save_gradient)

                outputs += [x]

        return outputs, x

    

class ModelOutputs():

#      """ Class for making a forward pass, and getting:

#      1. The network output.

#      2. Activations from intermeddiate targetted layers.

#      3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers,use_cuda):

        self.model = model

        self.feature_extractor = FeatureExtractor(self.model, target_layers)

        self.cuda = use_cuda

    def get_gradients(self):

        return self.feature_extractor.gradients

    def __call__(self, x):

        target_activations, output  = self.feature_extractor(x)

        output = output.view(output.size(0), -1)

        print('classfier=',output.size())

        if self.cuda:

            output = resnet_or.last_linear(output).cuda()

        else:

            output = resnet_or.last_linear(output)

        return target_activations, output

    

    

def preprocess_image(img):

    means = [0.485, 0.456, 0.406]

    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]

    for i in range(3):

        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]

        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    preprocessed_img = torch.from_numpy(preprocessed_img)

    preprocessed_img.unsqueeze_(0)

    input = torch.Tensor(preprocessed_img)

    return input

    

def show_cam_on_image(img, mask,name):

    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(img)

    cam = cam / np.max(cam)

    concat = np.concatenate((img, cam), axis=1)

    cv2.imwrite("./grad_cam/grad_{}.jpg".format(name), np.uint8(255 * concat))



class GradCam:

    def __init__(self, model, target_layer_names, use_cuda):

        self.model = model

        self.model.eval()

        self.cuda = use_cuda

        if self.cuda:

            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

        

    def forward(self, input):

          return self.model(input)

    

    def __call__(self, input, index = None):

        if self.cuda:

            features, output = self.extractor(input.cuda())

        else:

            features, output = self.extractor(input)

        if index == None:

            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)

        one_hot[0][index] = 1

        one_hot = torch.Tensor(torch.from_numpy(one_hot))

        if self.cuda:

            one_hot = torch.sum(one_hot.cuda() * output)

        else:

            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()

        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

  

        target = features[-1]

        target = target.cpu().data.numpy()[0, :]

         

        weights = np.mean(grads_val, axis = (2, 3))[0, :]

        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):

            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)

        cam = cv2.resize(cam, (224, 224))

        cam = cam - np.min(cam)

        cam = cam / np.max(cam)

        return cam

    

class GuidedBackpropReLU(Function):

    def forward(self, input):

        positive_mask = (input > 0).type_as(input)

        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)

        self.save_for_backward(input, output)

        return output

    def backward(self, grad_output):

        input, output = self.saved_tensors

        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)

        positive_mask_2 = (grad_output > 0).type_as(grad_output)

        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

    

class GuidedBackpropReLUModel:

    def __init__(self, model, use_cuda):

        self.model = resnet

        self.model.eval()

        self.cuda = use_cuda

        if self.cuda:

            self.model = model.cuda()



        for idx, module in self.model._modules.items():

           if module.__class__.__name__ == 'ReLU':

                self.model._modules[idx] = GuidedBackpropReLU()

                

    def forward(self, input):

        return self.model(input)

    

    def __call__(self, input, index = None):

        if self.cuda:

            output = self.forward(input.cuda())

        else:

            output = self.forward(input)

        if index == None:

            index = np.argmax(output.cpu().data.numpy())



        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)

        one_hot[0][index] = 1

        one_hot = torch.Tensor(torch.from_numpy(one_hot))

        if self.cuda:

            one_hot = torch.sum(one_hot.cuda() * output)

        else:

            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)

        output = input.grad.cuda().data.numpy()

        output = output[0,:,:,:]

        return output

    



if __name__ == '__main__':

    #  """ python grad_cam.py <path_to_image>

    #  1. Loads an image with opencv.

    #  2. Preprocesses it for VGG19 and converts to a pytorch variable.

    #  3. Makes a forward pass to find the category index with the highest score,

    #  and computes intermediate activations.

    #  Makes the visualization. """



    model_name = 'resnet50'

    res50 = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    num_ftrs = res50.last_linear.in_features

    res50.last_linear = nn.Linear(num_ftrs, 196)

    model = res50

    model.load_state_dict(torch.load('./weight_best.pt'))

    #print(model)



    del model.last_linear

    # Can work with any model, but it assumes that the model has a

    # feature method, and a classifier method,

    # as in the VGG models in torchvision.



    grad_cam = GradCam(model, target_layer_names = ["layer4"], use_cuda=True)

    image_path = './test_crop/'

    x=os.walk(image_path)

    for root,dirs,filename in x:



        pass

    for s in filename:

        image.append(cv2.imread(image_path+s,1))



    for img in image:

        img = np.float32(cv2.resize(img, (224, 224))) / 255

        input = preprocess_image(img)

        print('input.size()=',input.size())

        # If None, returns the map for the highest scoring category.

        # Otherwise, targets the requested index.

        target_index = None

        mask = grad_cam(input, target_index)

        #print(type(mask))

        i=i+1

        show_cam_on_image(img, mask,i)
