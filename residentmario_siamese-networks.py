import matplotlib.pyplot as plt

import numpy as np

import random

# from PIL import Image

# import PIL.ImageOps    



import torch

from torch.autograd import Variable

import torch.nn as nn

from torch import optim

import torch.nn.functional as F

import torchvision

import torchvision.datasets as dset

import torchvision.transforms as transforms

from torch.utils.data import DataLoader,Dataset

import torchvision.utils





# --------------------------------------------------------------------

# torch.nn.Module is the base class for all neural network modules;

# all PyTorch models are expected to inherit from it. This is the

# object-functional equivalent to the purely functional Keras Model()

# class definition.

#

# Model is expected to implement two methods: __init__ and forward.

# The former handles initialization, the latter, a single pass through

# training.

class SiameseNetwork(nn.Module):

    def __init__(self):

        super(SiameseNetwork, self).__init__()

        

        # PyTorch, like Keras, has a sequential API. Unlike Keras,

        # PyTorch allows for constructing the layer sequence from

        # a single input list.

        self.cnn1 = nn.Sequential(

            # Pads the input tensor using the reflection of the

            # input boundary. E.g. the last row will be mirrored to a

            # new additional last row, the last column will be

            # mirrored to a new additional last column, etcetera.

            #

            # This bit of padding is being used in place of padding=1

            # in the Conv2d layer. Padding ensures that the output of

            # the layer has the same dimensions as the input, as the

            # feature map is applied to every pixel in the input

            # image.

            #

            # Padding should be (kernel_size - 1) / 2. With a

            # kernel_size of 3, a padding value of 1 is appropriate.

            nn.ReflectionPad2d(1),

            nn.Conv2d(

                # input is black and white

                in_channels=1,

                # 4x simult feature maps, equivalent to node count,

                # the first parameter in keras

                out_channels=4,

                # 3x3 feature maps

                kernel_size=3

                # params set to their default value:

                # dilation=1 (none)

                # stride=1 (single-pixel window)

                # padding=0 (no zero-padding; see ReflectionPad2d)

            ),

            # Applies ReLU activation functions in the Conv2d 

            # nodes. inplace=True specifies that the operation is

            # to be performed in-place in-memory. This saves a

            # small amount of memory, but is not a valid operation

            # if the Conv2d layer has multiple consumer layers.

            # inplace=False by default.

            nn.ReLU(inplace=True),

            # Batch normalization is the technique of normalizing

            # the hidden vectors in a batch or mini-batch. Batch

            # normalization is performed on a vector component

            # basis, e.g. every component of the vectors in a batch

            # is scaled against that same component in its neighboring

            # vectors.

            #

            # Batch normalization has been shown to improve training

            # accuracy and speed in practice. The mechanism by which

            # it does so is not fully known, but it is believed to be

            # because it acts to smooth the error surface.

            #

            # Batch normalization is a defense against exploding or

            # vanishing gradients.

            nn.BatchNorm2d(

                # num_features is confusingly named. Should be the node

                # count in the case of a CNN.

                num_features=4

            ),

            # Dropout2d drops entire feature maps at once. This is

            # used instead of Dropout because a feature map consists

            # of a large number of nodes, one per pixel, which are

            # highly correlated with one another (as images are

            # highly spatially correlated). Thus using "regular"

            # dropout would serve only to reduce the accuracy of the

            # model.

            #

            # .2 is a conservative dropout rate. For hidden layers, .5

            # is a common, more aggressive dropout rate.

            nn.Dropout2d(p=.2),

            

            # Two more layers of convolutional feature maps follow.

            # No pooling is applied. This is a curious decision, when

            # you're building a CNN. Pooling layers are not strictly

            # necessary, but, eh.

            nn.ReflectionPad2d(1),

            nn.Conv2d(4, 8, kernel_size=3),

            nn.ReLU(inplace=True),

            nn.BatchNorm2d(8),

            nn.Dropout2d(p=.2),



            nn.ReflectionPad2d(1),

            nn.Conv2d(8, 8, kernel_size=3),

            nn.ReLU(inplace=True),

            nn.BatchNorm2d(8),

            nn.Dropout2d(p=.2)

        )



        # Finally, define the fully-connected layers. The Linear

        # layer takes two inputs, unlike the Keras version (Dense):

        # you have the specify the input_size as well as the output

        # size (the latter being the node count).

        self.fc1 = nn.Sequential(

            nn.Linear(8*100*100, 500),

            nn.ReLU(inplace=True),

            nn.Linear(500, 500),

            nn.ReLU(inplace=True),

            nn.Linear(500, 5)

        )



    # Helper method for forward, not API-required.

    def forward_once(self, x):

        output = self.cnn1(x)

        # The CNN outputs a feature map matrix, but the fully

        # connected layer expects vectorized linear input. So we have

        # to do the equivalent of np.ravel() to the output. In Keras

        # this was done using a Reshape layer. PyTorch doesn't have

        # an equivalent layer-based operation (!?) so we have to do

        # that glue transform at propogation time.

        output = output.view(output.size()[0], -1)

        output = self.fc1(output)

        return output



    # Handles one pass through the model.

    def forward(self, input1, input2):

        output1 = self.forward_once(input1)

        output2 = self.forward_once(input2)

        return output1, output2



# To use the GPU you execute .cuda() on the model constructor.

clf = SiameseNetwork().cuda()



# The optimizer takes parameters as input, in addition to the

# learning rate. parameters is a container iterable with a list of

# layer objects contained inside. This means that in PyTorch to

# freeze layers for example you do something like the following:

#     for param in model.parameters():

#         param.requires_grad = False

optimizer = optim.Adam(clf.parameters(), lr=0.0005)



# ContrastiveLoss is built into PyTorch! This is not true

# of Keras, so this is nice to have.

loss_func = ContrastiveLoss()



# Now for training...

for epoch in range(0, train_number_epochs):

    # train_dataloader is just an iterable that serves train

    # data. Not defined here; see the original code. Note that

    # you can 

    for i, data in enumerate(train_dataloader, 0):

        # Variables are wrapped in a Variable object, and you

        # have to call .cuda() on them to signal GPU training.

        img0, img1, label = data

        img0, img1, label = (

            Variable(img0).cuda(), Variable(img1).cuda() ,

            Variable(label).cuda()

        )

        

        # In PyTorch every step of backpropagation is handled

        # separately:

        #

        # 1. Perform the forward pass.

        # 2. Set the accumulated gradient to zero. *

        # 3. Apply the loss function to determine loss.

        # 4. Calculate the gradients (backwards).

        # 5. Backpropogate the gradients (step).

        #

        # * Setting the gradient to zero with zero_grad is

        #   required because PyTorch supports accumulating

        #   the gradient across multiple batches.

        #

        # The Siamese model is trained using stochastic mini-batches,

        # e.g. batches with a batch size of 1. In other contexts we

        # would batch the images ourselves coming out of the training

        # data loader.

        output1, output2 = clf(img0, img1)

        optimizer.zero_grad()

        

        loss_contrastive = loss_func(output1, output2, label)

        loss_contrastive.backward()

        optimizer.step()