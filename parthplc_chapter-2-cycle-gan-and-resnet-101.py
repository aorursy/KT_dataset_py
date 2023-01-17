from torchvision import models
# list of all models in torchvision

dir(models)
# Using the resnet101 function, we’ll now instantiate a 101-layer convolutional neural network

resnet = models.resnet101(pretrained=True)
#structure of resnet.

# What we are seeing here is modules, one per line. Note that they have nothing in common 

# with Python modules: they are individual operations, the building blocks of a

# neural network. They are also called layers in other deep learning frameworks.

resnet
# The resnet variable can be called like a function, taking as input one or more

# images and producing an equal number of scores for each of the 1,000 ImageNet

# classes. Before we can do that, however, we have to preprocess the input images so

# they are the right size and so that their values (colors) sit roughly in the same numerical 

# range. In order to do that, the torchvision module provides transforms, which

# allow us to quickly define pipelines of basic preprocessing functions:

from torchvision import transforms

preprocess = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(

    mean=[0.485, 0.456, 0.406],

    std=[0.229, 0.224, 0.225]

    )

])
!ls ../input
from PIL import Image

img = Image.open("../input/dogimage/download.jpeg")
img.show()
img
img_t = preprocess(img)
import torch

batch_t = torch.unsqueeze(img_t, 0)
# The process of running a trained model on new data is called inference in deep learning circles. 

# In order to do inference, we need to put the network in eval mode:

resnet.eval()
# If we forget to do that, some pretrained models, like batch normalization and dropout,

# will not produce meaningful answers, just because of the way they work internally.

# Now that eval has been set, we’re ready for inference:

out = resnet(batch_t)
with open('../input/imagenet/imagenet_classes.txt') as f:

    labels = [line.strip() for line in f.readlines()]
# 1,000 labels for the ImageNet dataset classes:

labels
# At this point, we need to determine the index corresponding to the maximum score

# in the out tensor we obtained previously. We can do that using the max function in

# PyTorch, which outputs the maximum value in a tensor as well as the indices where

# that maximum value occurred

_, index = torch.max(out, 1)
# We can now use the index to access the label. Here, index is not a plain Python number,

# but a one-element, one-dimensional tensor (specifically, tensor([207])), so we

# need to get the actual numerical value to use as an index into our labels list using

# index[0]. We also use torch.nn.functional.softmax (http://mng.bz/BYnq) to normalize our 

# outputs to the range [0, 1], and divide by the sum. That gives us something

# roughly akin to the confidence that the model has in its prediction. 

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

labels[index[0]], percentage[index[0]].item()
_, indices = torch.sort(out, descending=True)

[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
import torch

import torch.nn as nn



class ResNetBlock(nn.Module): # <1>



    def __init__(self, dim):

        super(ResNetBlock, self).__init__()

        self.conv_block = self.build_conv_block(dim)



    def build_conv_block(self, dim):

        conv_block = []



        conv_block += [nn.ReflectionPad2d(1)]



        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),

                       nn.InstanceNorm2d(dim),

                       nn.ReLU(True)]



        conv_block += [nn.ReflectionPad2d(1)]



        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),

                       nn.InstanceNorm2d(dim)]



        return nn.Sequential(*conv_block)



    def forward(self, x):

        out = x + self.conv_block(x) # <2>

        return out





class ResNetGenerator(nn.Module):



    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): # <3> 



        assert(n_blocks >= 0)

        super(ResNetGenerator, self).__init__()



        self.input_nc = input_nc

        self.output_nc = output_nc

        self.ngf = ngf



        model = [nn.ReflectionPad2d(3),

                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),

                 nn.InstanceNorm2d(ngf),

                 nn.ReLU(True)]



        n_downsampling = 2

        for i in range(n_downsampling):

            mult = 2**i

            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,

                                stride=2, padding=1, bias=True),

                      nn.InstanceNorm2d(ngf * mult * 2),

                      nn.ReLU(True)]



        mult = 2**n_downsampling

        for i in range(n_blocks):

            model += [ResNetBlock(ngf * mult)]



        for i in range(n_downsampling):

            mult = 2**(n_downsampling - i)

            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),

                                         kernel_size=3, stride=2,

                                         padding=1, output_padding=1,

                                         bias=True),

                      nn.InstanceNorm2d(int(ngf * mult / 2)),

                      nn.ReLU(True)]



        model += [nn.ReflectionPad2d(3)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        model += [nn.Tanh()]



        self.model = nn.Sequential(*model)



    def forward(self, input): # <3>

        return self.model(input)
netG = ResNetGenerator()
model_path = '../input/horse2zebra/horse2zebra_0.4.0.pth'

model_data = torch.load(model_path)

netG.load_state_dict(model_data)
netG.eval()
from PIL import Image

from torchvision import transforms
preprocess = transforms.Compose([transforms.Resize(256),

transforms.ToTensor()])
img = Image.open("../input/horse2zebra/horse2zebra/horse2zebra/testA/n02381460_1030.jpg")
img
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t,0)
batch_out = netG(batch_t)
out_t = (batch_out.data.squeeze() + 1.0) / 2.0

out_img = transforms.ToPILImage()(out_t)

out_img.save('zebra.jpg')

out_img
out_t = (batch_out.data.squeeze() + 1.0) / 2.0

out_img = transforms.ToPILImage()(out_t)
out_img