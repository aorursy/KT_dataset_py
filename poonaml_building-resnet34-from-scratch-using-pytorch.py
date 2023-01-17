import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision import models

import pdb
models.resnet34()
inp = torch.randn([2,3,224,224])
inp.shape # bs,rgb channels , width,height
conv_block = nn.Sequential(nn.Conv2d(3,64,kernel_size=7, stride=2, padding=3, bias=False), #112,112

                       nn.BatchNorm2d(64),

                       nn.ReLU(inplace=True),

                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # 56,56

conv_block
#inp=nn.Conv2d(3,64,kernel_size=7, stride=2, padding=3, bias=False)(inp)

#inp.shape
out=conv_block(inp)

out.shape
list(models.resnet34().children())[:4]
class BasicBlock(nn.Module):

    expansion = 1



    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,

                     padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,

                     padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        identity = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)



        if self.downsample is not None:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)



        return out
BasicBlock(64,128)
t = torch.randn((2,64,56,56))

t.shape
BasicBlock(64,64)(t).shape
def _make_layer(block, inplanes,planes, blocks, stride=1):

    downsample = None  

    if stride != 1 or inplanes != planes:

        downsample = nn.Sequential(            

            nn.Conv2d(inplanes, planes, 1, stride, bias=False),

            nn.BatchNorm2d(planes),

        )

    layers = []

    layers.append(block(inplanes, planes, stride, downsample))

    inplanes = planes

    for _ in range(1, blocks):

        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)
layers=[3, 4, 6, 3]
layer1 =_make_layer(BasicBlock, inplanes=64,planes=64, blocks=layers[0])

layer1
list(models.resnet34().children())[4]
layer2 = _make_layer(BasicBlock, 64, 128, layers[1], stride=2)

layer2
list(models.resnet34().children())[5]
t = torch.rand((2,64,56,56))

t.shape #batch, RGB channels/filters,width,height
o = nn.Conv2d(64,128,3,2,1)(t)

o.shape
o+t  #height and width not matching  56,56 !=28,26
t_d =nn.Conv2d(64,128,1,2,0)(t)

o.shape,t_d.shape
(o+t_d).shape
num_classes=1000

nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),

              nn.Linear(512 , num_classes))
list(models.resnet34().children())[8:]
class ResNet(nn.Module):



    def __init__(self, block, layers, num_classes=1000):

        super().__init__()

        

        self.inplanes = 64



        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,

                               bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 , num_classes)





    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None  

   

        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(

                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),

                nn.BatchNorm2d(planes),

            )



        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))

        

        self.inplanes = planes

        

        for _ in range(1, blocks):

            layers.append(block(self.inplanes, planes))



        return nn.Sequential(*layers)

    

    

    def forward(self, x):

        x = self.conv1(x)           # 224x224

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)         # 112x112



        x = self.layer1(x)          # 56x56

        x = self.layer2(x)          # 28x28

        x = self.layer3(x)          # 14x14

        x = self.layer4(x)          # 7x7



        x = self.avgpool(x)         # 1x1

        x = torch.flatten(x, 1)     # remove 1 X 1 grid and make vector of tensor shape 

        x = self.fc(x)



        return x
def resnet34():

    layers=[3, 4, 6, 3]

    model = ResNet(BasicBlock, layers)

    return model
model=resnet34()
model