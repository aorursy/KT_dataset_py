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
#https://www.kaggle.com/qi4589746/ccu-cs-stanford-car?fbclid=IwAR1OABNzofB93xoIIHhNXlq2CPFf4eUelida9DM7aZzQAfBGZ15wKW3WV5s

#裁切圖片



import numpy as np

import scipy.io as sio 

import os

import cv2

from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D

from keras import backend as K

import matplotlib.pyplot as plt

import pprint

import skimage.io



def get_labels(mat):

    annos = sio.loadmat(mat)

    _, total_size = annos["annotations"].shape

    print("total sample size is ", total_size)

    labels = np.zeros((total_size, 5)) 

    

    for i in range(total_size):

        path = annos["annotations"][:,i][0][5][0].split(".")

        id = int(path[0]) - 1

        for j in range(5):

            labels[id, j] = int(annos["annotations"][:,i][0][j][0])

    return labels



def crop():

    labels = get_labels('../input/devkit/devkit/cars_train_annos.mat')

    image_names = os.listdir('../input/cars_train/cars_train')

    i=0

    for i in range(8144):

        iname = image_names[i]

        im = cv2.imread("../input/cars_train/cars_train/" + iname)[:,:,::-1]

        image_label = labels[int(iname[:5]) - 1]

        cut = im[int(image_label[1]):int(image_label[3]),int(image_label[0]):int(image_label[2])]

        cv2.imwrite('../seg_train/'+str(int(image_label[4]))+'/'+iname, cv2.cvtColor(cut, cv2.COLOR_RGB2BGR))

        if(i%50 == 0):

            print(i,end=',')

    print(i-1)

    print('Images for training crop is finished.\n')

    

    labels = get_labels('../input/devkit/devkit/cars_test_annos_withlabels.mat')

    image_names = os.listdir('../input/cars_test/cars_test')

    i=0

    for i in range(8041):

        iname = image_names[i]

        im = cv2.imread("../input/cars_test/cars_test/" + iname)[:,:,::-1]

        image_label = labels[int(iname[:5]) - 1]

        cut = im[int(image_label[1]):int(image_label[3]),int(image_label[0]):int(image_label[2])]

        cv2.imwrite('../seg_test/'+str(int(image_label[4]))+'/'+iname, cv2.cvtColor(cut, cv2.COLOR_RGB2BGR))

        if(i%50 == 0):

            print(i,end=',')

    print(i-1)

    print('Images for testing crop is finished.\n')



def dirwork():

    os.mkdir('../seg_train')

    os.mkdir('../seg_test')

    for i in range(1,197):

        os.mkdir('../seg_train/'+str(i))

        os.mkdir('../seg_test/'+str(i))

dirwork()

crop()



#載入pretrained resnet101，並進行架構修改

import torch.nn as nn



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):

    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,

                     padding=dilation, groups=groups, bias=False, dilation=dilation)





def conv1x1(in_planes, out_planes, stride=1):

    """1x1 convolution"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)





class BasicBlock(nn.Module):

    expansion = 1



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,

                 base_width=64, dilation=1, norm_layer=None):

        super(BasicBlock, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:

            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        if dilation > 1:

            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = norm_layer(planes)

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





class Bottleneck(nn.Module):

    expansion = 4



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,

                 base_width=64, dilation=1, norm_layer=None):

        super(Bottleneck, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv1x1(inplanes, width)

        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)

        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)

        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        identity = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)



        out = self.conv3(out)

        out = self.bn3(out)



        if self.downsample is not None:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)



        return out





class ResNet(nn.Module):



    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,

                 groups=1, width_per_group=64, replace_stride_with_dilation=None,

                 norm_layer=None):

        super(ResNet, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer



        self.inplanes = 64

        self.dilation = 1

        if replace_stride_with_dilation is None:

            # each element in the tuple indicates if we should replace

            # the 2x2 stride with a dilated convolution instead

            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:

            raise ValueError("replace_stride_with_dilation should be None "

                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups

        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,

                               bias=False)

        self.bn1 = norm_layer(self.inplanes)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,

                                       dilate=replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,

                                       dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,

                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)



        # Zero-initialize the last BN in each residual branch,

        # so that the residual branch starts with zeros, and each residual block behaves like an identity.

        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        if zero_init_residual:

            for m in self.modules():

                if isinstance(m, Bottleneck):

                    nn.init.constant_(m.bn3.weight, 0)

                elif isinstance(m, BasicBlock):

                    nn.init.constant_(m.bn2.weight, 0)



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

        norm_layer = self._norm_layer

        downsample = None

        previous_dilation = self.dilation

        if dilate:

            self.dilation *= stride

            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                conv1x1(self.inplanes, planes * block.expansion, stride),

                norm_layer(planes * block.expansion),

            )



        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,

                            self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):

            layers.append(block(self.inplanes, planes, groups=self.groups,

                                base_width=self.base_width, dilation=self.dilation,

                                norm_layer=norm_layer))



        return nn.Sequential(*layers)



    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)



        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)



        x = self.avgpool(x)

        x = x.reshape(x.size(0), -1)

        x = self.fc(x)



        return x





def _resnet(arch, inplanes, planes, pretrained, progress, **kwargs):

    model = ResNet(inplanes, planes, **kwargs)

    if pretrained:

        state_dict = load_state_dict_from_url(model_urls[arch],

                                              progress=progress)

        model.load_state_dict(state_dict)

    return model





def resnet18(pretrained=False, progress=True, **kwargs):

    """Constructs a ResNet-18 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,

                   **kwargs)





def resnet34(pretrained=False, progress=True, **kwargs):

    """Constructs a ResNet-34 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,

                   **kwargs)





def resnet50(pretrained=False, progress=True, **kwargs):

    """Constructs a ResNet-50 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,

                   **kwargs)





def resnet101(pretrained=False, progress=True, **kwargs):

    """Constructs a ResNet-101 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,

                   **kwargs)





def resnet152(pretrained=False, progress=True, **kwargs):

    """Constructs a ResNet-152 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,

                   **kwargs)





def resnext50_32x4d(**kwargs):

    kwargs['groups'] = 32

    kwargs['width_per_group'] = 4

    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],

                   pretrained=False, progress=True, **kwargs)





def resnext101_32x8d(**kwargs):

    kwargs['groups'] = 32

    kwargs['width_per_group'] = 8

    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],

                   pretrained=False, progress=True, **kwargs)







import torchvision.models as models

import torch

import torch.nn as nn

import math

import torch.utils.model_zoo as model_zoo

 

class CNN(nn.Module):

 

    def __init__(self, block, layers, num_classes=196):

        self.inplanes = 64

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,

                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        #新增反卷基層

        self.convtranspose1 = nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1, bias=False, dilation=1)

        #新增池化層

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        #去除原來fc，改為符合我的分類數量

        self.fclass = nn.Linear(2048, num_classes)

 

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()

 

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                nn.Conv2d(self.inplanes, planes * block.expansion,

                          kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(planes * block.expansion),

            )

 

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):

            layers.append(block(self.inplanes, planes))

 

        return nn.Sequential(*layers)

 

    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

 

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

 

        x = self.avgpool(x)

        #x = x.view(x.size(0), -1)

        x = self.convtranspose1(x)

        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)

        x = self.fclass(x)

 

        return x

 

#載入model

import torchvision.models as models

resnet101 = models.resnet101(pretrained=True)



cnn = CNN(Bottleneck, [3, 4, 23, 3])

pretrained_dict = resnet101.state_dict()

model_dict = cnn.state_dict()

pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict)

cnn.load_state_dict(model_dict)

#印出模型架構

print(cnn)





#訓練

from torch.utils.data import Dataset

from torch.autograd import Variable

from torch.utils.data import DataLoader

from torchvision import transforms

from pathlib import Path

from PIL import Image

import torch.nn as nn

import torch.nn.functional as F

import math

import argparse

import torch

import time

import copy



import warnings

warnings.filterwarnings("ignore")



class IMAGE_Dataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)

        self.x = []

        self.y = []

        self.transform = transform

        self.num_classes = 0

        #print(self.root_dir.name)

        for i, _dir in enumerate(self.root_dir.glob('*')):

            for file in _dir.glob('*'):

                self.x.append(file)

                self.y.append(i)



            self.num_classes += 1

            #print(self.num_classes)

        #print(self.num_classes)

    def __len__(self):

        return len(self.x)



    def __getitem__(self, index):

        image = Image.open(self.x[index]).convert('RGB')

        if self.transform:

            image = self.transform(image)



        return image, self.y[index]

    

torch.manual_seed(123)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

CUDA_DEVICES = 0



def test(mname):

    TEST_ROOT = '../seg_test/'

    PATH_TO_WEIGHTS = mname



    data_transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[

                             0.229, 0.224, 0.225])

    ])



    test_set = IMAGE_Dataset(Path(TEST_ROOT), data_transform)

    data_loader = DataLoader(

        dataset=test_set, batch_size=32, shuffle=True, num_workers=1)

    classes = [_dir.name for _dir in Path(TEST_ROOT).glob('*')]



    model = torch.load(PATH_TO_WEIGHTS)

    model = model.cuda(CUDA_DEVICES)

    model.eval()



    total_correct = 0

    total = 0

    class_correct = list(0. for i in enumerate(classes))

    class_total = list(0. for i in enumerate(classes))

    with torch.no_grad():

        for inputs, labels in data_loader:

            inputs = Variable(inputs.cuda(CUDA_DEVICES))

            labels = Variable(labels.cuda(CUDA_DEVICES))

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            # totoal

            total += labels.size(0)

            total_correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()

            # batch size

            for i in range(labels.size(0)):

                label =labels[i]

                class_correct[label] += c[i].item()

                class_total[label] += 1



    print('Accuracy on the ALL test images: %d %%' % (100 * total_correct / total))



    for i, c in enumerate(classes):

        print('Accuracy of %5s : %2d %%' % (c, 100 * class_correct[i] / class_total[i]))

    print()

    

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def train(bsize, ep, LearnRate):

    DATASET_ROOT = '../seg_train/'

    print(time.ctime())



	#data augmentation

    data_transform = transforms.Compose([

        transforms.Resize(256),

        transforms.RandomCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[

                             0.229, 0.224, 0.225])

    ])



    #print(DATASET_ROOT)

    train_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)

    data_loader = DataLoader(dataset=train_set, batch_size=bsize, shuffle=True, num_workers=1)

    #print(train_set.num_classes)

   

    model = cnn

    model = model.cuda(CUDA_DEVICES)

    model.train()



    best_model_params = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    num_epochs = ep

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=LearnRate, momentum=0.9)



    for epoch in range(num_epochs):

        print(f'Epoch: {epoch + 1}/{num_epochs}')

        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        start_time = time.time()



        training_loss = 0.0

        training_corrects = 0



        for i, (inputs, labels) in enumerate(data_loader):

            inputs = Variable(inputs.cuda(CUDA_DEVICES))

            labels = Variable(labels.cuda(CUDA_DEVICES))



            optimizer.zero_grad()



            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)



            loss.backward()

            optimizer.step()



            training_loss += loss.item() * inputs.size(0)

            #revise loss.data[0]-->loss.item()

            training_corrects += torch.sum(preds == labels.data)

            #print(f'training_corrects: {training_corrects}')



        training_loss = training_loss / len(train_set)

        training_acc =training_corrects.double() /len(train_set)

        #print(training_acc.type())

        #print(f'training_corrects: {training_corrects}\tlen(train_set):{len(train_set)}\n')

        print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}', end = '   ....')

        end_time = time.time()

        seconds = end_time - start_time

        m, s = divmod(seconds, 60)

        h, m = divmod(m, 60)

        print ("Time cost: %02d:%02d:%02d\n" % (h, m, s))



        if training_acc > best_acc:

            best_acc = training_acc

            best_model_params = copy.deepcopy(model.state_dict())



        if (epoch == 0 or (epoch+1) % 10 == 0):

            model.load_state_dict(best_model_params)

            torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')

            print(time.ctime())

            test('model-%.2f-best_train_acc.pth' % best_acc)

train(32, 60, 0.001)