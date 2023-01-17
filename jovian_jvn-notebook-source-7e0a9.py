# -*- coding: utf-8 -*-

import argparse



def parse_args_function():

    parser = argparse.ArgumentParser()

    # Required arguments: input and output files.

    parser.add_argument(

        "--input_file",

        default='./datasets/obman/',

        help="Input image, directory"

    )

    parser.add_argument(

        "--output_file",

        default='./checkpoints/model-',

        help="Prefix of output pkl filename"

    )

    # Optional arguments.

    parser.add_argument(

        "--train",

        action='store_true',

        help="Training mode."

    )

    parser.add_argument(

        "--val",

        action='store_true',

        help="Use validation set."

    )

    parser.add_argument(

        "--test",

        action='store_true',

        help="Test model."

    )

    parser.add_argument(

        "--batch_size",

        type=int,

        default = 512,

        help="Mini-batch size"

    )

    parser.add_argument(

        "--model_def",

        default='HopeNet',

        help="Name of the model 'HopeNet', 'GraphUNet' or 'GraphNet'"

    )

    parser.add_argument(

        "--pretrained_model",

        default='',

        help="Load trained model weights file."

    )

    parser.add_argument(

        "--gpu",

        action='store_true',

        help="Switch for gpu computation."

    )

    parser.add_argument(

        "--gpu_number",

        type=int,

        nargs='+',

        default = [0],

        help="Identifies the GPU number to use."

    )

    parser.add_argument(

        "--learning_rate",

        type=float,

        default = 0.01,

        help="Identifies the optimizer learning rate."

    )

    parser.add_argument(

        "--lr_step",

        type=int,

        default = 1000,

        help="Identifies the adaptive learning rate step size."

    )

    parser.add_argument(

        "--lr_step_gamma",

        type=float,

        default = 0.1,

        help="Identifies the adaptive learning rate step gamma."

    )

    parser.add_argument(

        "--log_batch",

        type=int,

        default = 1000,

        help="Show log samples."

    )

    parser.add_argument(

        "--val_epoch",

        type=int,

        default = 1,

        help="Run validation on epochs."

    )

    parser.add_argument(

        "--snapshot_epoch",

        type=int,

        default = 10,

        help="Save snapshot epochs."

    )

    parser.add_argument(

        "--num_iterations",

        type=int,

        default = 10000,

        help="Maximum number of epochs."

    )

    args = parser.parse_args()

    return args
# -*- coding: utf-8 -*-



# import libraries

import numpy as np

import os

import torch.utils.data as data

from PIL import Image





"""# Load Dataset"""



class Dataset(data.Dataset):



    def __init__(self, root='./', load_set='train', transform=None):

        self.root = root#os.path.expanduser(root)

        self.transform = transform

        self.load_set = load_set  # 'train','val','test'



        self.images = np.load(os.path.join(root, 'images-%s.npy'%self.load_set))

        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy'%self.load_set))

        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy'%self.load_set))

        

        #if shuffle:

        #    random.shuffle(data)



    def __getitem__(self, index):

        """

        Args:

            index (int): Index

        Returns:

            tuple: (image, points2D, points3D).

        """

        

        image = Image.open(self.images[index])

        point2d = self.points2d[index]

        point3d = self.points3d[index]



        if self.transform is not None:

            image = self.transform(image)



        return image[:3], point2d, point3d



    def __len__(self):

        return len(self.images)
import argparse



def parse_args_function():

    parser = argparse.ArgumentParser()

    # Required arguments: input and output files.

    parser.add_argument(

        "--input_file",

        default='./datasets/obman/',

        help="Input image, directory"

    )

    parser.add_argument(

        "--output_file",

        default='./checkpoints/model-',

        help="Prefix of output pkl filename"

    )

    # Optional arguments.

    parser.add_argument(

        "--train",

        action='store_true',

        help="Training mode."

    )

    parser.add_argument(

        "--val",

        action='store_true',

        help="Use validation set."

    )

    parser.add_argument(

        "--test",

        action='store_true',

        help="Test model."

    )

    parser.add_argument(

        "--batch_size",

        type=int,

        default = 512,

        help="Mini-batch size"

    )

    parser.add_argument(

        "--model_def",

        default='HopeNet',

        help="Name of the model 'HopeNet', 'GraphUNet' or 'GraphNet'"

    )

    parser.add_argument(

        "--pretrained_model",

        default='',

        help="Load trained model weights file."

    )

    parser.add_argument(

        "--gpu",

        action='store_true',

        help="Switch for gpu computation."

    )

    parser.add_argument(

        "--gpu_number",

        type=int,

        nargs='+',

        default = [0],

        help="Identifies the GPU number to use."

    )

    parser.add_argument(

        "--learning_rate",

        type=float,

        default = 0.01,

        help="Identifies the optimizer learning rate."

    )

    parser.add_argument(

        "--lr_step",

        type=int,

        default = 1000,

        help="Identifies the adaptive learning rate step size."

    )

    parser.add_argument(

        "--lr_step_gamma",

        type=float,

        default = 0.1,

        help="Identifies the adaptive learning rate step gamma."

    )

    parser.add_argument(

        "--log_batch",

        type=int,

        default = 1000,

        help="Show log samples."

    )

    parser.add_argument(

        "--val_epoch",

        type=int,

        default = 1,

        help="Run validation on epochs."

    )

    parser.add_argument(

        "--snapshot_epoch",

        type=int,

        default = 10,

        help="Save snapshot epochs."

    )

    parser.add_argument(

        "--num_iterations",

        type=int,

        default = 10000,

        help="Maximum number of epochs."

    )

    args = parser.parse_args()

    return args
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn.parameter import Parameter

import numpy as np



class GraphConv(nn.Module):

    

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):

        super(GraphConv, self).__init__()

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

        #self.adj_sq = adj_sq

        self.activation = activation

        #self.scale_identity = scale_identity

        #self.I = Parameter(torch.eye(number_of_nodes, requires_grad=False).unsqueeze(0))





    def laplacian(self, A_hat):

        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)

        L = D_hat * A_hat * D_hat

        return L

    

    

    def laplacian_batch(self, A_hat):

        #batch, N = A.shape[:2]

        #if self.adj_sq:

        #    A = torch.bmm(A, A)  # use A^2 to increase graph connectivity

        #I = torch.eye(N).unsqueeze(0).to(device)

        #I = self.I

        #if self.scale_identity:

        #    I = 2 * I  # increase weight of self connections

        #A_hat = A + I

        batch, N = A_hat.shape[:2]

        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)

        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)

        return L





    def forward(self, X, A):

        batch = X.size(0)

        #A = self.laplacian(A)

        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)

        #X = self.fc(torch.bmm(A_hat, X))

        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))

        if self.activation is not None:

            X = self.activation(X)

        return X





class GraphPool(nn.Module):



    def __init__(self, in_nodes, out_nodes):

        super(GraphPool, self).__init__()

        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)





    def forward(self, X):

        X = X.transpose(1, 2)

        X = self.fc(X)

        X = X.transpose(1, 2)

        return X





class GraphUnpool(nn.Module):



    def __init__(self, in_nodes, out_nodes):

        super(GraphUnpool, self).__init__()

        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)





    def forward(self, X):

        X = X.transpose(1, 2)

        X = self.fc(X)

        X = X.transpose(1, 2)

        return X





class GraphUNet(nn.Module):



    def __init__(self, in_features=2, out_features=3):

        super(GraphUNet, self).__init__()



        self.A_0 = Parameter(torch.eye(29).float().cuda(), requires_grad=True)

        self.A_1 = Parameter(torch.eye(15).float().cuda(), requires_grad=True)

        self.A_2 = Parameter(torch.eye(7).float().cuda(), requires_grad=True)

        self.A_3 = Parameter(torch.eye(4).float().cuda(), requires_grad=True)

        self.A_4 = Parameter(torch.eye(2).float().cuda(), requires_grad=True)

        self.A_5 = Parameter(torch.eye(1).float().cuda(), requires_grad=True)



        self.gconv1 = GraphConv(in_features, 4)  # 29 = 21 H + 8 O

        self.pool1 = GraphPool(29, 15)



        self.gconv2 = GraphConv(4, 8)  # 15 = 11 H + 4 O

        self.pool2 = GraphPool(15, 7)



        self.gconv3 = GraphConv(8, 16)  # 7 = 5 H + 2 O

        self.pool3 = GraphPool(7, 4)



        self.gconv4 = GraphConv(16, 32)  # 4 = 3 H + 1 O

        self.pool4 = GraphPool(4, 2)



        self.gconv5 = GraphConv(32, 64)  # 2 = 1 H + 1 O

        self.pool5 = GraphPool(2, 1)



        self.fc1 = nn.Linear(64, 20)



        self.fc2 = nn.Linear(20, 64)



        self.unpool6 = GraphUnpool(1, 2)

        self.gconv6 = GraphConv(128, 32)



        self.unpool7 = GraphUnpool(2, 4)

        self.gconv7 = GraphConv(64, 16)



        self.unpool8 = GraphUnpool(4, 7)

        self.gconv8 = GraphConv(32, 8)



        self.unpool9 = GraphUnpool(7, 15)

        self.gconv9 = GraphConv(16, 4)



        self.unpool10 = GraphUnpool(15, 29)

        self.gconv10 = GraphConv(8, out_features, activation=None)



        self.ReLU = nn.ReLU()



    def _get_decoder_input(self, X_e, X_d):

        return torch.cat((X_e, X_d), 2)



    def forward(self, X):

        X_0 = self.gconv1(X, self.A_0)

        X_1 = self.pool1(X_0)



        X_1 = self.gconv2(X_1, self.A_1)

        X_2 = self.pool2(X_1)



        X_2 = self.gconv3(X_2, self.A_2)

        X_3 = self.pool3(X_2)



        X_3 = self.gconv4(X_3, self.A_3)

        X_4 = self.pool4(X_3)



        X_4 = self.gconv5(X_4, self.A_4)

        X_5 = self.pool5(X_4)



        global_features = self.ReLU(self.fc1(X_5))

        global_features = self.ReLU(self.fc2(global_features))



        X_6 = self.unpool6(global_features)

        X_6 = self.gconv6(self._get_decoder_input(X_4, X_6), self.A_4)



        X_7 = self.unpool7(X_6)

        X_7 = self.gconv7(self._get_decoder_input(X_3, X_7), self.A_3)



        X_8 = self.unpool8(X_7)

        X_8 = self.gconv8(self._get_decoder_input(X_2, X_8), self.A_2)



        X_9 = self.unpool9(X_8)

        X_9 = self.gconv9(self._get_decoder_input(X_1, X_9), self.A_1)



        X_10 = self.unpool10(X_9)

        X_10 = self.gconv10(self._get_decoder_input(X_0, X_10), self.A_0)



        return X_10





class GraphNet(nn.Module):

    

    def __init__(self, in_features=2, out_features=2):

        super(GraphNet, self).__init__()



        self.A_hat = Parameter(torch.eye(29).float().cuda(), requires_grad=True)

        

        self.gconv1 = GraphConv(in_features, 128)

        self.gconv2 = GraphConv(128, 16)

        self.gconv3 = GraphConv(16, out_features, activation=None)

        

    

    def forward(self, X):

        X_0 = self.gconv1(X, self.A_hat)

        X_1 = self.gconv2(X_0, self.A_hat)

        X_2 = self.gconv3(X_1, self.A_hat)

        

        return X_2
"""

This file contains the definitions of the various ResNet models.

Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.

Forward pass was modified to discard the last fully connected layer

"""

import torch

import torch.nn as nn

import torch.utils.model_zoo as model_zoo



model_urls = {

    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',

    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',

    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',

    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

}





def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):

    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,

                     padding=dilation, groups=groups, bias=False, dilation=dilation)





def conv1x1(in_planes, out_planes, stride=1):

    """1x1 convolution"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)





class BasicBlock(nn.Module):

    expansion = 1

    __constants__ = ['downsample']



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

    __constants__ = ['downsample']



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

        x = torch.flatten(x, 1)

        out = x

        x = self.fc(x)



        return x.view(-1, 29, 2), out



def resnet10(pretrained=False, num_classes=1000, **kwargs):

    """Constructs a ResNet-10 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """ 

    model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=1000, **kwargs)

    # if pretrained:

    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet10']))

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, num_classes)

    return model



def resnet18(pretrained=False, num_classes=1000, **kwargs):

    """Constructs a ResNet-18 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """ 

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000, **kwargs)

    if pretrained:

        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, num_classes)

    return model



def resnet50(pretrained=False, num_classes=1000, **kwargs):

    """Constructs a ResNet-50 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """ 

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000, **kwargs)

    if pretrained:

        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, num_classes)

    return model



def resnet101(pretrained=False, num_classes=1000, **kwargs):

    """Constructs a ResNet-101 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """ 

    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=1000, **kwargs)

    if pretrained:

        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, num_classes)

    return model





def resnet152(pretrained=False, num_classes=1000, **kwargs):

    """Constructs a ResNet-152 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """ 

    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=1000, **kwargs)

    if pretrained:

        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn.parameter import Parameter

import numpy as np



class GraphConv(nn.Module):

    

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):

        super(GraphConv, self).__init__()

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

        #self.adj_sq = adj_sq

        self.activation = activation

        #self.scale_identity = scale_identity

        #self.I = Parameter(torch.eye(number_of_nodes, requires_grad=False).unsqueeze(0))





    def laplacian(self, A_hat):

        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)

        L = D_hat * A_hat * D_hat

        return L

    

    

    def laplacian_batch(self, A_hat):

        #batch, N = A.shape[:2]

        #if self.adj_sq:

        #    A = torch.bmm(A, A)  # use A^2 to increase graph connectivity

        #I = torch.eye(N).unsqueeze(0).to(device)

        #I = self.I

        #if self.scale_identity:

        #    I = 2 * I  # increase weight of self connections

        #A_hat = A + I

        batch, N = A_hat.shape[:2]

        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)

        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)

        return L





    def forward(self, X, A):

        batch = X.size(0)

        #A = self.laplacian(A)

        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)

        #X = self.fc(torch.bmm(A_hat, X))

        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))

        if self.activation is not None:

            X = self.activation(X)

        return X





class GraphPool(nn.Module):



    def __init__(self, in_nodes, out_nodes):

        super(GraphPool, self).__init__()

        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)





    def forward(self, X):

        X = X.transpose(1, 2)

        X = self.fc(X)

        X = X.transpose(1, 2)

        return X





class GraphUnpool(nn.Module):



    def __init__(self, in_nodes, out_nodes):

        super(GraphUnpool, self).__init__()

        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)





    def forward(self, X):

        X = X.transpose(1, 2)

        X = self.fc(X)

        X = X.transpose(1, 2)

        return X





class GraphUNet(nn.Module):



    def __init__(self, in_features=2, out_features=3):

        super(GraphUNet, self).__init__()



        self.A_0 = Parameter(torch.eye(29).float().cuda(), requires_grad=True)

        self.A_1 = Parameter(torch.eye(15).float().cuda(), requires_grad=True)

        self.A_2 = Parameter(torch.eye(7).float().cuda(), requires_grad=True)

        self.A_3 = Parameter(torch.eye(4).float().cuda(), requires_grad=True)

        self.A_4 = Parameter(torch.eye(2).float().cuda(), requires_grad=True)

        self.A_5 = Parameter(torch.eye(1).float().cuda(), requires_grad=True)



        self.gconv1 = GraphConv(in_features, 4)  # 29 = 21 H + 8 O

        self.pool1 = GraphPool(29, 15)



        self.gconv2 = GraphConv(4, 8)  # 15 = 11 H + 4 O

        self.pool2 = GraphPool(15, 7)



        self.gconv3 = GraphConv(8, 16)  # 7 = 5 H + 2 O

        self.pool3 = GraphPool(7, 4)



        self.gconv4 = GraphConv(16, 32)  # 4 = 3 H + 1 O

        self.pool4 = GraphPool(4, 2)



        self.gconv5 = GraphConv(32, 64)  # 2 = 1 H + 1 O

        self.pool5 = GraphPool(2, 1)



        self.fc1 = nn.Linear(64, 20)



        self.fc2 = nn.Linear(20, 64)



        self.unpool6 = GraphUnpool(1, 2)

        self.gconv6 = GraphConv(128, 32)



        self.unpool7 = GraphUnpool(2, 4)

        self.gconv7 = GraphConv(64, 16)



        self.unpool8 = GraphUnpool(4, 7)

        self.gconv8 = GraphConv(32, 8)



        self.unpool9 = GraphUnpool(7, 15)

        self.gconv9 = GraphConv(16, 4)



        self.unpool10 = GraphUnpool(15, 29)

        self.gconv10 = GraphConv(8, out_features, activation=None)



        self.ReLU = nn.ReLU()



    def _get_decoder_input(self, X_e, X_d):

        return torch.cat((X_e, X_d), 2)



    def forward(self, X):

        X_0 = self.gconv1(X, self.A_0)

        X_1 = self.pool1(X_0)



        X_1 = self.gconv2(X_1, self.A_1)

        X_2 = self.pool2(X_1)



        X_2 = self.gconv3(X_2, self.A_2)

        X_3 = self.pool3(X_2)



        X_3 = self.gconv4(X_3, self.A_3)

        X_4 = self.pool4(X_3)



        X_4 = self.gconv5(X_4, self.A_4)

        X_5 = self.pool5(X_4)



        global_features = self.ReLU(self.fc1(X_5))

        global_features = self.ReLU(self.fc2(global_features))



        X_6 = self.unpool6(global_features)

        X_6 = self.gconv6(self._get_decoder_input(X_4, X_6), self.A_4)



        X_7 = self.unpool7(X_6)

        X_7 = self.gconv7(self._get_decoder_input(X_3, X_7), self.A_3)



        X_8 = self.unpool8(X_7)

        X_8 = self.gconv8(self._get_decoder_input(X_2, X_8), self.A_2)



        X_9 = self.unpool9(X_8)

        X_9 = self.gconv9(self._get_decoder_input(X_1, X_9), self.A_1)



        X_10 = self.unpool10(X_9)

        X_10 = self.gconv10(self._get_decoder_input(X_0, X_10), self.A_0)



        return X_10





class GraphNet(nn.Module):

    

    def __init__(self, in_features=2, out_features=2):

        super(GraphNet, self).__init__()



        self.A_hat = Parameter(torch.eye(29).float().cuda(), requires_grad=True)

        

        self.gconv1 = GraphConv(in_features, 128)

        self.gconv2 = GraphConv(128, 16)

        self.gconv3 = GraphConv(16, out_features, activation=None)

        

    

    def forward(self, X):

        X_0 = self.gconv1(X, self.A_hat)

        X_1 = self.gconv2(X_0, self.A_hat)

        X_2 = self.gconv3(X_1, self.A_hat)

        

        return X_2
# -*- coding: utf-8 -*-



def select_model(model_def):

    if model_def.lower() == 'hopenet':

        model = HopeNet()

        print('HopeNet is loaded')

    elif model_def.lower() == 'resnet10':

        model = resnet10(pretrained=False, num_classes=29*2)

        print('ResNet10 is loaded')

    elif model_def.lower() == 'resnet18':

        model = resnet18(pretrained=False, num_classes=29*2)

        print('ResNet18 is loaded')

    elif model_def.lower() == 'resnet50':

        model = resnet50(pretrained=False, num_classes=29*2)

        print('ResNet50 is loaded')

    elif model_def.lower() == 'resnet101':

        model = resnet101(pretrained=False, num_classes=29*2)

        print('ResNet101 is loaded')

    elif model_def.lower() == 'graphunet':

        model = GraphUNet(in_features=2, out_features=3)

        print('GraphUNet is loaded')

    elif model_def.lower() == 'graphnet':

        model = GraphNet(in_features=2, out_features=3)

        print('GraphNet is loaded')

    else:

        raise NameError('Undefined model')

    return model
!pip install jovian --upgrade -q

import jovian

jovian.commit()