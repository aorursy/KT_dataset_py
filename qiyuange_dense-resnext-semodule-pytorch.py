import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
class SEmodule(nn.Module):
    
    def __init__(self,channels,reduction):
        super(SEmodule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        feature = x
        x = self.avgpool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x * feature
class SENextBottleneck(nn.Module):

    def __init__(self, inplanes, growth_rate, groups=32, reduction=2, stride=1, base_width=6):
        super(SENextBottleneck, self).__init__()
        width = math.floor(growth_rate * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, growth_rate, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEmodule(growth_rate, reduction=reduction)
        self.stride = stride
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se_module(out)
        out = self.relu(out)

        return out
class Transition(nn.Module):
    
    def __init__(self, num_input_feature, num_output_feature):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_feature)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_feature, num_output_feature, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self,x):
        features = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avgpool(x)
        
        return x
class DenseBlock(nn.Module):
    
    def __init__(self, num_features, growth_rate=32, width=6):
        super(DenseBlock, self).__init__()
        self.block = SENextBottleneck(inplanes=num_features, growth_rate=growth_rate, base_width=width)

    def forward(self, init_features):
        features = [init_features]
        new_features = self.block(init_features)
        features.append(new_features)
        
        return torch.cat(features, 1)
class DenSENext(nn.Module):
    def __init__(self, num_classes=10, growth_rate=32, layers=[6, 12, 24, 16], num_init_feature=64, width=6, drop_rate=None):
        super(DenSENext, self).__init__()
        self.num_features = num_init_feature
        self.drop_rate = drop_rate
        self.layer0 = nn.Sequential(
            OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature, kernel_size=7, stride=2,
                                padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_feature)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                        ]))
        self.layer1 = self._makelayer(num_blocks=layers[0], growth_rate=growth_rate, width=width)
        self.layer2 = self._makelayer(num_blocks=layers[1], growth_rate=growth_rate, width=width)
        self.layer3 = self._makelayer(num_blocks=layers[2], growth_rate=growth_rate, width=width)
        self.layer4 = self._makelayer(num_blocks=layers[3], growth_rate=growth_rate, width=width, last_layer=True)
        self.bn = nn.BatchNorm2d(self.num_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_rate) if drop_rate is not None else None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.num_features, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def _makelayer(self, num_blocks, growth_rate=32, reduction=2, width=6, last_layer=False):
        num_features = self.num_features
        layer = []
        for i in range(num_blocks):
            block = DenseBlock(num_features=num_features, growth_rate=growth_rate, width=width)
            layer.append(block)
            num_features += growth_rate
            if last_layer == False and (i+1) == num_blocks:
                trans = Transition(num_features,num_features//2)
                layer.append(trans)
                num_features = num_features // 2     
        self.num_features = num_features
        
        return nn.Sequential(*layer)
        
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.drop_rate is not None:
            x = self.dropout(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        
        return out
def DenSENext121(num_classes=1000):
    model = DenSENext(growth_rate=32, layers=[6, 12, 24, 16], num_classes=num_classes, num_init_feature=64, width=8)
    
    return model


def DenSENext161(num_classes=1000):
    model = DenSENext(growth_rate=48, layers=[6, 12, 36, 24], num_classes=num_classes, num_init_feature=96, width=6)
    
    return model


def DenSENext169(num_classes=1000):
    model = DenSENext(growth_rate=32, layers=[6, 12, 32, 32], num_classes=num_classes, num_init_feature=64, width=8)
    
    return model


def DenSENext201(num_classes=100):
    model = DenSENext(growth_rate=32, layers=[6, 12, 48, 32], num_classes=num_classes, num_init_feature=64, width=8)
    
    return model

def measure_size_of_model(model):
    num = 0
    for weights in iter(model.parameters()):
        size = weights.view(-1).size(0)
        num += size
    if num//1e6 >=1:
        main = num/1e6
        print('Statistic Result : {:.2f}M parameters'.format(main))
    elif num//1e3 >=1:
        main = num/1e3
        print('Statistic Result : {:.2f}K parameters'.format(main))
    else:
        print('Statistic Result : {:.2f} parameters'.format(num))
import torchvision
measure_size_of_model(DenSENext121(num_classes=1000))
measure_size_of_model(DenSENext169(num_classes=1000))
measure_size_of_model(DenSENext201(num_classes=1000))
measure_size_of_model(DenSENext161(num_classes=1000))
measure_size_of_model(torchvision.models.resnext50_32x4d())
measure_size_of_model(torchvision.models.resnext101_32x8d())
from IPython.display import Image
Image('../input/birds-cam/birds_cam/camap.png')
Image('../input/birds-cam/birds_cam/camap134_1.png')
Image('../input/birds-cam/birds_cam/camap138_2.png')
Image('../input/birds-cam/birds_cam/camap189_1.png')
Image('../input/birds-cam/birds_cam/camap189_2.png')
Image('../input/birds-cam/birds_cam/camap189_3.png')