# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install fastai2
# importing libraries
import torch
import torchvision
import torchvision.transforms as tf
import torch.nn as nn
import torch.optim as optim

# augmentations or different operations necessary operation for training
trans = tf.Compose([tf.Resize((224, 224)),
                    tf.ToTensor(),
                    tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ])
# creating dataloaders for training
train_dataset = torchvision.datasets.ImageFolder("/kaggle/input/pulse-data/pulse/train", transform=trans)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=64,
                                            num_workers=4,
                                            shuffle=True)
# creating loaders for validation 
valid_dataset = torchvision.datasets.ImageFolder("/kaggle/input/pulse-data/pulse/val", transform=trans)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=64,
                                            num_workers=4,
                                            shuffle=True)
# loading pretrained model
model = torchvision.models.mobilenet_v2(pretrained=True)
# changing head depending on no. of classes
model.classifier[1] = nn.Linear(in_features=1280, out_features=70, bias=True)
# calculating no. of parameters
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# shifting model to cuda
model = model.cuda()
# function for calculating validation accuracy
def valid_acc(model):
    acc=0
    for i, (b_x, b_y) in enumerate(valid_loader):
        model.eval()
        test_output = model(b_x.cuda())
        outs = test_output.cpu().detach().numpy().argmax(axis=1)
        acc += (outs == b_y.cpu().numpy()).sum()
    return acc*100.0 / (3333)
        
    
valid_dataset
# initialing loss function and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.95)
# training model
print("Start of the Optimization Processs..")
best_accuracy = 0
for epoch in range(10):  
    running_loss = 0.0
    for step, (b_x, b_y) in enumerate(train_loader):
        model.train()
        # erasing grad to store new one otherwise, it will be added
        optimizer.zero_grad()

        
        outputs = model(b_x.cuda())
        loss = criterion(outputs, b_y.cuda())
        
        # starting backpropagation
        loss.backward() 
        # applying new grads to weights
        optimizer.step()

        
        running_loss += loss.item()
    accuracy = valid_acc(model)
    if accuracy>best_accuracy:
        best_accuracy = accuracy
        torch.save(model, "/kaggle/working/pure_model.pth")
        
    print('Epoch: ', epoch, '| Step: ', step, '| Accuracy: ', accuracy, "| Train Loss:", running_loss, "| Best Accuracy: ", best_accuracy)

print("Training Complete")
nb_classes = 70

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(valid_loader):
        inputs = inputs.cuda()
        classes = classes.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)
print(" Class Wise Accuracy")
print(confusion_matrix.diag()/confusion_matrix.sum(1))
from fastai2.vision.all import *
from fastai2.test_utils import *
# creating datablock or we can say dataloader
data = ImageDataLoaders.from_folder(Path("/kaggle/input/pulse-data/pulse/"), valid="val", bs = 64, item_tfms=Resize(224), batch_tfms=Normalize.from_stats(*imagenet_stats))
# no. of classe and no. of images in train and valid dataset
data.c, len(data.train_ds), len(data.valid_ds)
model = torchvision.models.mobilenet_v2(pretrained=True)
# model.classifier[1] = nn.Linear(in_features=1280, out_features=70, bias=True)
# creating learner
learn = Learner(data, model, metrics=accuracy, opt_func=ranger, model_dir="/kaggle/working/")
# creating learner
learn.fine_tune(8, freeze_epochs=4, base_lr=0.003, cbs = SaveModelCallback(monitor="accuracy", fname="best"))

learn.fit_one_cycle(5, lr_max=0.001, cbs = SaveModelCallback(monitor="accuracy", fname="best1"))
interp = Interpretation.from_learner(learn)
torch.save(learn.model, "/kaggle/working/pure_pyfast.pth")
# classes with top losses
interp.plot_top_losses(15, figsize=(20,15))

nb_classes = 70
model = learn.model
confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(valid_loader):
        inputs = inputs.cuda()
        classes = classes.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)
print(" Class Wise Accuracy: ")
print(confusion_matrix.diag()/confusion_matrix.sum(1))
# import math
# import torch.nn.functional as F
# class Bottleneck(nn.Module):
#     def __init__(self, nChannels, growthRate):
#         super(Bottleneck, self).__init__()
#         interChannels = 4*growthRate
#         self.bn1 = nn.BatchNorm2d(nChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(interChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,padding=1, bias=False)

#     def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
#         out = self.conv2(F.relu(self.bn2(out)))
#         out = torch.cat((x, out), 1)
#         return out

    
# class SingleLayer(nn.Module):
#     def __init__(self, nChannels, growthRate):
#         super(SingleLayer, self).__init__()
#         self.bn1 = nn.BatchNorm2d(nChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,padding=1, bias=False)

#     def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
#         out = torch.cat((x, out), 1)
#         return out

    
# class Transition(nn.Module):
#     def __init__(self, nChannels, nOutChannels):
#         super(Transition, self).__init__()
#         self.bn1 = nn.BatchNorm2d(nChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,bias=False)

#     def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
#         out = nn.AdaptiveAvgPool2d(out, 2)
#         return out


    
# class DenseNet(nn.Module):
#     def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
#         super(DenseNet, self).__init__()

#         nDenseBlocks = (depth-4) // 3
#         if bottleneck:
#             nDenseBlocks //= 2

#         nChannels = 2*growthRate
#         self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
#         self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
#         nChannels += nDenseBlocks*growthRate
#         nOutChannels = int(math.floor(nChannels*reduction))
#         self.trans1 = Transition(nChannels, nOutChannels)

#         nChannels = nOutChannels
#         self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
#         nChannels += nDenseBlocks*growthRate
#         nOutChannels = int(math.floor(nChannels*reduction))
#         self.trans2 = Transition(nChannels, nOutChannels)

#         nChannels = nOutChannels
#         self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
#         nChannels += nDenseBlocks*growthRate

#         self.bn1 = nn.BatchNorm2d(nChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.fc = nn.Linear(nChannels, nClasses)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()

#     def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
#         layers = []
#         for i in range(int(nDenseBlocks)):
#             if bottleneck:
#                 layers.append(Bottleneck(nChannels, growthRate))
#             else:
#                 layers.append(SingleLayer(nChannels, growthRate))
#             nChannels += growthRate
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.trans1(self.dense1(out))
#         out = self.trans2(self.dense2(out))
#         out = self.dense3(out)
#         out = torch.squeeze(nn.AdaptiveAvgPool2d(F.relu(self.bn1(out)), 8))
#         out = nn.LogSoftmax(self.fc(out))
#         return out
# densenet = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=70).cuda()
# densenet
# loading the smallest densenet
model = torchvision.models.densenet121(pretrained=True)
# deleting different layers from 4th denseblock
del model.features.denseblock4.denselayer16
del model.features.denseblock4.denselayer15
del model.features.denseblock4.denselayer14
del model.features.denseblock4.denselayer13
del model.features.denseblock4.denselayer12
del model.features.denseblock4.denselayer11
del model.features.denseblock4.denselayer10
del model.features.denseblock4.denselayer9
del model.features.denseblock4.denselayer8

del model.features.denseblock3.denselayer24
del model.features.denseblock3.denselayer23
del model.features.denseblock3.denselayer22
del model.features.denseblock3.denselayer21
del model.features.denseblock3.denselayer20
del model.features.denseblock3.denselayer19
del model.features.denseblock3.denselayer18
del model.features.denseblock3.denselayer17
del model.features.denseblock3.denselayer16
del model.features.denseblock3.denselayer15
del model.features.denseblock3.denselayer14
del model.features.denseblock3.denselayer13
del model.features.denseblock3.denselayer12
del model.features.denseblock3.denselayer11
del model.features.denseblock3.denselayer10
del model.features.denseblock3.denselayer9