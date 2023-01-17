from __future__ import print_function, division  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.autograd import Variable  
import numpy as np  
import torchvision  
from torchvision import datasets, models, transforms  
import matplotlib.pyplot as plt  
import time  
import copy  
import os
from sklearn.model_selection import train_test_split
import pandas as pd
plt.ion()   # interactive mode  
import PIL.Image as Image
class dataset(torch.utils.data.Dataset):
    def __init__(self, imgroot, anno_pd, transforms=None):
        self.root_path = imgroot
        self.paths = anno_pd['ImageName'].tolist()
        self.labels = anno_pd['label'].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.labels[item]-1
        return img, label

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
def collate_fn(batch):
    imgs = []
    label = []

    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
    return torch.stack(imgs, 0), \
           label
traindata_root = '../input/baidudata/try/try/train'
train_pd = pd.read_csv("../input/baidudata/try/try/train.txt",sep=" ",
                       header=None, names=['ImageName', 'label'])
#train_pd, val_pd = train_test_split(all_pd, test_size=0.15, random_state=43,
#                                    stratify=all_pd['label'])
testdata_root = '../input/baidudata/try/try/test'
val_pd = pd.read_csv("../input/truetest/true_test_label.csv",
                       header=None, names=['ImageName', 'label'])
val_pd = val_pd.iloc[1:]
val_pd['label']=val_pd['label'].astype(int)
'''数据扩增与归一化(对val仅仅归一化)'''
data_transforms = {
    'train': transforms.Compose([  
        transforms.RandomRotation(degrees=20), #随机旋转
        transforms.Resize([224,224]),
        #transforms.Resize(224),                    
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        #transforms.RandomResizedCrop(224,scale=(0.49,1.0)),       #随机切割后resize
        #transforms.RandomHorizontalFlip(),                    #随机水平翻转,默认概率0.5
        transforms.ToTensor(),   # 0-255 to 0-1                
#取值范围为[0, 255]的PIL.Image，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor；
#形状为[H, W, C]的numpy.ndarray，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor。    
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  #归一化
    ]),
    'val': transforms.Compose([
        transforms.Resize([224,224]),
        #transforms.Resize(224),
        #transforms.CenterCrop(224),  #将给定的PIL.Image进行中心切割，得到给定的size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
plt.ion()   # interactive mode
dsets = {}
dsets['train'] = dataset(imgroot=traindata_root,anno_pd=train_pd,
                           transforms=data_transforms["train"],
                           )
dsets['val'] = dataset(imgroot=testdata_root,anno_pd=val_pd,
                           transforms=data_transforms["val"],
                           )


dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=8, shuffle=True, num_workers=4)  
                for x in ['train', 'val']}  
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}  
#dset_classes = dsets['train'].classes  
  
use_gpu = torch.cuda.is_available()
def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):  
    since = time.time()  
  
    best_model = model  
    best_acc = 0.0  
  
    for epoch in range(num_epochs):  
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))  
        print('-' * 10)  
  
        # Each epoch has a training and validation phase  
        for phase in ['train', 'val']:  
            if phase == 'train':  
                optimizer = lr_scheduler(optimizer, epoch)  
                model.train(True)  # Set model to training mode  
            else:  
                model.train(False)  # Set model to evaluate mode  
  
            running_loss = 0.0  
            running_corrects = 0  
  
            # Iterate over data.  
            for data in dset_loaders[phase]:  
                # get the inputs  
                inputs, labels = data  
  
                # wrap them in Variable  
                if use_gpu:  
                    inputs, labels = Variable(inputs.cuda()),Variable(labels.cuda())  
                else:  
                    inputs, labels = Variable(inputs), Variable(labels)  
  
                # zero the parameter gradients  
                optimizer.zero_grad()  
  
                # forward  
                outputs = model(inputs)  
                _, preds = torch.max(outputs.data, 1)  
                loss = criterion(outputs, labels)  
  
                # backward + optimize only if in training phase  
                if phase == 'train':  
                    loss.backward()  
                    optimizer.step()  
  
                # statistics  
                running_loss += loss.data[0]  
                running_corrects += torch.sum(preds == labels.data)  
  
            epoch_loss = running_loss / dset_sizes[phase]  
            epoch_acc = running_corrects / dset_sizes[phase]  
  
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(  
                phase, epoch_loss, epoch_acc))  
  
            # deep copy the model  
            if phase == 'val' and epoch_acc > best_acc:  
                best_acc = epoch_acc  
                best_model = copy.deepcopy(model)  
  
        print()  
  
    time_elapsed = time.time() - since  
    print('Training complete in {:.0f}m {:.0f}s'.format(  
        time_elapsed // 60, time_elapsed % 60))  
    print('Best val Acc: {:4f}'.format(best_acc))  
    return best_model  
def exp_lr_scheduler(optimizer, epoch, init_lr=0.0001, lr_decay_epoch=8):  
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""  
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))  
  
    if epoch % lr_decay_epoch == 0:  
        print('LR is set to {}'.format(lr))  
  
    for param_group in optimizer.param_groups:  
        param_group['lr'] = lr  
  
    return optimizer
model_ft = models.densenet161(pretrained=False)
model_ft.load_state_dict(torch.load('../input/pretrained-pytorch-models/densenet161-17b70270.pth'))
num_ftrs = model_ft.classifier.in_features
#model_ft.classifier = nn.Linear(num_ftrs, 100)
model_ft.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 100)
        )
for param in model_ft.parameters():
    param.requires_grad = True
for param in model_ft.features.conv0.parameters():
    param.requires_grad = False
for param in model_ft.features.norm0.parameters():
    param.requires_grad = False
for param in model_ft.features.relu0.parameters():
    param.requires_grad = False
for param in model_ft.features.pool0.parameters():
    param.requires_grad = False
for param in model_ft.features.denseblock1.parameters():
    param.requires_grad = False
for param in model_ft.features.transition1.parameters():
    param.requires_grad = False
for param in model_ft.features.denseblock2.parameters():
    param.requires_grad = False
for param in model_ft.features.transition2.parameters():
    param.requires_grad = False
#for param in model_ft.features.denseblock3.parameters():
#    param.requires_grad = False
#for param in model_ft.features.transition3.parameters():
#    param.requires_grad = False
if use_gpu:  
    model_ft = model_ft.cuda()  
  
criterion = nn.CrossEntropyLoss()  
# Observe that all parameters are being optimized  
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9) 
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.01)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,  
                       num_epochs=50) 
torch.save(model_ft, 'model982.pth')