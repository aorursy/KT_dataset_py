import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import random
import time
import copy

import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print(os.listdir("../input"))
use_gpu = torch.cuda.device_count() > 0
print("{} GPU's available:".format(torch.cuda.device_count()) )
def get_images(path2images, path2labels):
    
    df_images = pd.read_csv(path2images, header=None)
    df_labels = pd.read_csv(path2labels)
    
    X = np.vstack(df_images.values)
    #X = X / 255.   # scale pixel values to [0, 1]
    #X = X.astype(np.float32)
    X = X.astype(np.uint8)    
    X = X.reshape(-1, 1, 110, 110) # return each images as 1 x 110 x 110
    
    y = df_labels['Volcano?'].values
    
    return X, y
path2trainImages = '../input/volcanoes_train/train_images.csv'
path2trainLabels = '../input/volcanoes_train/train_labels.csv'

X, y = get_images(path2trainImages, path2trainLabels)
path2testImages = '../input/volcanoes_test/test_images.csv'
path2testLabels = '../input/volcanoes_test/test_labels.csv'

X_test, y_test = get_images(path2trainImages, path2trainLabels)
print('Train Dataset shape: {} Labels: {}'.format(X.shape, y.shape))
print('Test Dataset shape: {} Labels: {}'.format(X_test.shape, y_test.shape))
print('Is Volcano: {}'.format(np.sum(y==1)))
print('Is Not Volcano: {}'.format(np.sum(y==0)))
X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42,
                                                    stratify = y)
class VolcanoesDataset(Dataset):

    def __init__(self, X, y, transforms=None):
        self.transform = transforms    
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        image = self.X[index]
        label = self.y[index]        
        
        image = image.reshape(110, 110, 1)

        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.X)
class AdjustGamma(object):
    def __call__(self, img):
        return transforms.functional.adjust_gamma(img, 0.8, gain=1)
class AdjustContrast(object):
    def __call__(self, img):
        return transforms.functional.adjust_contrast(img, 2)
class AdjustBrightness(object):
    def __call__(self, img):
        return transforms.functional.adjust_brightness(img, 2)
data_transforms = {
    'train': transforms.Compose([
        #CloneArray(),
        transforms.ToPILImage(), # because the input dtype is numpy.ndarray
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        AdjustGamma(),
        AdjustContrast(),
        #AdjustBrightness(),
        transforms.ToTensor()
    ]),
    'valid': transforms.Compose([
        transforms.ToPILImage(), # because the input dtype is numpy.ndarray
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        AdjustGamma(),
        AdjustContrast(),
        transforms.ToTensor(),
    ]),
}
dsets = {
    'train': VolcanoesDataset(X_train, y_train, transforms=data_transforms['train']),
    'valid': VolcanoesDataset(X_valid, y_valid, transforms=data_transforms['valid']),
    'test':  VolcanoesDataset(X_test, y_test, transforms=data_transforms['valid']),
}
batch_size = 32
random_seed = 3
valid_size = 0.2
shuffle = True
class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_train])

samples_weight = torch.from_numpy(samples_weight)

sampler = {'train':  WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight)),
          'valid': None,
          'test': None}
def create_dataLoader(dsets, batch_size, sampler={'train': None, 'valid': None,'test': None},
                      pin_memory=False):
    dset_loaders = {} 
    for key in dsets.keys():
        if sampler[key] != None:
            dset_loaders[key] = DataLoader(dsets[key], batch_size=batch_size, sampler=sampler[key], pin_memory=pin_memory)
        else:          
            dset_loaders[key] = DataLoader(dsets[key], batch_size=batch_size, pin_memory=pin_memory, shuffle=False)

    return dset_loaders
dset_loaders = create_dataLoader(dsets, batch_size, sampler, pin_memory=False)
dset_loaders.keys()
image, label = next(iter(dset_loaders['train']))
print(image.size(), label.size())
def plot_volcanos(dset_loaders, is_train = True, preds_test = [], preds_train = []):
    
    X, y = next(iter(dset_loaders))
    X, y = X.numpy(), y.numpy()
    
    plt.figure(figsize=(20,10))
    for i in range(0, 4):
        plt.subplot(1,4,i+1)
        
        rand_img = random.randrange(0, X.shape[0])
        img = X[rand_img,:,:,:]
        
        #img = np.clip(img, 0, 1.0)
        plt.imshow(img[0,:,:], cmap = 'gray')
        plt.title('Volcano: {}'.format(y[rand_img]))
        plt.axis('off')
plot_volcanos(dset_loaders['train'])
class Net(nn.Module):
    def __init__(self, nb_out=2, nb_channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(nb_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(9216, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, nb_out)
                
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = Net()
print(model)
if use_gpu:
    print("Using all GPU's ")
    model = torch.nn.DataParallel(model) #device_ids=[1,3]
    model.cuda()
else:
    print("Using CPU's")
def evaluate_model(loader, model, loss_fn, use_gpu = False):
    
    total_loss = 0
    for i, ( inputs, labels) in enumerate(loader):     
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
                
        # forward pass
        outputs = model(inputs)
        
        # loss
        loss = loss_fn(outputs, labels)
        
        # metrics
        total_loss += loss.item()
            
    return (total_loss / i)
def train(model, train_loader, test_loader ,num_epochs, loss_fn, optimizer, patience  ):
    
    loss_train = []
    loss_test = []
    best_test_acc =  np.inf
    
    patience_count= 0
    ii_n = len(train_loader)
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            print('\rpredict: {}/{}'.format(i, ii_n - 1), end='')

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            predict = model(inputs)
            
            loss = loss_fn(predict, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_train.append(loss.item())
        loss_test.append( evaluate_model(test_loader, model,loss_fn, use_gpu) )
        
        print('\nEpoch: {}  Loss Train: {}  Lost Test: {}'.format(epoch, loss_train[-1], loss_test[-1]), end='\n')
        
        #Early stopping
        if(best_test_acc > loss_test[-1]):
            patience_count = 0
            best_test_acc = loss_test[-1]
            best_model = copy.deepcopy(model)

        if(patience_count > patience):
            break;

        patience_count += 1
        
        
    print('\rDone!')
    return loss_train, loss_test, model
loss_fn = torch.nn.CrossEntropyLoss()
optimizer =  optim.RMSprop(model.parameters(), lr=1e-4)
num_epochs = 100
patience = 5
params = {'model' : model, 
        'train_loader':dset_loaders['train'],
         'test_loader':dset_loaders['valid'],
         'num_epochs': num_epochs,
         'loss_fn': loss_fn,
         'optimizer': optimizer, 
         'patience': patience 
         }
loss_train, loss_test, model = train(**params)
def predict(dset_loaders, model,use_gpu=False):
    
    predictions = []
    labels_lst = []

    ii_n = len(dset_loaders)
    start_time = time.time()

    for i, (inputs, labels) in enumerate(dset_loaders):
                   
        if use_gpu:
          inputs = inputs.cuda()
          labels = labels.cuda()

        inputs = Variable(inputs)
        labels = Variable(labels)

        predictions.append(model(inputs).data)
        labels_lst.append(labels)
        
        print('\rpredict: {}/{}'.format(i, ii_n - 1), end='')
    print(' ok')
    print('Execution time {0:.2f} s'.format(round(time.time()- start_time), 2))
    if len(predictions) > 0:
        return {'pred': torch.cat(predictions, 0), 'true': torch.cat(labels_lst, 0) }
results_fine = {}
true_dict = {}
pred_dict = {}

for k in dset_loaders.keys(): 
    
    prediction = predict(dset_loaders[k], model, use_gpu=use_gpu)    
    
    _, predicted = torch.max(prediction['pred'], 1)  
    if use_gpu:
        true, pred = prediction['true'].cpu(), predicted.cpu()
    
    true, pred = true.data.numpy(), pred.numpy()
    correct = (true == pred).sum()
    print('{}: {}/{}'.format(k, correct, len(prediction['pred'])))
    print('\n----------------\n')
    
    true_dict[k] = true
    pred_dict[k] = pred
plt.figure(figsize=(20,5))

idx = 1
for k in dset_loaders.keys():
    true, pred = true_dict[k], pred_dict[k]
    plt.subplot(1,3,idx);

    mc = confusion_matrix(true, pred)
    plt.imshow(mc/mc.sum(axis=0), cmap = 'jet');
    plt.colorbar();
    plt.axis('off');
    
    plt.title(k, fontsize=20)
    plt.suptitle('Confusion Matrix', fontsize=22)
    idx+=1
print(classification_report(pred_dict['valid'], true_dict['valid']))  
print(classification_report(pred_dict['test'], true_dict['test']))  
