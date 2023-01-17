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
!tar -zxvf ../input/cifar-10-python.tar.gz
import torch

import torchvision

from torchvision import transforms

import matplotlib.pyplot as plt
means = (0.491, 0.482, 0.447)

stds = (0.247, 0.243, 0.262)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(means,stds)])

trainset = torchvision.datasets.CIFAR10(root='.', train=True,

                                        download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,

                                          shuffle=True, num_workers=2)



testset = torchvision.datasets.CIFAR10(root='.', train=False,

                                       download=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=128,

                                         shuffle=False, num_workers=2)



classes = ('plane', 'car', 'bird', 'cat',

           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def imshow(img, figsize=None, denormalize=True):

    npimg = img.numpy()

    npimg = np.transpose(npimg, (1, 2, 0))

    if denormalize:

        npimg = npimg * stds + means

    npimg = np.clip(npimg, 0, 1)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot('111')

    ax.set_aspect('equal')

    ax.imshow(npimg)
images, labels = iter(trainloader).next()

print(type(images))

print(images.shape)
grid = torchvision.utils.make_grid(images[:32])

imshow(grid)

for i in range(32):

    if i % 8 == 0:

        print('\n')

    print(classes[labels[i]],end=' ')
from tqdm import tqdm_notebook

import tqdm

import time
from torch import nn

import torch.nn.functional as F
class ConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, batch_norm=True, pooling=None, pooling_kernel_size=None, dropout=None):

        super().__init__()

        layers = []

        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size))

        if batch_norm:

            layers.append(nn.BatchNorm2d(output_channels))

        if pooling == 'max':

            layers.append(nn.MaxPool2d(pooling_kernel_size))

        elif pooling == 'avg':

            layers.append(nn.AvgPool2d(pooling_kernel_size))

            

        layers.append(nn.ReLU())

        if dropout is not None:

            layers.append(nn.Dropout2d(p=dropout))

        self.block = nn.Sequential(*layers)

        

    def forward(self, x):

        return self.block(x)



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv_part = nn.Sequential(

            ConvBlock(3, 96, kernel_size=5), # 96 x 28 x 28

            ConvBlock(96, 128, kernel_size=5, pooling='max', pooling_kernel_size=2, batch_norm=False, dropout=0.2), # 128 x 12 x 12

            ConvBlock(128, 256, kernel_size=5, pooling='max', pooling_kernel_size=2,  batch_norm=False, dropout=0.2), # 128 x 4 x 4

            ConvBlock(256, 256, kernel_size=3,  batch_norm=False, dropout=0.2),

            nn.AdaptiveAvgPool2d((1,1))

        )

        

        self.dense1 = nn.Sequential(

#             nn.BatchNorm1d(256),

            nn.Dropout(p=0.2),

            nn.Linear(256, 10)

        )

        

        

    def linear_size(self):

        x = torch.empty(1, 3, 32, 32, dtype=torch.float32).to(next(iter(self.parameters())).device)

        x = self.conv_part(x)

        print(x.size())

        

    def forward(self,x):

        x = self.conv_part(x)

        x = x.view(x.size(0), -1)

        x = self.dense1(x)

        return x

        
def foo(first, second,**kws):

    print(first, second, kws)

    

foo(**{'first':1, 'second': 2, 'third': 3})
class HyperSampler:

    def sample(self, size):

        raise NotImplementedError
class LambdaSampler(HyperSampler):

    def __init__(self, fn):

        super().__init__()

        self._fn = fn

        

    def sample(self, size):

        return self._fn(size)
class FixedSampler(HyperSampler):

    def __init__(self, array):

        super().__init__()

        self._arr = np.asarray(array)

    

    def sample(self, size):

        if len(self._arr) < size:

            raise ValueError("len(self._arr) < size")

        return self._arr[:size]
class ArraySampler(HyperSampler):

    def __init__(self, array):

        super().__init__()

        self._arr = np.asarray(array)

        

    def sample(self, size):

        idx = np.random.choice(len(self._arr), size=size)

        return self._arr[idx]
def h_enum(*values):

    return ArraySampler(values)



def h_set(values):

    return ArraySampler(values)



def h_fixed_enum(*values):

    return FixedSampler(values)



def h_fixed_set(value):

    return FixedSampler(values)
def split_dictionary(d, criterion):

    a, b = {}, {}

    for k, v in d.items():

        if criterion(k, v):

            a[k] = v

        else:

            b[k] = v

    return a, b
print(dict({'a': 5} ,**{'b': 6}))
class BestModel:

    def __init__(self, path, initial_criterion):

        self.path = path

        self.criterion = initial_criterion

        

    def update(self, model, optimizer, criterion):

        self.criterion = criterion

        torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'criterion': criterion}, self.path)

        

    def load_model_data(self):

        return torch.load(self.path)

    

    def restore(self, model, optimizer):

        model_data = self.load_model_data()

        model.load_state_dict(model_data['model_state'])

        optimizer.load_state_dict(model_data['optimizer_state'])
from collections import defaultdict

import itertools



def hyper_search(num_trials,

                 model_factory, optimizer_factory,

                 training_set, validation_set,

                 training_function, best_model,

                 parameters, device, repeats=1, random_seed=438):

    param_groups = ('model', 'optimizer', 'train_loader', 'val_loader', 'training')

    fixed_params, sampled_params = {}, {}

    for param_group in param_groups:

        sampled, fixed = split_dictionary(parameters.get(param_group, {}), lambda _,v: isinstance(v, HyperSampler))

        fixed_params[param_group] = fixed

        sampled_params[param_group] = sampled

        

    def flatten(dict_of_dicts):

        result = {}

        for pg, params in dict_of_dicts.items():

            for k,v in params.items():

                yield pg + '_' + k, v

                

    def merge(frame_dict, flattened):

        for k, v in flattened:

            frame_dict[k].append(v)

        

    random_queue = defaultdict(dict)

    for p in param_groups:

        for k, population in sampled_params[p].items():

            sample = population.sample(num_trials)

            random_queue[p][k] = sample

            

#     print(random_queue)

#     print(fixed_params)

    

    best_setting = None

    

    result_table = defaultdict(list)

    

    def decay(x):

#         print(x, type(x))

        if isinstance(x, np.generic):

            return x.item()

        return x

    

    for trial in tqdm_notebook(range(num_trials), total=num_trials):

        np.random.seed(random_seed)

        torch.manual_seed(random_seed)

        current_trial_random = {pg: {k: decay(sample[trial]) for k, sample in random_queue[pg].items()} for pg in param_groups}

        iteration_setting = {pg:dict(fixed_params[pg],**current_trial_random[pg]) for pg in param_groups}

        

        print('Trial', trial + 1)

        print('Setting', dict(flatten(current_trial_random)))

        model = model_factory(**iteration_setting['model']).to(device)

        optimizer = optimizer_factory(model.parameters(), **iteration_setting['optimizer'])

#         print(type(iteration_setting['train_loader']['batch_size']))

        train_loader = torch.utils.data.DataLoader(training_set,**iteration_setting['train_loader'])

        val_loader = torch.utils.data.DataLoader(validation_set, **iteration_setting['val_loader'])

        

        try:

            setting_best_model:BestModel = training_function( model, optimizer, train_loader, val_loader, **iteration_setting['training'])

        except Exception as ex:

            print(ex)

        else:

            if setting_best_model.criterion is not None:

                if setting_best_model.criterion < best_model.criterion:

                    best_model, best_setting = setting_best_model, iteration_setting



            print('Trial', trial + 1)

            print('Setting', dict(flatten(current_trial_random)))

            print('criterion', setting_best_model.criterion)

            print('=============')



            merge(result_table, flatten(iteration_setting))

            result_table['criterion'].append(setting_best_model.criterion)

                

    return best_model, best_setting, pd.DataFrame(result_table)

    
# hyper_search(10,None,None,None,None, {

#     'model': {'n_layers' : h_enum(1,2,3), 'dropout': h_enum(0.1,0.2,0.3,0.4,0.5)},

#     'optimizer': {'learning_rate': h_set(np.logspace(-5,0,num=6)), 'class': 'Adam'},

#     'train_loader': {'batch_size': h_enum(64,128,256)},

#     'val_loader': {'batch_size': 512}

# })
from collections import namedtuple


        
def train_model(net, optimizer, trainloader, testloader, criterion, n_epochs, best_model:BestModel=None, patience=5, n_prints=5):

    attempts_left = patience

    

    print_every = len(trainloader) // n_prints

#     print('Printing every', print_every, 'batches')

    

    if best_model is None:

        best_model = BestModel('temp', 10000)

    

    for epoch in tqdm_notebook(range( n_epochs)):  

        if attempts_left < 0:

            break

            

        running_loss = 0.0

        net.train()

        for i, data in tqdm_notebook(enumerate(trainloader, 0), total = len(trainloader)):

            # get the inputs

            inputs, labels = data



            inputs, labels = inputs.cuda(), labels.cuda()



            # zero the parameter gradients

            optimizer.zero_grad()



            # forward + backward + optimize

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            

            

            # print statistics

            running_loss += loss.item()

            if i % print_every == print_every - 1:    # print every 2000 mini-batches

#                 print( labels.tolist())

                grad_norm = 0.0

                for p in net.parameters():

                    grad_norm += p.grad.data.norm().item()

                    

                running_loss /= print_every

                print('[%d, %5d] loss: %.3f grad_norm: %.3f' %

                      (epoch + 1, i + 1, loss, grad_norm))



                running_loss = 0.0

        



        net.eval()



        with torch.no_grad():

            running_val_loss = 0.0

            for i, data in tqdm_notebook(enumerate(testloader, 0), total = len(testloader)):

                inputs, labels = data

                inputs, labels = inputs.cuda(), labels.cuda()



                outputs = net(inputs)

                loss = criterion(outputs, labels)

#                 print(loss.item())



                running_val_loss += loss.item()

                

            val_loss = running_val_loss / len(testloader)

            print('[%d] val_loss: %.3f' % (i+1, val_loss))

            

            if val_loss < best_model.criterion:

                print('Val_loss improved from {} to {}'.format(best_model.criterion, val_loss))

                best_model.update(net,optimizer,val_loss)

                attempts_left = patience

            else:

                print('No improvement attempts left {}'.format(attempts_left))

                attempts_left -= 1

                



    print('Finished Training')

    return best_model

#     best_model.restore(net, optimizer)
net = Net().cuda()

print(net.linear_size())

optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

criterion = nn.CrossEntropyLoss()
best_model = BestModel('conv_hs', 10000)
del optimizer

del net

loader = torch.utils.data.DataLoader(trainset, **dict(batch_size=64))

del loader
saved_model, best_setting, results = hyper_search(1, Net, torch.optim.Adam, training_set=trainset, validation_set=testset, training_function=train_model, best_model=best_model,

             parameters={

                 'model': {},

                 'optimizer': {'lr': 3e-4},

                 'training': {'criterion': nn.CrossEntropyLoss(),

                                      'n_epochs': h_fixed_enum(2),

                                      'patience': 4,

                                      'n_prints': 1},

                 'train_loader': {'batch_size': h_fixed_enum(512), 'shuffle':True},

                 'val_loader': {'batch_size': 256}

             }, device=torch.device('cuda'), random_seed=531)



print(best_setting)
results
best_model = BestModel('conv_v1', 1000)
train_model(net, optimizer, criterion, 40, best_model, patience=6, n_prints=5)
best_model.restore(net, optimizer)
dataiter = iter(testloader)

images, labels = dataiter.next()



net.eval()

with torch.no_grad():

    outputs = net(images[:32].cuda())

    _, predicted = torch.max(outputs.data, 1)

    print(predicted)





# print images

imshow(torchvision.utils.make_grid(images[:32]))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(32)))

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(32)))

from sklearn import metrics
def evaluate(net, loader):

    net.eval()

    all_pred = []

    correct_pred = []

    loss = nn.CrossEntropyLoss()

    with torch.no_grad():

        running_loss = 0.0

        for data in tqdm_notebook(loader):

            images, labels = data 

            output = net(images.cuda())

            running_loss += loss(output, labels.cuda()).item()

            preds = output.max(1)[1].tolist()

            all_pred.extend(preds)

            correct_pred.extend(labels.tolist())

    

    running_loss /= len(loader)

    

    print(metrics.classification_report(correct_pred, all_pred,target_names=classes))

    cm = metrics.confusion_matrix(correct_pred, all_pred)

    cm = pd.DataFrame(cm, index=classes, columns=classes)

    print(cm)

    print('Accuracy:', metrics.accuracy_score(correct_pred, all_pred))

    print('Loss: ', running_loss)
evaluate(net,testloader)
kernels = net.conv_part[0].block[0].weight.data.cpu()

# mm_kernels = (mm_kernels - min_values) / (max_values - min_values)

# print(mm_kernels)

grid_of_kernels = torchvision.utils.make_grid(kernels)

imshow(grid_of_kernels.cpu(), denormalize=True, figsize=(6,6))

plt.show()

# imshow(torchvision.utils.make_grid(kernels.cpu(), normalize=True), denormalize=False)
testimages, _ = next(iter(testloader))
grid = torchvision.utils.make_grid(testimages,nrow=16)

imshow(grid,(16,16))
with torch.no_grad():

    transformed = net.conv_part[:4](testimages.cuda()).cpu()

#     transformed = testimages
transformed.shape
def normalize_image(img):

    tr_img = img.transpose(0,1).reshape(img.size(1), -1)

    channel_min = tr_img.min(dim=1)[0].view(1,-1,1,1)

    channel_max = tr_img.max(dim=1)[0].view(1,-1,1,1)

#     print(channel_max - channel_min)

    return (img - channel_min) / (channel_max - channel_min + 0.01)

    
transformed = normalize_image(transformed)
def show_image_number(num):

    fig = plt.figure(figsize=(12,12))

    total_channels = min(transformed[num].size(0), 96)

    print(total_channels)

    for i in range(total_channels):

        ax = fig.add_subplot(12,8,i+1)

        ax.imshow(transformed[num,i].numpy(), cmap='gray')

        ax.axis('off')

    plt.show()

    imshow(testimages[num])

#     ax.imshow(np.transpose(testimages[num].numpy(),(1,2,0)))

    plt.show()
show_image_number(56)

