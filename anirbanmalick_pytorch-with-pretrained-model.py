import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from zipfile import ZipFile

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('.zip'):
            print(os.path.join(dirname, filename))
            zip = ZipFile((os.path.join(dirname, filename)))
            zip.extractall()
            zip.close()

import os
import shutil
from tqdm import tqdm
import random

data_dir = 'train'
cat_path_name = data_dir + '/cat'
dog_path_name = data_dir + '/dog'
train_test_indices = list(range(1,25000)) 
random.shuffle(train_test_indices)
nnp = round(len(train_test_indices)*0.8)
train_indices = train_test_indices[:nnp]
#print(len(train_indices))
test_indices = [i for i in train_test_indices if i not in train_indices]

train_path_name = data_dir+'/train_classified'
test_path_name = data_dir+'/test_classified'
train_cat_path = train_path_name +'/cat'
test_cat_path = test_path_name + '/cat'
train_dog_path = train_path_name + '/dog'
test_dog_path = test_path_name + '/dog'
make_folders = [train_path_name,test_path_name,train_cat_path,test_cat_path,train_dog_path,test_dog_path]

for path in tqdm(make_folders):
    if not os.path.exists(path):
        os.makedirs(path)

all_files = os.listdir(data_dir+'/')

train_files = [all_files[i] for i in train_indices]
test_files = [all_files[i] for i in test_indices]

def copy_data(files,destination):
    for f in tqdm(files):
        #img = 
        if f.startswith('cat'):
            shutil.copy(data_dir+'/'+f,destination+'/cat')
        if f.startswith('dog'):
            shutil.copy(data_dir+'/'+f,destination+'/dog')
    return

copy_data(files=train_files,destination=train_path_name)
copy_data(files=test_files,destination=test_path_name)
# # TODO: Define transforms for the training data and testing data
import random
random.seed(2)

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.Resize((256,256)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5], 
                                                            [0.5])
                                      ])

test_transforms = transforms.Compose([transforms.Resize((256,256)),
                                       transforms.ToTensor(),
                                      transforms.Normalize([0.5], 
                                                            [0.5])
                                       ])

# # Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir+'/train_classified', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir+'/test_classified', transform=test_transforms)


def sample_data(collection,n_sample):
    indices = list(range(1,len(collection.samples))) 
    random.shuffle(indices)
    indices = indices[:n_sample]
    collection.samples = [collection.samples[idx] for idx in indices]
    collection.targets = [collection.targets[idx] for idx in indices]
    return collection



#train_data = sample_data(train_data,n_sample=1000)
#test_data = sample_data(test_data,n_sample=200)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32,shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=True)
train_data,test_data
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

model = models.densenet121(pretrained=True)
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 512)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(512,64)),
                        ('relu', nn.ReLU()),
    
        ('fc3', nn.Linear(64, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

device = 'cuda'

model.to(device)
# torch.cuda.is_available()
model.classifier
#model = Classifier()


epochs = 3
steps = 0


train_losses, test_losses = [], []
for e in range(epochs):
    #print(e)
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        #print(running_loss)
        
    else:
        test_loss = 0
        accuracy = 0
        #print('Validation Starts!')
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)

                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        model.train()
        
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(test_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
if not os.path.exists('test'):
    shutil.copytree('test1','test/test1')
test_dir = 'test'
submission_data = datasets.ImageFolder(test_dir, transform=test_transforms)
ids = [int(i[0][11:-4]) for i in submission_data.samples]
submission_data.targets = ids
testloader = torch.utils.data.DataLoader(submission_data, batch_size=32,shuffle=False)
samples, _ = iter(testloader).next()
samples = samples.to(device)
fig = plt.figure(figsize=(24, 16))
fig.tight_layout()
output = model(samples[:24])
pred = torch.argmax(output, dim=1)
pred = [p.item() for p in pred]
ad = {0:'cat', 1:'dog'}
for num, sample in enumerate(samples[:24]):
    plt.subplot(4,6,num+1)
    plt.title(ad[pred[num]])
    plt.axis('off')
    sample = sample.cpu().numpy()
    plt.imshow(np.transpose(sample, (1,2,0)))
# # Test out your network!

model.eval()
fn_list = []
pred_list = []

for x, fn in tqdm(testloader):
    with torch.no_grad():
        x = x.to(device)
        logps = model.forward(x)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        pred_list.extend(top_class)
        
        
#         pred = torch.argmax(output, dim=1)
#         fn_list += [n[:-4] for n in fn]
#         pred_list += [p.item() for p in pred]




#pred_list = [i.item() for i in pred_list]
submission = pd.DataFrame({"id":ids, "label":pred_list})
#submission.to_csv('pytorch_sample.csv', index=False)
submission.to_csv('pytorch_sample.csv', index=False)
os.listdir()