!pip install albumentations > /dev/null 2>&1
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import time



import os

import cv2



import albumentations

from albumentations import torch as AT



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



import torch

from torch import Tensor



import torch.nn as nn

import torch.nn.functional as F



import torchvision

from torchvision import models



import torch.optim as optim



from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SubsetRandomSampler
# Constantes

data_dir = '../input/histopathologic-cancer-detection'



seed=42



img_size = 32

batch_size = 32

epochs = 100

num_workers = 4



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transforms = {

    'train' : albumentations.Compose([

        albumentations.Resize(img_size, img_size),

        albumentations.HorizontalFlip(),

        albumentations.RandomBrightness(),

        albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.10),

        albumentations.JpegCompression(80),

        albumentations.HueSaturationValue(),

        albumentations.Normalize(),

        AT.ToTensor()

    ]),

    

    'valid' : albumentations.Compose([

    albumentations.Resize(img_size, img_size),

    albumentations.Normalize(),

    AT.ToTensor()

    ]),

    

    'test' : albumentations.Compose([

    albumentations.Resize(img_size, img_size),

    albumentations.Normalize(),

    AT.ToTensor()

    ])

}
classes = ('no cancer', 'cancer')



df_labels = pd.read_csv(f'{data_dir}/train_labels.csv')

df_labels.head()
dic_labels = {k:v for k, v in zip(df_labels.id, df_labels.label)}

list(dic_labels.items())[:10]
class CancerDataset(Dataset):

    def __init__(self, datafolder,

                 datatype='train',

                 transform = albumentations.Compose([albumentations.Resize(224, 224), albumentations.Normalize(), AT.ToTensor()]),

                 labels_dict={}

                ):

        self.datafolder = datafolder

        self.datatype = datatype

        self.image_files_list = [s for s in os.listdir(datafolder)]

        self.transform = transform

        self.labels_dict = labels_dict



    def __len__(self):

        return len(self.image_files_list)



    def __getitem__(self, idx):

        img_name = os.path.join(self.datafolder, self.image_files_list[idx])

        img = cv2.imread(img_name)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = self.transform(image=img)

        image = image['image']



        img_name_short = self.image_files_list[idx].split('.')[0]



        if self.datatype == 'train':

            label = self.labels_dict[img_name_short]

        else:

            label = 0

        return image, label
# indices for validation

tr, val= train_test_split(df_labels.id, stratify=(df_labels.label), test_size=0.1, random_state=seed)

#tr[:10], val[:10], len(tr), len(val)
image_datasets = {x[0] : CancerDataset(datafolder=f'{data_dir}/{x[1]}/', datatype=f'{x[1]}', transform=data_transforms[x[0]], labels_dict=dic_labels)

                  for x in [

                      #dataset-name, dataset-dir

                      ('train', 'train'),

                      ('valid', 'train'),

                      ('test',  'test')]}



indexes = {'train' : tr.index,

           'valid' : val.index}



sampler = {x : SubsetRandomSampler(list(indexes[x]))

           for x in ['train', 'valid']}



dataset_sizes = {x: len(indexes[x])

                 for x in ['train', 'valid']}



loader = {'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, num_workers=num_workers, sampler=sampler['train']),

          'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, num_workers=num_workers, sampler=sampler['valid']),

          'test'  : torch.utils.data.DataLoader(image_datasets['test'],  batch_size=batch_size, num_workers=num_workers)}
def imshow(img):

    #img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()
# get some random training images

dataiter = iter(loader['train'])

images, labels = dataiter.next()



# show images

imshow(torchvision.utils.make_grid(images))
# print labels

print(' '.join('%s,' % labels[j].numpy() for j in range(batch_size)))
def Net(output_dim=2):

    model_ft = models.resnet18(pretrained=True)



    #for i, param in model_ft.named_parameters():

    #    param.requires_grad = False



    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, output_dim)

        

    model_ft = model_ft.to(device)



    return model_ft
net = Net()

net = net.to(device)

net
criterion = nn.CrossEntropyLoss()



#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

optimizer = optim.AdamW(net.parameters(), lr=0.01)
def train(net, criterion, optimizer):

    since = time.time()



    for epoch in range(epochs):  # loop over the dataset multiple times



        running_loss = 0.0

        for i, data in enumerate(loader['train'], 0):

            # get the inputs; data is a list of [inputs, labels]

            #inputs, labels = data

            inputs, labels = data[0].to(device), data[1].to(device)



            # zero the parameter gradients

            optimizer.zero_grad()



            # forward + backward + optimize

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()



            # print statistics

            running_loss += loss.item()

            if i % 1000 == 999:    # print every 100 mini-batches

                print('[%d, %5d] loss: %.5f - time: %.2f' % (epoch + 1, i + 1, running_loss / 2000, time.time() - since))

                running_loss = 0.0



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



train(net, criterion, optimizer)
dataiter = iter(loader['valid'])

images, labels = dataiter.next()



# print images

imshow(torchvision.utils.make_grid(images))



# print labels

print(' '.join('%s,' % labels[j].numpy() for j in range(batch_size)))
outputs = net(images.to(device))

_, predicted = torch.max(outputs.cpu(), 1)



print('Predicted: ', ' '.join('%5s' % predicted[j].numpy() for j in range(batch_size)))
correct = 0

total = 0

with torch.no_grad():

    for data in loader['valid']:

        images, labels = data

        outputs = net(images.to(device))

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted.cpu() == labels).sum().item()



print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))
def eval_net(net):

    class_correct = list(0. for i in range(10))

    class_total = list(0. for i in range(10))

    with torch.no_grad():

        for data in loader['valid']:

            images, labels = data

            outputs = net(images.to(device))

            _, predicted = torch.max(outputs, 1)

            c = (predicted.cpu() == labels).squeeze()

            for i in range(len(c)):

                label = labels[i]

                class_correct[label] += c[i].item()

                class_total[label] += 1



    for i in range(len(classes)):

        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

        

eval_net(net)
# referencias

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py