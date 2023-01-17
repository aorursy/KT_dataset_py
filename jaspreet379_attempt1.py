#importing required libraries

import os

import torch

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torchvision.transforms as transforms

import torch.nn as nn

import torch.optim as optim

from torchvision import models

import torchvision.datasets as datasets

from PIL import Image

from torch.utils.data import DataLoader, Dataset, random_split



device = torch.device('cuda:0')

batch_size = 32

data_path = {'train' : "../input/padhai-hindi-vowel-consonant-classification/train/train/", 'test' : "../input/padhai-hindi-vowel-consonant-classification/test/test/"}

print(torch.cuda.get_device_name(0))
#For converting the dataset to torchvision dataset format

class HindiVowelConsonantDataset(Dataset):

    

    def __init__(self, data_path, transform = None, train = True):

        self.train_img_path = data_path['train']

        self.test_img_path = data_path['test']

        self.train_img_files = os.listdir(self.train_img_path)

        self.test_img_files = os.listdir(self.test_img_path)

        self.transform = transform

        self.train = train

    

    def __len__(self):

        return len(self.train_img_files)

    

    def __getitem__(self, indx):

            

        if self.train:  

            

            if indx >= len(self.train_img_files):

                raise Exception("Index should be less than {}".format(len(self.train_img_files)))

               

            image = Image.open(self.train_img_path + self.train_img_files[indx]).convert('RGB')

            labels = self.train_img_files[indx].split('_')

            V = int(labels[0][1])

            C = int(labels[1][1])

            val = 12*C + V

            label = torch.tensor(val)



            if self.transform:

                image = self.transform(image)



            return image, label

        

        if self.train == False:

            image = Image.open(self.test_img_path + self.test_img_files[indx]).convert('RGB')

            if self.transform:

                image = self.transform(image)



            return image, self.test_img_files[indx]
transform = transforms.Compose([transforms.ToTensor(),

                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
data = HindiVowelConsonantDataset(data_path, transform = transform, train = True)



train_size = int(0.9 * len(data))

test_size = len(data) - train_size



train_data, validation_data = random_split(data, [train_size, test_size])



train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=False)
test_data = HindiVowelConsonantDataset(data_path, transform = transform, train = False)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=False)
vgg = models.vgg16_bn()
final_in_features = vgg.classifier[6].in_features

vgg.classifier[6] = nn.Linear(final_in_features, 408) #another way of doing output layer=10
print(vgg)
vgg = vgg.to(device)

loss_fn = nn.CrossEntropyLoss()

opt = optim.SGD(vgg.parameters(), lr=0.05)
def evaluation(dataloader, model):

    total, correct = 0, 0

    for data in dataloader:

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        _, pred = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (pred == labels).sum().item()

    return 100 * correct / total
def test_output(dataloader,model):

    for i, data in enumerate(dataloader, 0):

        inputs ,index = data

        inputs = inputs.to(device)

        output = model(inputs)

        _, pred = torch.max(output.data, 1)

        y_pred.append(pred)

        ind.append(index)

    return np.array(y_pred),np.array(ind)
loss_epoch_arr = []

max_epochs = 39

train = []

val = []

n_iters = np.ceil(9000/batch_size)



for epoch in range(max_epochs):



    for i, data in enumerate(train_loader, 0):



        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

      

        opt.zero_grad()

        

        outputs = vgg(inputs)

        

        loss = loss_fn(outputs, labels)

        loss.backward()

        opt.step()

        

        if i % 20 == 0:

            print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))

            

        del inputs, labels, outputs

        torch.cuda.empty_cache()

        

    loss_epoch_arr.append(loss.item())

    v = evaluation(validation_loader, vgg)

    val.append(v)

    t = evaluation(train_loader, vgg)

    train.append(t)

    print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (

        epoch, max_epochs, 

        v, t))

        

    

plt.plot(loss_epoch_arr)

plt.show()
y_pred = []

ind = []

Y,index = test_output(test_loader,vgg)
out = []

img = []
for i in range(len(Y)-1):

    for j in range(32):

        out.append(Y[i][j].item())

        img.append(index[i][j])

for i in range(16):

    out.append(Y[312][i].item())

    img.append(index[312][i])
for i in range(len(out)):

    out[i]='V'+str(out[i]%12)+'_'+'C'+str(out[i]//12)
no=[]

for i in range(len(out)):

    l,r = img[i].split('.')

    no.append(int(l))
submission = {}

submission['ImageId'] = img

submission['Class'] = out

submission['value'] = no



submission = pd.DataFrame(submission)

submission = submission[['value', 'ImageId', 'Class']]

submission = submission.sort_values(['value'])

submission = submission[['ImageId', 'Class']]

submission.to_csv("submisision.csv", index=False)