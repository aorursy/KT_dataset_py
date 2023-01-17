#resources:

#https://github.com/HHTseng/video-classification
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import numpy as np

import io

import base64

from IPython.display import HTML

import matplotlib.pyplot as plt

from PIL import Image

import cv2

import os

import torch

from torch.utils.data.dataset import Dataset

import torchvision.transforms as transforms

from torch.utils.data import DataLoader



import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.autograd import Variable

import torchvision

from torchvision import datasets, models, transforms

import time

FOLDER = "/kaggle/input//dataucf//"

plt.rcParams['figure.figsize'] = 12, 8

folderucf = '/kaggle/input/ucf101/UCF-101/'

folderkaggle ='/kaggle/working/'

# from __future__ import print_function, division

# plt.ion()   # interactive mode

use_gpu = True and torch.cuda.is_available()

#error

os.chdir("/kaggle/input/utilityucf101/")

from functions import *

data_path = "/kaggle/input/ucf101/UCF-101//"    # define UCF-101 RGB data path

action_name_path = '/kaggle/input/utilityucf101/UCF101actions.pkl'

save_model_path = "/kaggle/input/ResNetCRNN_ckpt/"

with open(FOLDER+"train1.txt") as f:

    content = f.readlines()

train = [x.strip().split(" ") for x in content] 

with open(FOLDER+"/train_test//validation1.txt") as f:

    content = f.readlines()

validation = [x.strip().split(" ") for x in content] 



with open(FOLDER+"/train_test//trainlist01.txt") as f:

    content = f.readlines()

all_data = [x.strip().split(" ") for x in content] 



with open(FOLDER+"train_test/sample.txt") as f:

    content = f.readlines()

sample = [x.strip().split(" ") for x in content] 

sample
filelist = []

for video,label in all_data:

    category = video.split("/")[0]

    filename = video.split("/")[1].split(".avi")[0]    

    directory_name = folderkaggle+'ucfimages//' + category 

    if not os.path.exists(directory_name):

        os.makedirs(directory_name)

    # The directroy needs to be changed according to the directory of UCF 101 Video Files

    video_file = folderucf + video.split("/")[0]+ "//"+video.split("/")[1]

    cap = cv2.VideoCapture(video_file)

    counter = 0

    while(cap.isOpened()):

        frameId = cap.get(1) #current frame number

        ret, frame = cap.read()

        if (ret != True):

            break

#             1 second is 24 frames. Therefore we are spliting the video to 1 second jpg

        if (frameId % 24) == 0:

            f_ = directory_name + "/" + filename + "_" + str(counter) + ".jpg"

            cv2.imwrite(f_, frame)

            counter += 1

    cap.release()

    filelist.append([category + "/" + filename, counter, label])

os.mkdir('/kaggle/working/data')

np.savetxt('/kaggle/working/data/all_images1.txt', (filelist), fmt=['%s', '%s', '%s'], delimiter=' ')
counter = {i:0 for i in range(101)}

train_counter = {i:0 for i in range(100)}

val_counter = {i:0 for i in range(100)}



with open("/kaggle/working/data/all_images1.txt") as f:

    for line in f:

        label = int(line.split(" ")[2])

        counter[label-1] += 1        

trainlist = []

validlist = []

with open("/kaggle/working/data/all_images1.txt") as f:

    for line in f:

        label = int(line.split(" ")[2])

        if label != 37:

            if label > 37:

#                 print(label)

                label -= 1

            filename = line.split(" ")[0] + " " + line.split(" ")[1]

            train_or_val = np.random.rand()

            if train_or_val < 0.21:

                val_counter[label-1] += 1            

                validlist.append([filename, label-1])

            else:

                train_counter[label-1] += 1

                trainlist.append([filename, label-1])
np.savetxt('/kaggle/working/data/train1.txt', (trainlist), fmt=['%s', '%s'], delimiter=' ')

np.savetxt('/kaggle/working/data/val1.txt', (validlist), fmt=['%s', '%s'], delimiter=' ')
plt.subplot(2,1,1),plt.bar(train_counter.keys(), train_counter.values(), 1, color='b',label = 'Train')

plt.title('Training images count per category')



plt.legend()



plt.subplot(2,1,2),plt.bar(val_counter.keys(), val_counter.values(), 1, color='orange',label ='Validation')

plt.legend()

plt.title('Validation images count per category')

plt.show()
for i in train_counter:

    if i < 36:

        j = i - 1

        print (int(train_counter[i]+val_counter[i]/float(counter[i])))

    if i > 35:

        j = i + 1

        print (int(train_counter[i]+val_counter[i]/float(counter[j])))

train_counter[35]
transformation = transforms.Compose([

transforms.RandomCrop(224),

transforms.ToTensor(),

])



class DataClass(Dataset):



    def __init__(self, data_folder, image_folder, file_name, transform=None):

        self.transform = transform

        self.data_folder = data_folder

        self.image_folder = image_folder

        self.height = 240

        self.width = 320

        

        with open(self.data_folder + file_name) as f:

            content = f.readlines()

        

        self.data = np.asarray([

            [i.split(" ")[0], i.split(" ")[1], i.split(" ")[2].split("\n")[0]] for i in content])



    # Generate image files for the given batch of videos

    # return batch_size * longest_sequence * channels * height * width

    def generatebatch(self, meta_batch):

        CHANNELS = 3

        

        folder = self.image_folder

        batch_len = len(meta_batch)



        maximum_video_length = meta_batch[:,1].astype(int).max()        

        arr = []

        for batch_index, file in enumerate(meta_batch):

            

            filename = file[0]

            sequence_len = int(file[1])

            # generate transformation here if you want to

            current_image = []

            for i in range(0, sequence_len): #pad the beginning

                image = cv2.imread(folder + filename + "_" + str(i) + ".jpg")                

                # apply transformation here if you want to

                image = cv2.resize(image, (267,200), interpolation = cv2.INTER_AREA)



                image = image.transpose(2,0,1)

                current_image.append(image)

#             current_image = np.asarray(current_image)

            

            #repeat image/reflection

            current_image = np.tile(current_image, (int(np.ceil(maximum_video_length/float(sequence_len))),1,1,1))

            

            #add it to the batch_array

            arr.append(current_image[:maximum_video_length])            

        return np.asarray(arr)

        

        

    # Get a batch of given batch size

    def getbatch(self, batchsize):

        batch = np.random.choice(len(self.data), batchsize, replace=False)

        batch = self.data[batch]

        labels = batch[:,2].astype(int)

        final_batch = self.generatebatch(batch)

        return final_batch, labels

    

    # Override to give PyTorch size of dataset

    def __len__(self):

        return len(self.data)

FOLDER_DATASET = "/kaggle/working/data/"

IMAGE_DATASET = "/kaggle/working/ucfimages/"



train = DataClass(FOLDER_DATASET, IMAGE_DATASET, "train1.txt")

validation = DataClass(FOLDER_DATASET, IMAGE_DATASET, "val1.txt")
len(train), len(validation)
input, labels = train.getbatch(3)

print(input.shape)
plt.figure(figsize=(30, 20))

plt.subplot(3,2,1),plt.imshow(cv2.cvtColor(input[0][0].transpose(1,2,0), cv2.COLOR_BGR2RGB))

plt.subplot(3,2,2),plt.imshow(cv2.cvtColor(input[0][-1].transpose(1,2,0), cv2.COLOR_BGR2RGB))

plt.subplot(3,2,3),plt.imshow(cv2.cvtColor(input[1][0].transpose(1,2,0), cv2.COLOR_BGR2RGB))

plt.subplot(3,2,4),plt.imshow(cv2.cvtColor(input[1][-1].transpose(1,2,0), cv2.COLOR_BGR2RGB))

plt.subplot(3,2,5),plt.imshow(cv2.cvtColor(input[2][0].transpose(1,2,0), cv2.COLOR_BGR2RGB))

plt.subplot(3,2,6),plt.imshow(cv2.cvtColor(input[2][-1].transpose(1,2,0), cv2.COLOR_BGR2RGB))
counter = {i:0 for i in range(101)}

train_counter = {i:0 for i in range(100)}

val_counter = {i:0 for i in range(100)}

arr = []

with open("/kaggle/working/data/all_images1.txt") as f:

    for line in f:

        label = int(line.split(" ")[1])

        arr.append(label)

arr = np.asarray(arr)

arr[::-1].sort()

arr[:100]
dataloader = {'train' : DataClass(FOLDER_DATASET, IMAGE_DATASET, "train1.txt"),

              'validation' : DataClass(FOLDER_DATASET, IMAGE_DATASET, "val1.txt")}

input, label = dataloader['train'].getbatch(3)

input = Variable(torch.from_numpy(input).float())
class CNNGRU(nn.Module):

    def __init__(self):

        super(CNNGRU, self).__init__()

        self.input_dim = 1000

        self.hidden_layers = 101

        self.rnn_layers = 2

#         self.classes = 101

#         self.sample_rate = 12

        

        self.conv = torchvision.models.resnet18(pretrained=False)

        for param in self.conv.parameters():

            param.requires_grad = False



        self.lstm = nn.LSTM(self.input_dim, self.hidden_layers, self.rnn_layers)

        self.gru = nn.GRU(self.input_dim, self.hidden_layers, self.rnn_layers, dropout=0.2)

#         self.linear = nn.Linear(

#             in_features=self.hidden_layers, out_features=self.classes)



    def forward(self, x):

        n, t,c, w, h = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)

        x = x.view(t*n,c,w,h)

        conv_output = self.conv(x) #convolve allframes       

        conv_output = conv_output.view(n,t,-1).transpose(1,0)

#         conv_output = self.conv(x).view(x.size(0),x.size(1),-1).transpose(1,0)

        out, _ = self.gru(conv_output) # pass convolution to gru

        lstm_output = out[-1, :, :].data

#         print(lstm_output.size())

#         output = self.linear(lstm_output) #linear layer 

        return lstm_output
use_gpu = False

model_ft = CNNGRU()

if use_gpu:

    model_ft = model_ft.cuda()

# print(model_ft)

criterion = nn.CrossEntropyLoss()



#Remove all parameters not to be optimized

ignored_params = list(map(id, model_ft.conv.parameters()))

base_params = filter(lambda p: id(p) not in ignored_params,

                     model_ft.parameters())

                     

# Observe that all parameters are being optimized

optimizer_ft = optim.SGD([{'params': base_params}], lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_parameters = filter(lambda p: p.requires_grad, model_ft.parameters())

params = sum([np.prod(p.size()) for p in model_parameters])

params
def func():

    input, label = dataloader['train'].getbatch(3)

    input = Variable(torch.from_numpy(input).float())

    model_ft(input)

%timeit func()
#Takes about 15minutes to be completed

with open("/kaggle/working/data/all_images1.txt") as f:

    for line in f:

        image_folder = line.split(" ")[0]

        length = line.split(" ")[1]

        image_url =  "/kaggle/working/" + "ucfimages/" + image_folder

        image_resize_url =  "/kaggle/working/" + "ucfimages_r/" + image_folder

        

        

        for i in range(0, int(length)): #pad the beginning

            image = cv2.imread(image_url + "_" + str(i) + ".jpg")                

            image = cv2.resize(image, (267,200), interpolation = cv2.INTER_AREA)

            cv2.imwrite(image_url + "_r_" + str(i) + ".jpg",image)
t = torch.rand(3)

t

r = t.cuda()

r
dataloader = {'train' : DataClass(FOLDER_DATASET, IMAGE_DATASET, "train1.txt"),

              'validation' : DataClass(FOLDER_DATASET, IMAGE_DATASET, "val1.txt")}

class CNNGRU(nn.Module):

    def __init__(self):

        super(CNNGRU, self).__init__()

        self.input_dim = 1000

        self.hidden_layers = 101

        self.rnn_layers = 2

#         self.classes = 101

#         self.sample_rate = 12

        

        self.conv = torchvision.models.resnet18(pretrained=False)

        for param in self.conv.parameters():

            param.requires_grad = False



        self.lstm = nn.LSTM(self.input_dim, self.hidden_layers, self.rnn_layers)

        self.gru = nn.GRU(self.input_dim, self.hidden_layers, self.rnn_layers, dropout=0.2)

#         self.linear = nn.Linear(

#             in_features=self.hidden_layers, out_features=self.classes)



    def forward(self, x):

        print(x.shape)

        n, t = x.size(0), x.size(1)

        

        x = x.view(t*n,x.size(2),x.size(3),x.size(4))

        conv_output = self.conv(x).view(x.size(0),x.size(1),-1).transpose(1,0)

        out, _ = self.gru(conv_output) # pass convolution to gru

        lstm_output = out[-1, :, :]

#         print(lstm_output.size())

#         output = self.linear(lstm_output) #linear layer 

        return lstm_output
model_ft = CNNGRU()

if use_gpu:

    model_ft = model_ft.cuda()

# print(model_ft)

criterion = nn.CrossEntropyLoss()



#Remove all parameters not to be optimized

ignored_params = list(map(id, model_ft.conv.parameters()))

base_params = filter(lambda p: id(p) not in ignored_params,

                     model_ft.parameters())

                     

# Observe that all parameters are being optimized

optimizer_ft = optim.SGD([{'params': base_params}], lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

start = time.time()

input, labels = dataloader['train'].getbatch(2)

input = Variable(torch.from_numpy(input)).float()

labels = Variable(torch.from_numpy(labels))

output = model_ft(input)

loss = criterion(output, labels)

print(loss)

print ("Time taken", time.time() - start)
# a = np.arange(12)

# a[0:6] = 0

# a[6:] = 1

a = np.asarray([['a1','a2','a3','a5','a5','a6'],['b1','b2','b3','b4','b5','b6']])

print(a)

print(a.reshape(-1))

b = a.reshape(2,6)

print(b)

print("\n\n\n")

# print(a.reshape(6,2))

print(b.transpose(1,0))
def train_model(model, criterion, optimizer, scheduler, dataloader, batch_size, use_gpu, num_epochs=25):

    since = time.time()

    dataset_sizes = {x: len(dataloader[x]) for x in ['train', 'validation']}

    best_model_wts = model.state_dict()

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'validation']:

            if phase == 'train':

                scheduler.step()

                model.train(True)  # Set model to training mode

            else:

                model.train(False)  #  Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0

            start = time.time()

            # Iterate over data.

            for i in range(int(dataset_sizes[phase]/batch_size)):

                # get the inputs

                inputs, labels = dataloader[phase].getbatch(batch_size)



                # wrap them in Variable

                if use_gpu:

                    inputs = Variable(torch.from_numpy(inputs).float().cuda())

                    labels = Variable(torch.from_numpy(labels).cuda())

                else:

                    inputs, labels = Variable(torch.from_numpy(inputs).float()), Variable(torch.from_numpy(labels))



                # zero the parameter gradients

                optimizer.zero_grad()

                if i%100 == 99:

                    print('{:.0f} videos in {:.0f}m {:.0f}s'.format(100*float(batch_size), 

                                                                    (time.time() - start) // 60, (time.time() - start) % 60))

                    start = time.time()

                # forward

                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)

#                 print(outputs.view(-1), labels.view(1))

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase

                if phase == 'train':

                    loss.backward()

                    optimizer.step()

                

                # statistics

                running_loss += loss.data[0]

                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects / dataset_sizes[phase]



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'validation' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = model.state_dict()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloader, 2, use_gpu, num_epochs=25)