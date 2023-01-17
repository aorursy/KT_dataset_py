# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import torch

from torchvision import datasets, transforms, models

from torch.autograd import Variable

import numpy as np

from matplotlib import pyplot as plt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip

#!unzip ngrok-stable-linux-amd64.zip
import os

LOG_DIR = 'runs'

#os.makedirs(LOG_DIR, exist_ok=True)

#get_ipython().system_raw(

#    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'

#    .format(LOG_DIR)

#)
#get_ipython().system_raw('./ngrok http 6006 &')
#! curl -s http://localhost:4040/api/tunnels | python3 -c \

#    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
from PIL import Image

data_dir = '../input/training-a'

name = os.listdir(data_dir)[10]

Image.open(data_dir+"/"+name)
data_dir = '../input/training-c'

print(os.listdir(data_dir)[:5])

len(os.listdir(data_dir))

#10908+19702+24298
print("Final Size:", str(10908+19702+24298))
a_csv = pd.read_csv('../input/training-a.csv')

a_csv.columns
a_csv = a_csv.drop(columns=['original filename', 'scanid',

       'database name original', 'contributing team', 'database name'])

a_csv.iloc[:10, 0:]
c_csv = pd.read_csv('../input/training-c.csv')

c_csv.columns
c_csv = c_csv.drop(columns=['original filename', 'scanid',

       'database name original', 'contributing team', 'database name'])

c_csv.iloc[:10, 0:]
d_csv = pd.read_csv('../input/training-d.csv')

d_csv.columns
d_csv = d_csv.drop(columns=['original filename', 'scanid', 'num', 'database name original',

       'database name'])

d_csv.iloc[:10, 0:]
frames = [a_csv, c_csv, d_csv]

label_csv = pd.concat(frames)
# almost minist

len(label_csv)
path = 'train'

os.mkdir(path)
import os

import shutil

src = '../input/training-a/'

src_files = os.listdir(src)

for file_name in src_files:

    full_file_name = os.path.join(src, file_name)

    if os.path.isfile(full_file_name):

        shutil.copy(full_file_name, path)



print("A Done")
src = '../input/training-c/'

src_files = os.listdir(src)

for file_name in src_files:

    full_file_name = os.path.join(src, file_name)

    if os.path.isfile(full_file_name):

        shutil.copy(full_file_name, path)



print("C Done")
src = '../input/training-d/'

src_files = os.listdir(src)

for file_name in src_files:

    full_file_name = os.path.join(src, file_name)

    if os.path.isfile(full_file_name):

        shutil.copy(full_file_name, path)



print("D Done")
print(len(os.listdir(path)))
t = label_csv.iloc[1000]

print(t)

print("Label: ", t[0])

Image.open(path+"/"+t[1])
import torch

from torch.utils.data import Dataset



class Dataset(Dataset):

    def __init__(self, df, root, transform=None):

        self.data = df

        self.root = root

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        item = self.data.iloc[index]

        

        path = self.root + "/" + item[1]

        image = Image.open(path).convert('L')

        label = item[0]

        

        if self.transform is not None:

            image = self.transform(image)

            

        return image, label
mean = [0.5,]

std = [0.5, ]



train_transform = transforms.Compose([

    transforms.Resize(180),

    transforms.RandomRotation(30),

    #transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    #transforms.Normalize(mean, std)

])



test_transform = transforms.Compose([

        transforms.Resize(180),

        transforms.ToTensor(),

        #transforms.Normalize(mean, std)

])



train_data  = Dataset(label_csv, path, train_transform)

test_data = Dataset(label_csv, path, test_transform)



print("Trainig Samples: ",len(train_data))
from torch.utils.data.sampler import SubsetRandomSampler



#batch size

batch_size=128



# split data 20% for testing

test_size = 0.2

# obtain training indices that will be used for validation

num_train = len(train_data)



# mix data

# index of num of train

indices = list(range(num_train))

# random the index

np.random.shuffle(indices)

split = int(np.floor(test_size * num_train))

# divied into two part

train_idx, test_idx = indices[split:], indices[:split]



# define the sampler

train_sampler = SubsetRandomSampler(train_idx)

test_sampler = SubsetRandomSampler(test_idx)



# prepare loaders

train_loader = torch.utils.data.DataLoader(

    train_data, batch_size=batch_size,

    sampler=train_sampler)



test_loader = torch.utils.data.DataLoader(

    test_data, batch_size=batch_size,

    sampler=test_sampler)



print("Train dataloader:{}".format(len(train_loader)))

print("Test dataloader:{}".format(len(test_loader)))
classes = list()

for i in range(10):

    classes.append(str(i))

classes
!wget https://raw.githubusercontent.com/Iamsdt/DLProjects/master/utils/Helper.py

import Helper
import Helper

from torch.utils.tensorboard import SummaryWriter



data_iter = iter(train_loader)

images, labels = data_iter.next()



# Write images

tb = SummaryWriter()

tb.add_images("Train Images", images)

tb.close()



fig = plt.figure(figsize=(25, 10))

for idx in range(5):

        ax = fig.add_subplot(1, 10, idx + 1, xticks=[], yticks=[])

        ax.imshow(np.squeeze(images[idx]), cmap='gray')

        ax.set_title(classes[labels[idx]])

#model = models.densenet161(pretrained=True)

#model.classifier
#model = Helper.freeze_parameters(model)
import torch.nn as nn

from collections import OrderedDict



classifier = nn.Sequential(

  nn.Linear(in_features=2208, out_features=1024),

  nn.ReLU(),

  nn.Dropout(p=0.3),

  nn.Linear(in_features=1024, out_features=10),

  nn.LogSoftmax(dim=1)  

)

    

#model.classifier = classifier

#model.classifier
import torch.nn as nn



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        

        self.conv1 = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, stride=2, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(32),

            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)

        )

        

        self.conv2 = nn.Sequential(

            nn.Conv2d(32, 64, 3, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(64),

            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)

        )

        

        self.conv3 = nn.Sequential(

            nn.Conv2d(64, 128, 3, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, stride=2, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(128),

            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)

        )

        

        self.conv4 = nn.Sequential(

            nn.Conv2d(128, 256, 3, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(256),

            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)

        )

        

        self.fc = nn.Sequential(

            nn.Linear(256, 128),

            nn.Dropout(0.4),

            nn.ReLU(),

            nn.Linear(128, 10),

            nn.Softmax(dim=1)

        )

                

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        # flaten tensor

        x = x.view(x.size(0), -1)

        return self.fc(x)
from torch import optim

model = Net()



# Gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#move tensor to default device

model.to(device)



criterion = nn.NLLLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
optimizer
global_epoch = 0
epoch = 25
import time

from torch.utils.tensorboard import SummaryWriter



def train(model, train_loader, test_loader,

          epochs, optimizer, criterion, scheduler=None, global_epoch = 0,

          name="model.pt", path=None):

  

    global_epoch = global_epoch + 1



    # Write images

    tb = SummaryWriter()



    # compare overfitted

    train_loss_data, valid_loss_data = [], []

    # check for validation loss

    valid_loss_min = np.Inf

    # calculate time

    since = time.time()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    for epoch in range(epochs):

        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # monitor training loss

        train_loss = 0.0

        valid_loss = 0.0

        total = 0

        correct = 0

        e_since = time.time()



        ###################

        # train the model #

        ###################

        model.train()  # prep model for training

        if scheduler is not None:

            scheduler.step()  # step up scheduler



        for images, labels in train_loader:

            # Move input and label tensors to the default device

            images, labels = images.to(device), labels.to(device)

            # clear the gradients of all optimized variables

            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model

            log_ps = model(images)

            # calculate the loss

            loss = criterion(log_ps, labels)

            # backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()

            # perform a single optimization step (parameter update)

            optimizer.step()

            # update running training loss

            train_loss += loss.item() * images.size(0)

            # Write on tensorbroad

            tb.add_scalar("Loss", loss.item(), global_epoch)



        ######################

        # validate the model #

        ######################

        print("\t\tGoing for validation")

        model.eval()  # prep model for evaluation

        for data, target in test_loader:

            # Move input and label tensors to the default device

            data, target = data.to(device), target.to(device)

            # forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # calculate the loss

            loss_p = criterion(output, target)

            # update running validation loss

            valid_loss += loss_p.item() * data.size(0)

            # calculate accuracy

            proba = torch.exp(output)

            top_p, top_class = proba.topk(1, dim=1)

            equals = top_class == target.view(*top_class.shape)

            # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()



            _, predicted = torch.max(output.data, 1)

            total += target.size(0)

            correct += (predicted == target).sum().item()

            # Write on tensorbroad

            tb.add_scalar("Loss", loss.item(), global_epoch)



        # print training/validation statistics

        # calculate average loss over an epoch

        train_loss = train_loss / len(train_loader.dataset)

        valid_loss = valid_loss / len(test_loader.dataset)



        # calculate train loss and running loss

        train_loss_data.append(train_loss * 100)

        valid_loss_data.append(valid_loss * 100)



        accuracy = (correct / total) * 100



        print("\tTrain loss:{:.6f}..".format(train_loss),

              "\tValid Loss:{:.6f}..".format(valid_loss),

              "\tAccuracy: {:.4f}".format(accuracy))

        

        tb.add_scalar("Global Loss", train_loss, global_epoch)

        tb.add_scalar("Global Loss", valid_loss, global_epoch)

        tb.add_scalar("Global Accuracy Loss", accuracy, global_epoch)

        tb.close()



        # Update global epoch

        global_epoch += 1



        # save model if validation loss has decreased

        if valid_loss <= valid_loss_min:

            print('\tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

                valid_loss_min,

                valid_loss))

            torch.save(model.state_dict(), name)

            valid_loss_min = valid_loss

            # save to google drive

            if path is not None:

                torch.save(model.state_dict(), path)



        # Time take for one epoch

        time_elapsed = time.time() - e_since

        print('\tEpoch:{} completed in {:.0f}m {:.0f}s'.format(

            epoch + 1, time_elapsed // 60, time_elapsed % 60))



    # compare total time

    time_elapsed = time.time() - since

    print('Training completed in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))



    # load best model

    #model = load_latest_model(model, name)

    

    # close writer

    #tb.close()



    # return the model

    return [model, train_loss_data, valid_loss_data, global_epoch]
model, train_loss, test_loss, global_epoch = train(

    model, train_loader, test_loader, epoch, optimizer, criterion, scheduler, global_epoch=global_epoch)
Helper.check_overfitted(train_loss, test_loss)
Helper.test(model, test_loader)
Helper.test_per_class(model, test_loader, criterion, classes)
from PIL import Image



def test(file):

    file = Image.open(file).convert('L')

    img = test_transform(file).unsqueeze(0)

    with torch.no_grad():

        out = model(img.to(device))

        proba = torch.exp(out)

        top_p, top_class = proba.topk(1, dim=1)

        print(f"Predicted Label: {top_class.item()}")

        plt.imshow(np.array(file))

        plt.show()
from PIL import Image

from matplotlib import pyplot as plt

data_dir = '../input/testing-d'

name = os.listdir(data_dir)[4]

file = data_dir+"/"+name

print(file)



test(file)
data_dir = '../input/testing-c'

name = os.listdir(data_dir)[4]

file = data_dir+"/"+name

print(file)



test(file)
data_dir = '../input/testing-a'

name = os.listdir(data_dir)[4]

file = data_dir+"/"+name

print(file)



test(file)
data_dir = '../input/testing-b'

name = os.listdir(data_dir)[4]

file = data_dir+"/"+name

print(file)



test(file)
data_dir = '../input/testing-b'

name = os.listdir(data_dir)[15]

file = data_dir+"/"+name

print(file)



test(file)
!wget https://i.imgur.com/jJz1GUB.png
test('jJz1GUB.png')
!wget https://i.imgur.com/Pd5P7C3.png
test('Pd5P7C3.png')
import os

import shutil



def remove_contents(path):

    for c in os.listdir(path):

        full_path = os.path.join(path, c)

        if os.path.isfile(full_path):

            os.remove(full_path)

        else:

            shutil.rmtree(full_path)
remove_contents(path)
os.listdir(path)
def save_check_point(model, epoch, classes, optimizer, scheduler=None,

                     path=None, name='model.pt'):

    try:

        classifier = model.classifier

    except AttributeError:

        classifier = model.fc

        

    

    print(classifier)



    checkpoint = {

        'class_to_name': classes,

        'epochs': epoch,

        'classifier': classifier,

        'state_dict': model.state_dict(),

        'optimizer_state_dict': optimizer.state_dict()

    }



    if scheduler is not None:

        checkpoint['scheduler_state_dict'] = scheduler.state_dict()



    if path is None:

        d = name

    else:

        d = path + "/" + name



    torch.save(checkpoint, d)

    print(f"Model saved at {d}")
save_check_point(model, epoch, classes, optimizer, scheduler=scheduler,

                     path=None, name='saved_model.pt')