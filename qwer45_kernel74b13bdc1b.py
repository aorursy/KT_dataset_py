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
import torch

import torchvision

from torchvision import datasets,transforms, models

import os

import numpy as np

import matplotlib.pyplot as plt

from torch.autograd import Variable

import time



path = "/kaggle/input/cat_dog/"

transform = transforms.Compose([transforms.CenterCrop(224),

                                transforms.ToTensor(),

                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

data_image = {x:datasets.ImageFolder(root = os.path.join(path,x),

                                     transform = transform)

              for x in ["train", "val"]}



data_loader_image = {x:torch.utils.data.DataLoader(dataset=data_image[x],

                                                batch_size = 4,

                                                shuffle = True)

                     for x in ["train", "val"]}

use_gpu = torch.cuda.is_available()



print(use_gpu)

classes = data_image["train"].classes

classes_index = data_image["train"].class_to_idx

print(classes)

print(classes_index)

print("train data set:", len(data_image["train"]))

print("val data set:", len(data_image["val"]))

X_train,y_train = next(iter(data_loader_image["train"]))

mean = [0.5, 0.5, 0.5]

std = [0.5, 0.5, 0.5]

img = torchvision.utils.make_grid(X_train)

img = img.numpy().transpose((1,2,0))

img = img*std + mean



print([classes[i] for i in y_train])

plt.imshow(img)

model = models.vgg16(pretrained = True)

print(model)

for parma in model.parameters():

    parma.requires_grad = False



model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),

                                       torch.nn.ReLU(),

                                       torch.nn.Dropout(p=0.5),

                                       torch.nn.Linear(4096, 4096),

                                       torch.nn.ReLU(),

                                       torch.nn.Dropout(p=0.5),

                                       torch.nn.Linear(4096, 2))



#for index, parma in enumerate(model.classifier.parameters()):

    #if index == 6:

     #  parma.requires_grad = True



if use_gpu:

    model = model.cuda()



cost = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.classifier.parameters(), lr=1e-8, momentum=0.9)

n_epochs = 1

for epoch in range(n_epochs):

    since = time.time()

    print("Epoch{}/{}".format(epoch, n_epochs))

    print("-" * 10)

    for param in ["train", "val"]:

        if param == "train":

            model.train = True

        else:

            model.train = False



        running_loss = 0.0

        running_correct = 0

        batch = 0

        for data in data_loader_image[param]:

            batch += 1



            X, y = data

            if use_gpu:

                X, y - Variable(X.cuda()), Variable(y.cuda())

            else:

                X, y = Variable(X), Variable(y)



            optimizer.zero_grad()

            y_pred = model(X)

            _, pred = torch.max(y_pred.data, 1)



            loss = cost(y_pred, y)

            if param == "train":

                loss.backward()

                optimizer.step()

            running_loss += loss.item()

            running_correct += torch.sum(pred == y.data)

            if param == "train" and batch<=10:

                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(

                    batch, running_loss / (4 * batch), 100 * running_correct / (4 * batch)))

    data_test_img = datasets.ImageFolder(root="/kaggle/input/cat_dog/test/", transform=transform)

    print(data_test_img)

    data_loader_test_img = torch.utils.data.DataLoader(dataset=data_test_img,

                                                           batch_size=32)

    image, label = next(iter(data_loader_test_img))

    images = Variable(image)

    y_pred = model(images)

    _, pred = torch.max(y_pred.data, 1)

    print(pred)

    img = torchvision.utils.make_grid(image)

    img = img.numpy().transpose(1, 2, 0)

    mean = [0.5, 0.5, 0.5]

    std = [0.5, 0.5, 0.5]

    img = img * std + mean

    print("Pred Label:", [classes[i] for i in pred])

    print(label)

    plt.imshow(img)

    plt.show()