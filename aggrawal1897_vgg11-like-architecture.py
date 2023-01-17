# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import cv2

import matplotlib.pyplot as plt

import random

print(os.listdir("../input"))



import torch

import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

import torch.optim as optim

from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import accuracy_score

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

torch.manual_seed(11)

np.random.seed(13)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

train_data.head()



train, test = train_test_split(train_data, test_size=0.1)

print("Training size : " + str(len(train)))

print("Testing size : " + str(len(test)))

ROOT_DIR = "../input/train_images/"
class MyDataset(Dataset):

    def __init__(self, root_dir, csv_file, transforms = None):

        self.root_dir = root_dir

        self.transform = transforms

        self.csvfile = csv_file

        

    def __len__(self):

        return len(self.csvfile)

    

    def __getitem__(self, idx):

        label = self.csvfile.iloc[idx, 1]

        label = to_categorical(label, num_classes=5)

        

        img = cv2.imread(self.root_dir + self.csvfile.iloc[idx, 0] + ".png")

        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)

        if self.transform:

            img = self.transform(img)

        

        return (img, torch.FloatTensor(label))
training_data = MyDataset(root_dir=ROOT_DIR, csv_file = train,

                                transforms = transforms.Compose([

                                    transforms.ToPILImage(),

                                    transforms.RandomApply([

                                        transforms.RandomAffine(10.0, shear=15.0),

                                        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)

                                    ]),

                                    transforms.ToTensor()

                                ]))



testing_data = MyDataset(root_dir=ROOT_DIR, csv_file = test,

                                transforms = transforms.Compose([

                                    transforms.ToPILImage(),

                                    transforms.RandomApply([

                                        transforms.RandomAffine(19.0, shear=15.0),

                                        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)

                                    ]),

                                    transforms.ToTensor()

                                ]))



nloader = DataLoader(training_data, batch_size=32)

ploader = DataLoader(testing_data, batch_size=32)
class Flatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)



class VGG11(nn.Module):

    def __init__(self, inchannels = 3):

        super(VGG11, self).__init__()

        self.network = nn.Sequential(

            nn.Conv2d(inchannels, 64, 7, stride=2, padding = 1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Conv2d(64, 128, 3, stride=2, padding = 1),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Conv2d(128, 256, 3, stride=1, padding = 1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=2, padding = 1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=2, padding = 1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=2, padding = 1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=2, padding = 1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=2, padding = 1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=2, padding = 1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.AvgPool2d(2),

            Flatten(),

            nn.Linear(256, 128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(128, 5),

            nn.Softmax(dim=1)

        )



    def forward(self, x):

        return self.network(x)
model = VGG11()

model.cuda()



opt = optim.Adam(model.parameters())

loss = nn.BCELoss()



train_loss_epoch = []

train_acc_epoch = []



loss_epoch = []

acc_epoch = []
for j in tqdm(range(10), desc="Epoch : "):

    losses = 0

    accuracy = 0

    with tqdm(total = len(nloader)) as ibar:

        model = model.train()

        for i, batch in enumerate(nloader):

            pred = model(batch[0].cuda())

            l = loss(pred, batch[1].cuda())

            l.backward()

            opt.step()

            accuracy += accuracy_score(np.argmax(batch[1].numpy(), axis=1), np.argmax(pred.detach().cpu().numpy(), axis=1))

            losses += l.item()

            ibar.set_description("TRAINING Loss:{:.4f},Acc:{:.3f}".format(

                losses/(i+1), (accuracy/(i+1)) * 100.0

            ))

            ibar.update(1)

        train_loss_epoch.append(losses/(i+1))

        train_acc_epoch.append(accuracy/(i+1))

        

    losses = 0

    accuracy = 0

    with tqdm(total = len(ploader)) as kbar:

        model = model.eval()

        for k, batch in enumerate(ploader):

            pred = model(batch[0].cuda())

            l = loss(pred, batch[1].cuda())

            accuracy += accuracy_score(np.argmax(batch[1].numpy(), axis=1), np.argmax(pred.detach().cpu().numpy(), axis=1))

            losses += l.item()

            kbar.set_description("TESTING Loss:{:.4f},Acc:{:.3f}".format(

                losses/(k+1), (accuracy/(k+1)) * 100.0

            ))

            kbar.update(1)

        loss_epoch.append(losses/(k+1))

        acc_epoch.append(accuracy/(k+1))
plt.rcParams["figure.figsize"] = (15,10)



plt.subplot(221)

plt.title("Training Accuracy")

plt.xlabel("Iteration")

plt.ylabel("Accuracy")

plt.plot(list(range(len(train_acc_epoch))), train_acc_epoch)



plt.subplot(222)

plt.title("Training Loss")

plt.xlabel("Iteration")

plt.ylabel("Loss")

plt.plot(list(range(len(train_loss_epoch))), train_loss_epoch)



plt.subplot(223)

plt.title("Testing Accuracy")

plt.xlabel("Iteration")

plt.ylabel("Accuracy")

plt.plot(list(range(len(acc_epoch))), acc_epoch)



plt.subplot(224)

plt.title("Testing Loss")

plt.xlabel("Iteration")

plt.ylabel("Loss")

plt.plot(list(range(len(loss_epoch))), loss_epoch)

plt.show()
test = os.listdir("../input/test_images/")



model = model.eval()

ans = {}

for i in range(len(test)):

    img = cv2.imread("../input/test_images/" + test[i])

    img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_NEAREST)

    img = img.transpose((2, 0, 1))

    img = np.expand_dims(img, axis=0) / 255.0

    img = torch.FloatTensor(img)

    pred = model(img.cuda())

    ans[test[i]] = np.argmax(pred.detach().cpu().numpy()[0])
data = {'id_code' : list(ans.keys()), 'diagnosis' : list(ans.values())}

data = pd.DataFrame.from_dict(data)
data.to_csv("submission.csv", index=False)