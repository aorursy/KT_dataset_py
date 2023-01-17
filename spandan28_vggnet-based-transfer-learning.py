# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torch.functional as F

from torch.optim import Adam

from torchvision import models, datasets,transforms

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from PIL import Image

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# Any results you write to the current directory are saved as output.
train_data=pd.read_csv('../input/aerial-cactus-identification/train.csv')

sample = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")

train_data.head(5)
train_data["has_cactus"].value_counts().plot(kind="pie")

train_dir = "train/train/"

val_dir = "train/train"

train_data.head(5)

train_data.values[:,1]
class dataset_(torch.utils.data.Dataset):

    def __init__(self,labels,data_directory,transform):

        super().__init__()

        #characterizes a dataset for Pytorch        

        self.list_id=labels.values[:,0]

        self.labels=labels.values[:,1]

        self.data_dir=data_directory

        self.transform=transform

    

    def __len__(self):

        # Denotes the tota number of samples

        return len(self.list_id)

    

    def __getitem__(self,index):

        name=self.list_id[index]

        img=Image.open('../input/aerial-cactus-identification/{}/{}'.format(self.data_dir,name))

        img=self.transform(img)

        return img,torch.tensor(self.labels[index],dtype=torch.float32)
image_transform={"train":transforms.Compose([transforms.RandomRotation(degrees=0),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),

                "test": transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])}
test_dir = "test/test/"

test_df = dataset_(sample, test_dir, image_transform["test"])

test_data_loader = torch.utils.data.DataLoader(test_df, batch_size=64,shuffle=False)
train_set, validation_set = train_test_split(train_data,stratify=train_data.has_cactus.values, test_size=0.2)
train_set["has_cactus"].value_counts().plot(kind="pie")

validation_set["has_cactus"].value_counts().plot(kind="pie")
train_data_set = dataset_(train_data, train_dir,image_transform["train"])

validation_data_set = dataset_(train_data,val_dir, image_transform["test"])
train_data_frame = torch.utils.data.DataLoader(dataset=train_data_set, batch_size=32, shuffle=True)

valid_data_frame = torch.utils.data.DataLoader(dataset=validation_data_set, batch_size=32)
vgg16 = models.vgg16(pretrained=True)

for param in vgg16.parameters():

    param.requires_grad = False

from collections import OrderedDict



classifier = torch.nn.Sequential(OrderedDict([

    ('fc1',torch.nn.Linear(512*7*7,1024)),

    ('relu',torch.nn.ReLU()),

    ('fc2',torch.nn.Linear(1024,512)),

    ('relu', torch.nn.ReLU()),

    ('fc3',torch.nn.Linear(512,2)),

    ('output',torch.nn.LogSoftmax(dim=1))

]))

vgg16.classifier = classifier
optimizer = Adam(vgg16.classifier.parameters(), lr =0.001)

criterion = torch.nn.NLLLoss()
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
ex_data, ex_label = next(iter(train_data_frame))

from matplotlib import pyplot as plt

plt.imshow(ex_data[1].view(224,224,-1))
total_epochs = 10

vgg16.to(device)

train_error_array = []

test_error_array = []

for epoch in range(total_epochs):

    running_train_loss = 0.0

    running_accuracy = 0.0

    vgg16.train()

    for step, (data, train_label) in enumerate(train_data_frame):

        data, train_label = data.to(device), train_label.to(device)

        output_pred = vgg16.forward(data)

        train_ps = torch.exp(output_pred)

        train_prob, train_cls = train_ps.topk(1, dim=1)

        train_equals = (train_cls == train_label.long().view(*train_cls.shape))

        running_accuracy += torch.mean(train_equals.type(torch.FloatTensor))

        optimizer.zero_grad()

        train_loss = criterion(output_pred, train_label.long())

        running_train_loss += train_loss.item()

        train_loss.backward()

        optimizer.step()        

        

    else:

        

        test_accuracy= 0.0

        testloss = 0.0

        vgg16.eval()

        with torch.no_grad():

            for valid_data, valid_label in valid_data_frame:

                valid_data, valid_label = valid_data.to(device), valid_label.to(device)

                test_pred = vgg16.forward(valid_data)

                test_loss = criterion(test_pred, valid_label.long())

                testloss += test_loss.item()

                out = torch.exp(test_pred)

                prob, cls = out.topk(1, dim=1)

                equals = (cls==valid_label.long().view(*cls.shape))

                test_accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_error_array.append(running_train_loss/len(train_data_frame))

        test_error_array.append(testloss/len(valid_data_frame))

        print(f"EPoch: {epoch+1} >> ")

        print(f"training loss >> {running_train_loss/len(train_data_frame)}")

        print(f"test loss >> {testloss/len(valid_data_frame)}")

        print(f"test accuracy >> {test_accuracy/len(valid_data_frame)}")

        print("\n")

        



            



                        

            
plt.plot(train_error_array, "b")

plt.plot(test_error_array,'r')

plt.show()
output = []

with torch.no_grad():

    vgg16.eval()

    for test_data, _ in test_data_loader:

        test_data = test_data.to(device)

        test_output = vgg16.forward(test_data)

        test_prob = torch.exp(test_output)



        output += list(test_prob[:,1].cpu().data.numpy())

        
sample_output = {"id": sample["id"], "has_cactus":output}

sample_output_dataframe = pd.DataFrame(sample_output)

sample_output_dataframe.head(10)
sample_output_dataframe.to_csv("submission.csv",index=False)