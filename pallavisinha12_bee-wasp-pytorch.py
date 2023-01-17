import numpy as np

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

import torchvision

import torchvision.datasets as datasets

import torchvision.models as models

import torch.optim as optim

import torchvision.transforms as transforms

import pandas as pd

import cv2

from torch.utils.data import Dataset, DataLoader, random_split

import copy

import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
data = pd.read_csv("../input/bee-vs-wasp/kaggle_bee_vs_wasp/labels.csv")

data.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(data["label"])

data["label"] = le.transform(data["label"])
for i in data.index:

    data["path"].iloc[i] = data["path"].iloc[i].replace("\\", "/")
def split(dt):

    idx = []

    a = pd.DataFrame()

    b = pd.DataFrame()

    for i in data.index:

        if data["is_validation"].iloc[i] == 1:

            a = a.append(dt.iloc[i])

            idx.append(i)

        if data["is_final_validation"].iloc[i] == 1:

            b = b.append(dt.iloc[i])

            idx.append(i)

    dt = dt.drop(dt.index[idx])

    a = a.reset_index()

    b = b.reset_index()

    dt = dt.reset_index()

    return dt,a,b

        
train_df, val_df, test_df = split(data)

print("Length of train dataset: ", len(train_df))

print("Length of validation dataset: " ,len(val_df))

print("Length of test dataset: ", len(test_df))
train_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])
val_df.label = val_df.label.astype(np.int64)

test_df.label = test_df.label.astype(np.int64)
class Bee_Wasp(Dataset):

    def __init__(self, df:pd.DataFrame, imgdir:str,

                 transforms=None):

        self.df = df

        self.imgdir = imgdir

        self.transforms = transforms

    

    def __getitem__(self, index):

        im_path = os.path.join(self.imgdir, self.df.iloc[index]["path"])

        x = cv2.imread(im_path)

        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        x = cv2.resize(x, (224, 224))



        if self.transforms:

            x = self.transforms(x)

            

        y = self.df.iloc[index]["label"]

        return x, y

    

    def __len__(self):

        return len(self.df)
train_data = Bee_Wasp(df=train_df,

                        imgdir="../input/bee-vs-wasp/kaggle_bee_vs_wasp",

                        transforms=train_transform)



val_data = Bee_Wasp(df=val_df,

                      imgdir="../input/bee-vs-wasp/kaggle_bee_vs_wasp",

                      transforms=test_transform)



test_data = Bee_Wasp(df=test_df,

                       imgdir="../input/bee-vs-wasp/kaggle_bee_vs_wasp",

                       transforms=test_transform)
train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=32, num_workers=4)

val_loader = DataLoader(dataset = val_data, shuffle = True, batch_size = 32, num_workers=4)

test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size=32, num_workers=4)
import os

dataiter = iter(train_loader)

images, labels = dataiter.next()

print(images.shape)

print(labels.shape)
model = torchvision.models.resnet50(pretrained=True, progress=True)
for p in model.parameters():

    p.requires_grad = False
model.fc = nn.Sequential(

    nn.Linear(2048, 1024),

    nn.ReLU(),

    nn.Linear(1024,4),

    nn.LogSoftmax(dim=1)

)
for param in model.parameters():

    if param.requires_grad:

        print(param.shape)
model = model.to(device)

opt = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.NLLLoss()
def evaluation(dataloader):

    total, correct = 0,0

    model.eval()

    for data in dataloader:

        inputs, labels = data

        inputs =inputs.to(device)

        labels = labels.to(device)

        outputs = model(inputs)

        _ , pred = torch.max(outputs.data, 1 )

        total += labels.size(0)

        correct += (pred == labels).sum().item()

    return 100*correct/total
batch_size = 32

import copy

loss_epoch_arr = []

max_epochs = 6

min_loss = 1000

n_iters = np.ceil(50000/batch_size)

for epoch in range(max_epochs):

    for i,data in enumerate(train_loader):

        inputs,labels = data

        inputs = inputs.to(device)

        labels =  labels.to(device)

        opt.zero_grad()

        model.train()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        loss.backward()

        opt.step()

        if min_loss>loss.item():

            min_loss = loss.item()

            best_model = copy.deepcopy(model.state_dict())

            print('Min loss %0.2f' % min_loss)

            del inputs, labels, outputs

            torch.cuda.empty_cache()

    loss_epoch_arr.append(loss.item())

    model.eval()

    print("Epoch %d/%d, Train acc: %0.2f, Val acc: %0.2f" %(epoch, max_epochs, evaluation(train_loader), evaluation(val_loader)))
plt.plot(loss_epoch_arr)

plt.show()
model.load_state_dict(best_model)

print(evaluation(test_loader))
classes = ['bee', 'wasp', 'other_insect', 'noninsect']

dataiter = iter(test_loader)

images, labels = dataiter.next()

def imshow(img):

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1,2,0)))

    plt.show()

imshow(torchvision.utils.make_grid(images[:1]))

print("Ground_Truth - ")

print(classes[labels[0]])

images = images.to(device)

outputs = model(images)

max_values, pred_class = torch.max(outputs.data, 1)

print("Predicted_class - ")

print(classes[pred_class[0]])