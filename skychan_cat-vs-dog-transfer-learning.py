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
import torch

from torch import nn

from torch.autograd import Variable

import torchvision

from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

import time

from sklearn.model_selection import train_test_split

from PIL import Image
%matplotlib inline
data_dir = "/kaggle/input/train/train"
class myDataset(torch.utils.data.Dataset):

    def __init__(self, prefix, files, transform, img_loader):

        self.imgs = files

        self.transform = transform

        self.loader = img_loader

        self.prefix = prefix

    

    def __len__(self):

        return len(self.imgs)

    

    def __getitem__(self, idx):

        img = self.imgs[idx]

        label = 0 if 'cat' in img else 1

        img = self.loader(f"{self.prefix}/{img}")

        return self.transform(img), label
img_train, img_test = train_test_split(os.listdir(data_dir), test_size=0.1, random_state=42)
transform = transforms.Compose([

    transforms.Resize([224, 224]),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])
train_data = myDataset(data_dir, img_train, transform, lambda file: Image.open(file).convert('RGB'))

test_data = myDataset(data_dir, img_test, transform, lambda file: Image.open(file).convert('RGB'))
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
x_exp, y_exp = next(iter(test_data_loader))
y_exp
img = torchvision.utils.make_grid(x_exp)

img = img.numpy().transpose([1,2,0])

plt.imshow(img)

plt.show()
model = models.vgg16(pretrained=True)
print(model)
# 模型中的参数不需要更新

for parma in model.parameters():

    parma.requires_grad = False
model.classifier = nn.Sequential(

    nn.Linear(25088, 4096),

    nn.ReLU(),

    nn.Dropout(),

    nn.Linear(4096, 4096),

    nn.ReLU(),

    nn.Dropout(),

    nn.Linear(4096, 2)

)
if torch.cuda.is_available():

    model = model.cuda()
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-5)
from tqdm import tqdm_notebook as tqdm
epoch_n = 5

time_s = time.time()



for epoch in range(epoch_n):

    print(f"Epoch {epoch+1}/{epoch_n}")

    print("-"*10)

    

    for phase in ["train", "valid"]:

        if phase == 'train':

            print("Training...")

            model.train(True)

            dataloader = train_data_loader

            img_datasets = train_data

        else:

            print("Validing...")

            model.eval()

            dataloader = test_data_loader

            img_datasets = test_data

        

        running_loss = 0.0

        running_acc = 0.0

        

        for batch, data in tqdm(enumerate(dataloader, 1), total=len(dataloader)):

            X, y = data

            X, y = Variable(X.cuda()), Variable(y.cuda())

            y_pred = model(X)

            _, pred = torch.max(y_pred.data, 1)

    

            optimizer.zero_grad()

            batch_loss = loss(y_pred, y)



            if phase == "train":

                batch_loss.backward()

                optimizer.step()

            

            running_loss += batch_loss.data

            running_acc += torch.sum(pred == y.data)

            

            if batch%100 == 0 and phase == "train":

                print(f"Batch {batch}, Train Loss {running_loss/(batch):.4f}, Train Acc {100*running_acc/(64*(batch))}")

        

        epoch_loss = running_loss*64/len(img_datasets)

        epoch_acc = 100*running_acc/(len(img_datasets) + 0.0)

        

        print(f"{phase} Loss {epoch_loss:.4f} Acc {epoch_acc:.4f}")

time_e = time.time()

print(f"Spend {(time_e-time_s)/60:.3f} mins")
torch.save(model, './transfer_vgg16.pth')
class resultDataset(torch.utils.data.Dataset):

    def __init__(self, prefix, transform, img_loader):

        self.imgs = os.listdir(prefix)

        self.transform = transform

        self.loader = img_loader

        self.prefix = prefix

    

    def __len__(self):

        return len(self.imgs)

    

    def __getitem__(self, idx):

        name = self.imgs[idx]

        img = self.loader(f"{self.prefix}/{name}")

        return self.transform(img), name
T_data = resultDataset("/kaggle/input/test1/test1", transform, lambda file: Image.open(file).convert('RGB'))
T_dataloader = torch.utils.data.DataLoader(T_data)
model.eval()

 

result = {

    name[0]: torch.max(model(Variable(X_test.cuda())) ,1)[1].data.cpu().numpy() 

    for X_test, name in tqdm(T_dataloader, total=len(T_data))

}
res_sample = np.random.choice([k for k in result.keys()], 18, replace=False)
result[res_sample[0]]
model.eval()

y_pred = model(x_exp.cuda())
pred = torch.max(y_pred,1)[1].data.cpu().numpy()
img = torchvision.utils.make_grid(x_exp)

img = img.numpy().transpose([1,2,0])

plt.imshow(img)

plt.title("Cat" if pred == 0 else "dog")

plt.show()
from keras.preprocessing.image import load_img

prefix = "/kaggle/input/test1/test1"

IMAGE_SIZE = (224, 224)
plt.figure(figsize=(12, 24))

for i, name in enumerate(res_sample, 1):

    img = load_img(f"{prefix}/{name}", target_size=IMAGE_SIZE)

    plt.subplot(6, 3, i)

    plt.imshow(img)

    plt.xlabel("cat" if result[name] == 0 else "dog")

plt.tight_layout()

plt.show()
result_df = pd.DataFrame.from_dict(result, orient='index', columns=["label"])
result_df['id'] = result_df.index.str.split(".").str[0].astype(int)
result_df = result_df[['id', 'label']]
result_df = result_df.sort_values(by=["id"])
result_df.to_csv('submission.csv', index=False)
model = models.resnet50(pretrained=True)
for parma in model.parameters():

    parma.requires_grad = False
model.fc = nn.Linear(2048, 2)
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-5)
model = model.cuda()
epoch_n = 5

time_s = time.time()



for epoch in range(epoch_n):

    print(f"Epoch {epoch+1}/{epoch_n}")

    print("-"*10)

    

    for phase in ["train", "valid"]:

        if phase == 'train':

            print("Training...")

            model.train(True)

            dataloader = train_data_loader

            img_datasets = train_data

        else:

            print("Validing...")

            model.eval()

            dataloader = test_data_loader

            img_datasets = test_data

        

        running_loss = 0.0

        running_acc = 0.0

        

        for batch, data in tqdm(enumerate(dataloader, 1), total=len(dataloader)):

            X, y = data

            X, y = Variable(X.cuda()), Variable(y.cuda())

            y_pred = model(X)

            _, pred = torch.max(y_pred.data, 1)

    

            optimizer.zero_grad()

            batch_loss = loss(y_pred, y)



            if phase == "train":

                batch_loss.backward()

                optimizer.step()

            

            running_loss += batch_loss.data

            running_acc += torch.sum(pred == y.data)

            

            if batch%100 == 0 and phase == "train":

                print(f"Batch {batch}, Train Loss {running_loss/(batch):.4f}, Train Acc {100*running_acc/(64*(batch))}")

        

        epoch_loss = running_loss*64/len(img_datasets)

        epoch_acc = 100*running_acc/(len(img_datasets) + 0.0)

        

        print(f"{phase} Loss {epoch_loss:.4f} Acc {epoch_acc:.4f}")

time_e = time.time()

print(f"Spend {(time_e-time_s)/60:.3f} mins")
torch.save(model, './transfer_resnet50.pth')
model.eval()

result_resnet = {

    name[0]: torch.max(model(Variable(X_test.cuda())) ,1)[1].data.cpu().numpy() 

    for X_test, name in tqdm(T_dataloader, total=len(T_data))

}
result_df_r = pd.DataFrame.from_dict(result_resnet, orient='index', columns=["label"])

result_df_r['id'] = result_df_r.index.str.split(".").str[0].astype(int)

result_df_r = result_df_r[['id', 'label']]

result_df_r = result_df_r.sort_values(by=["id"])
a = result_df[result_df_r.label != result_df.label]
b = result_df_r[result_df_r.label != result_df.label]
from keras.preprocessing.image import load_img

prefix = "/kaggle/input/test1/test1"

IMAGE_SIZE = (224, 224)
plt.figure(figsize=(12, 24))

for i, name in enumerate(a.index[105:123], 1):

    img = load_img(f"{prefix}/{name}", target_size=IMAGE_SIZE)

    plt.subplot(6, 3, i)

    plt.imshow(img)

    plt.xlabel("cat" if a.at[name, 'label'] == 0 else "dog")

plt.tight_layout()

plt.show()