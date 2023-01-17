import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch, torchvision

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from tqdm.notebook import tqdm

from torchvision import transforms,models

import gc



# Ignore warnings

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1 = pd.read_parquet("/kaggle/input/bengaliai-cv19/train_image_data_0.parquet")

df2 = pd.read_parquet("/kaggle/input/bengaliai-cv19/train_image_data_1.parquet")
df3 = pd.read_parquet("/kaggle/input/bengaliai-cv19/train_image_data_2.parquet")

df4 = pd.read_parquet("/kaggle/input/bengaliai-cv19/train_image_data_3.parquet")
full_data = pd.concat([df1, df2, df3, df4], ignore_index = True)
train_csv = pd.read_csv("/kaggle/input/bengaliai-cv19/train.csv")

del df1, df2, df3, df4

gc.collect()
from PIL import Image

import matplotlib.pyplot as plt

img_0 = full_data.iloc[0, 1:].values.reshape(137,236)

img = Image.fromarray(img_0.astype('uint8'), 'L')

plt.figure(figsize=(4,4))

plt.axis('off')

plt.imshow(img)
class GraphemeDataset(Dataset):

    def __init__(self, csv_file, df, transform = None, train = True):

        self.label = csv_file

        self.df = df

        self.transform = transform

        self.train = train

        

    def __len__(self):

        return len(self.label)

    

    def __getitem__(self, indx):

        img = self.df.iloc[indx,1:].values.reshape(137,236)

        img = Image.fromarray(img.astype('uint8'), 'L')

        

        if self.transform:

            img = self.transform(img)

            

        if self.train == True:

            label1 = self.label.iloc[indx, 1]

            label2 = self.label.iloc[indx, 2]

            label3 = self.label.iloc[indx, 3]

            return img, label1, label2, label3

        else:

            img_id = self.df.iloc[indx, 0]

            return img, img_id 

    
transform = transforms.Compose([transforms.Resize((224, 224)),

                                transforms.Grayscale(3),

                                transforms.ToTensor(), 

                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  
train_data = GraphemeDataset(train_csv, full_data, transform = transform, train = True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
for data in train_loader: 

    img, label1, label2, label3 = data

    print(img[0].shape)

    print(label1[0].shape)

    img = np. transpose(img[0], (1,2,0))

    img= np.squeeze(img)

    plt.imshow(img)

    break
model_gr = torchvision.models.resnet34()

model_vd = torchvision.models.resnet34()

model_cd = torchvision.models.resnet34()
model_gr.fc = nn.Linear(512, 168)

model_vd.fc = nn.Linear(512, 11)

model_cd.fc = nn.Linear(512, 7)
device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_gr.to(device)

model_vd.to(device)

model_cd.to(device)

optimizer_gr = torch.optim.Adam (model_gr.parameters(), lr = 0.01)

optimizer_vd = torch.optim.Adam (model_vd.parameters(), lr = 0.01)

optimizer_cd = torch.optim.Adam (model_cd.parameters(), lr = 0.01)

Criterion_gr = nn.CrossEntropyLoss()

Criterion_vd = nn.CrossEntropyLoss()

Criterion_cd = nn.CrossEntropyLoss()
epochs = 20

model_gr.train()

model_vd.train()

model_cd.train()

losses = []

accuracy = []

for epoch in tqdm(range(epochs)):

    print("epochs {}/{}".format(epoch+1, epochs))

    acc = 0.0

    for idx , data in tqdm(enumerate(train_loader)):

        img, label1, label2, label3 = data

        img, label1, label2, label3 = img.to(device), label1.to(device), label2.to(device), label3.to(device)

        optimizer_gr.zero_grad()

        optimizer_vd.zero_grad()

        optimizer_cd.zero_grad()

        output1 = model_gr(img)

        output2 = model_vd(img)

        output3 = model_cd(img)

        loss1= Criterion_gr(output1, label1) 

        loss2= Criterion_vd(output2, label2)

        loss3= Criterion_cd(output3, label3)

        total_loss = loss1 + loss2 + loss3

        acc += (output1.argmax(1) == label1).float().mean()

        acc += (output2.argmax(1) == label2).float().mean()

        acc += (output3.argmax(1) == label3).float().mean()

        loss1.backward()

        loss2.backward()

        loss3.backward()

        optimizer_gr.step()

        optimizer_vd.step()

        optimizer_cd.step()

        del img, label1, label2, label3

        torch.cuda.empty_cache()

        

    losses.append(total_loss)

    accuracy.append(acc/len(trainloader)*3)

    print('acc: {:.3f}%'.format(acc/len(train_loader)*3))

    print('loss:{:.3f}%'.format(total_loss))

torch.save(model_gr.state_dict, 'gr_resnet34_20epochs_saved_weights.pth')

torch.save(model_vd.state_dict, 'vd_resnet34_20epochs_saved_weights.pth')

torch.save(model_cd.state_dict, 'cd_resnet34_20epochs_saved_weights.pth')

        
fig, ax = plt.subplots(1, 2, figsize = (15,5))

ax[0].plot(losses)

ax[0].set_title("Loss")

ax[1].plot(accuracy)

ax[1].set_title("Accuracy")
df1 = pd.read_parquet("/kaggle/input/bengaliai-cv19/test_image_data_0.parquet")

df2 = pd.read_parquet("/kaggle/input/bengaliai-cv19/test_image_data_1.parquet")

df3 = pd.read_parquet("/kaggle/input/bengaliai-cv19/test_image_data_2.parquet")

df4 = pd.read_parquet("/kaggle/input/bengaliai-cv19/test_image_data_3.parquet")

full_test_data = pd.concat([df1, df2, df3, df4], ignore_index = True)

test_csv= pd.read_csv("/kaggle/input/bengaliai-cv19/test.csv")
test_data = GraphemeDataset(test_csv, full_test_data, transform = transform, train = False)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)
model_gr.load_state_dict(torch.load("gr_resnet34_20epochs_saved_weights.pth"))

model_vd.load_state_dict(torch.load("vd_resnet34_20epochs_saved_weights.pth"))

model_cd.load_state_dict(torch.load("cd_resnet34_20epochs_saved_weights.pth"))

model_gr.to(device)

model_vd.to(device)

model_cd.to(device)

model_gr.eval()

model_vd.eval()

model_cd.eval()
predictions = []

for img, img_id in tqdm(test_loader):

    img = img.to(device)

    pred1 = model_gr(img)

    pred2 = model_vd(img)

    pred3 = model_cd(img)

    _, ind1 = torch.max(pred1, 1)

    _, ind2 = torch.max(pred2, 1)

    _, ind3 = torch.max(pred3, 1)

    output1 = ind1.squeeze().cpu().numpy()

    output2 = ind2.squeeze().cpu().numpy()

    output3 = ind3.squeeze().cpu().numpy()

    predictions.append(output1)

    predictions.append(output2)

    predictions.append(output3)
submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')

submission['target'] = predictions

submission.to_csv('submission.csv', index = False)