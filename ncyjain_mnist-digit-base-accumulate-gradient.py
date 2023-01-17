!pip install torchsummary
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import torch

from torch.utils.data import Dataset , DataLoader



import albumentations as A



from  sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



print(df.shape, test_df.shape)

class MNIST_dataset(Dataset):

    def __init__(self, df):

        self.df = df



        self.aug = A.Compose([

            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=.75)

        ])

        

    def __len__(self):

        return(len(self.df))

        

    def __getitem__(self, idx):

        img_data = self.df.iloc[idx,1:].values.reshape((1,28,28)).astype(np.float32) / 255.

        img_data = self.aug(image=img_data)['image']

        label = self.df.iloc[idx, 0]

        return img_data, label

    
train_df , valid_df = train_test_split(df, test_size=0.2, random_state=1)
train_dl = DataLoader(MNIST_dataset(train_df), batch_size=128)

valid_dl = DataLoader(MNIST_dataset(valid_df), batch_size=128)
from torch import nn

num_groups = 4

class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),

            nn.ReLU(inplace=True),

            nn.BatchNorm2d(32),

            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),   

            nn.ReLU(inplace=True),

            nn.BatchNorm2d(32),

            

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

            nn.ReLU(inplace=True),

            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  

            nn.ReLU(inplace=True),

            nn.BatchNorm2d(64),

                     

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

            nn.ReLU(inplace=True),

            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  

            nn.ReLU(inplace=True),

            nn.BatchNorm2d(128),

            nn.Dropout2d(0.25),

        )

        

        self.classifier = nn.Sequential(

            nn.Linear(4*4*128, 256),

            nn.ReLU(inplace=True),

            nn.BatchNorm1d(256),

            nn.Dropout2d(0.25),

            nn.Linear(256, 10),

        )

            

    

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return(x)

    
model = Model().cuda()

# print(model)
from torchsummary import summary

summary(model, input_size=(1, 28, 28))
from torch import optim

from tqdm.auto import tqdm
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
def train_one_epoch(dl, epoch_num):

    

    total_loss = 0.0

    accumulation_steps = 1024 // 128

    optimizer.zero_grad()

    for i,(X, y)  in enumerate(tqdm(dl)):

        y1 = model(X.cuda())

        loss = criterion(y1, y.cuda())

        loss /= accumulation_steps

        loss.backward()

        

        if((i+1) % accumulation_steps == 0):

             

            optimizer.step()

            optimizer.zero_grad()

        

        total_loss += loss.detach().cpu().item()

        

    print(f'epoch : {epoch_num}, Loss : {total_loss/len(dl.dataset):.6f}')

        
def evaluate(dl):

    total_loss = 0.0

    total_correct = 0.0

    

    with torch.no_grad():

        

        for X,y in dl :

            

            y1 = model(X.cuda())

            loss = criterion(y1,y.cuda())

            pred = torch.argmax(y1, dim=1).cpu()

            

            total_loss+=loss.item()

            total_correct += torch.sum(y==pred).float().item()

            

        accuracy = total_correct/len(dl.dataset)

        

    print(f'Loss : {total_loss/len(dl.dataset):.6f}, Accuracy : {accuracy*100:.3f}%')

            
epoch_num = 20

for epoch in range(epoch_num):

    train_one_epoch(train_dl, epoch)

    evaluate(valid_dl)
class MNIST_dataset(Dataset):

    def __init__(self, df):

        self.df = df

        

    def __len__(self):

        return(len(self.df))

        

    def __getitem__(self, idx):

        img_data = self.df.iloc[idx].values.reshape((1,28,28)).astype(np.float32) /  255.

        return img_data

    
test_dl = DataLoader(MNIST_dataset(test_df), batch_size=128)
def evaluate_Submssions(dl):

    total_loss = 0.0

    total_correct = 0.0

    pred_list =[]

    

    with torch.no_grad():

        

        for X in dl :

            

            y1 = model(X.cuda())

            

            pred = torch.argmax(y1, dim=1).detach().cpu().numpy().tolist()

            pred_list.extend(pred)

            

    return pred_list

            

            
pred_list = evaluate_Submssions(test_dl)

subs = pd.DataFrame({

    'ImageId': range(1, len(pred_list)+1),

    'Label' : pred_list

})



subs.head()
subs.to_csv("submission.csv", index= False)