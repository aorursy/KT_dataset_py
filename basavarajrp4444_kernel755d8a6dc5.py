%matplotlib inline
import torch
import pandas as pd
import numpy as np
import torchvision
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset,random_split
import torch.nn as nn
import torch.nn.functional as F
data_dir = '../input/jovian-pytorch-z2g/Human protein atlas'
train_data = data_dir + '/train'
test_data = data_dir + '/test'
train_csv = data_dir + '/train.csv'
df = pd.read_csv(train_csv)
x,y=df.loc[0]
x,y

labels = {
    0: 'Mitochondria',
    1: 'Nuclear bodies',
    2: 'Nucleoli',
    3: 'Golgi apparatus',
    4: 'Nucleoplasm',
    5: 'Nucleoli fibrillar center',
    6: 'Cytosol',
    7: 'Plasma membrane',
    8: 'Centrosome',
    9: 'Nuclear speckles'
}
x = []
y = []
for i in df.index:
    img_id,label = df['Image'][i],df['Label'][i]
    img_fname = train_data+'/'+str(img_id)+'.png'
    x.append(img_fname)
    y.append(label)

data = pd.DataFrame(x,columns = ['Images'])
data['Labels'] = y
data.head()
print(*data.iloc)
class dataprocessing(Dataset):
    
    def __init__(self, csv_file,transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
   
    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.loc[idx]
        img_id, img_label = row['Image'], row['Label']
        img_fname = train_data + "/" + str(img_id) + ".png"
        img = Image.open(img_fname)
        if self.transform:
            img = self.transform(img)
        return img, encode_label(img_label)

    
def encode_label(labels):
        lab = [0,0,0,0,0,0,0,0,0,0]
        x = list(labels.split(' '))
        for i in x:
            lab[int(i)] = 1
        return lab
    
def decode_label():
        proteins = []
        for i,j in enumerate(target):
            if j>=0.5:
                proteins.append(labels[i])
        return proteins         
    
            
            
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),    
    transforms.ToTensor()
    
    
])
dataset = dataprocessing(train_csv,transform=transform)
len(dataset)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)
train_loader = DataLoader(train_ds, batch_size=1000, shuffle=True)
val_loader = DataLoader(val_ds, 1000*2)
print(train_ds[0][0].size())
print(val_ds[0][0].size())
for img,lab in train_loader:
    print(img.size())
    img.flatten(start_dim=1).size()
    
    break
#class HumanProteinClass:
 #   def __init__(self):
  #      super().__init__()
   #     self.network = nn.Sequential(
    #        nn.Conv2d(3, 6, kernel_size=3,stride=1,padding=1),
     #       nn.ReLU(),
      #      nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1),
       #     nn.ReLU(),
        #    nn.MaxPool2d(2, 2), # output: 256 x 256 x12
#
 #           nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),
  #          nn.ReLU(),
   #         nn.Conv2d(24, 48 , kernel_size=3, stride=1, padding=1),
    #        nn.ReLU(),
     #       nn.MaxPool2d(2, 2), # output: 128 x 128 x 48

      #      nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
       #     nn.ReLU(),
        #    nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
         #   nn.ReLU(),
          #  nn.MaxPool2d(2, 2), # output: 64 x 64 x 192
           # 
    #        nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
     #       nn.ReLU(),
      #      nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=1),
       #     nn.ReLU(),
        #    nn.MaxPool2d(2, 2), # output: 32 x 32 x 768
#
 #           nn.Flatten(), 
  #          nn.Linear(32*32*768, 1024),
   #         nn.ReLU(),
    #        nn.Linear(1024, 512),
     #       nn.ReLU(),
      #      nn.Linear(512, 10))
        
#    def forward(self, xb):
 #       return self.network(xb)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(24, 48 , kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(in_features=32*32*768, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
    
        
        
    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv4(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv5(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv6(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv7(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv8(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = t.reshape(-1, 32*32*768)
        t = self.fc1(t)
        t = F.relu(t)
        
        t = self.fc2(t)
        t = F.relu(t)
    
        t = self.out(t)
        t = F.relu(t)
        

        return t
