import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import torch
import torchvision

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from PIL import Image

from matplotlib.pyplot import figure


%matplotlib inline
fig = plt.figure(figsize=(8, 8), dpi=100,facecolor='w', edgecolor='k')
train_imgs = os.listdir("../input/train")
for idx, img in enumerate(np.random.choice(train_imgs, 12)):
    ax = fig.add_subplot(4, 20//5, idx+1, xticks=[], yticks=[])
    im = Image.open("../input/train/" + img)
    plt.imshow(im)
# load image data
#image name
df = pd.read_csv('../input/train.csv')
df.head()
# lets find total number of different whales present
print(f'Training examples: {len(df)}')
print("Unique whales: ",df['Id'].nunique()) # it includes new_whale as a separate type.
training_pts_per_class = df.groupby('Id').size()
print(training_pts_per_class)
print("Min example a class can have: ",training_pts_per_class.min())
print("0.99 quantile: ",training_pts_per_class.quantile(0.99))
print("Max example a class can have: \n",training_pts_per_class.nlargest(2))    
# max value belongs to new_whale category so the second max is the appropriate 
# representation of the max data points for a particualar class.
data = training_pts_per_class.copy()
data.loc[data > data.quantile(0.99)] = '22+'
plt.figure(figsize=(15,10))
sns.countplot(data.astype('str'))
plt.title("#classes with different number of images",fontsize=15)
plt.show()
# new_whales is addes as a new class.
# above graph shows that there are more than 2000 classes with just one training example.
# and around 1300 classes with 2 training examples.
# it also shows that around 50 classes have more than 22 training examples.
print(data)
print(len(os.listdir('../input/train/')))
class HW_Dataset(Dataset):
    def __init__(self,filepath, csv_path,transform=None):
        self.file_path = filepath
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_list = [x for x in os.listdir(self.file_path)]
        
    def __len__(self):
        return(len(self.image_list))
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.file_path,self.df.Image[idx])
        label = self.df.Id[idx]
        
        #img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        return img, label
        
transform = transforms.Compose([transforms.Resize((256,256)),
                                transforms.ToTensor()])
                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

train_dataset = HW_Dataset('../input/train/','../input/train.csv', transform)
data_generator = DataLoader(train_dataset,batch_size=16, shuffle=True)
images, labels = next(iter(data_generator))
print(f'images size: {images.size()}')
figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
grid = torchvision.utils.make_grid(images,nrow=4)
plt.imshow(grid.numpy().transpose((1,2,0)))
plt.axis('off')
#plt.title(labels)
