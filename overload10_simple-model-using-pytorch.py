# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input/aptos2019-blindness-detection/"))





# Any results you write to the current directory are saved as output.

from tqdm import tqdm

import pathlib

import matplotlib

import torch.nn as nn

import torch.optim as optim 

import torch

import torchvision

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image, ImageFile
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

df_sample = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

df_train.shape, df_test.shape, df_sample.shape
train_image_path = '../input/aptos2019-blindness-detection/train_images/'

test_image_path = '../input/aptos2019-blindness-detection/test_images/'
# Adding another column to contain the image path

df_train['image_file'] = train_image_path + df_train['id_code']+'.png' 

df_test['image_file'] = test_image_path + df_test['id_code']+'.png' 
df_train.head()
df_test.head()
df_test['diagnosis'] = 999
def load_image(filepath):

#     return io.imread(filepath)

#     return cv2.imread(filepath, cv2.COLOR_BGR2RGB)

    return Image.open(filepath)



def normalize(x, m, s): 

    return (x-m)/s



def apply_transform(image):

    transform_fn = transforms.Compose([transforms.Resize((480,480)),

#                                        transforms.CenterCrop((480,480)),                                       

                                      ])

    unnorm_img = transform_fn(image)

    norm_tf = transforms.Compose([transforms.ToTensor(), 

                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                                 ])

    

    

    return norm_tf(unnorm_img), unnorm_img
img = load_image(df_train['image_file'][0])

np.array(img).shape
#Setting up gpu parameter and batch_size

classes = 5

per_device_batch_size = 16



num_gpus = torch.cuda.device_count()

num_workers = num_gpus

ctx = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = per_device_batch_size * max(num_gpus, 1)

print(ctx,batch_size, num_workers)
# Example of images 

plt.figure(figsize=[15,15])

i = 1

for img_name in df_train['image_file'][:10]:

    img = load_image(img_name)

    plt.subplot(6,5,i)

    plt.imshow(img)

    i += 1

plt.title('Raw image without transformation')    

plt.show()
# Image after transformation

plt.figure(figsize=[15,15])

i = 1

for img_name in df_train['image_file'][:10]:

    _, img = apply_transform(load_image(img_name))

    print(np.array(img).shape)

    plt.subplot(6,5,i)

    plt.imshow(img)

    i += 1

# plt.title('Images after applying transformation')    

plt.show()
class CustomDataset(Dataset):

    def __init__(self, df, is_test= False):

        super().__init__()

        self.df = df

        self.is_test = is_test

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):                

        if not self.is_test:

            label = self.df.diagnosis[idx]

        else:

            label = None

            

        img_path = self.df.image_file[idx]

        img = load_image(df_train['image_file'][idx])

        image, _ = apply_transform(img)                    

        

        return image, label        
# from sklearn.model_selection import train_test_split



# X_train, X_test, y_train, y_test = train_test_split(df_train[['id_code','image_file']], df_train['diagnosis'],test_size=0.25, random_state=42, stratify=df_train['diagnosis'] )

# X_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)

# X_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

# X_train.shape, X_test.shape
# X_train.diagnosis.value_counts()
#  X_test.diagnosis.value_counts()
train_dataset = CustomDataset(df=df_train)

# valid_dataset = CustomDataset(df=X_test)

len(train_dataset)

# len(valid_dataset)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)

# valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers)
#Unit test dataloader block

for data, label in train_dl:

    print(data.shape, label.shape)

    break
ctx, batch_size
net =  torchvision.models.resnet50(pretrained=False)



# Freeze model weights

for param in net.parameters():

    param.requires_grad = False

    

net.fc = nn.Linear(in_features=2048, out_features=classes)

    

# nn.init()    

# net.initialize(init=init.Xavier(), ctx=ctx)

# net.collect_params().reset_ctx(ctx)

# net.hybridize()
# Transfer execution to device

model = net.to(ctx)
optimizer = optim.Adam(model.parameters())

loss_func = nn.CrossEntropyLoss()
%%time

# Train model

loss_log=[]

loss_values = []

for epoch in range(15):

    running_loss = 0.0

    model.train()

    dl = tqdm(train_dl, total=int(len(train_dl)))

    for ii, (data, target) in enumerate(dl):        

        data, target = data.to(ctx), target.to(ctx)

        optimizer.zero_grad()

        output = model(data)                    

        loss = loss_func(output, target)

        loss.backward()

        optimizer.step()          

        if ii % 1000 == 0:

            loss_log.append(loss.item())

        running_loss =+ loss.item() * data.size(0)

        

    loss_values.append(running_loss / len(train_dataset))

    

    print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))
torch.save(net.state_dict(), "model.bin")
plt.plot(loss_values)
# df_sample

test_dataset = CustomDataset(df_test, is_test=False)

test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
#Unit test dataloader block

for data, label in test_dl:

    print(data.shape, label.shape)

    break
%%time

# Prediction

predict = []

net.eval()

dl = tqdm(test_dl, total=int(len(test_dl)))

for i, (data, _) in enumerate(dl):

    data = data.cuda()

    output = net(data)  

    output = output.cpu().detach().numpy()    

    predict.append(output[0])
df_sample['diagnosis'] = np.argmax(predict, axis=1)

df_sample.head()



df_sample.to_csv('submission.csv', index=False)