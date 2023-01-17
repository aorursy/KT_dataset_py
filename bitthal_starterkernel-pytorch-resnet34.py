import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import gc





from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim





from sklearn.model_selection import train_test_split



BASE_DIR = '/kaggle/input/bengaliai-cv19'



# os.listdir(BASE_DIR)
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))

print('Shape of train_df: ', train_df.shape)



test_df = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

print('Shape of test_df: ', test_df.shape)



class_map = pd.read_csv(os.path.join(BASE_DIR, 'class_map.csv'))

print('Shape of class_map: ', class_map.shape)



sample_submission_df = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))

print('Shape of sample submission: ', sample_submission_df.shape)



train_image_files = [x for x in os.listdir(BASE_DIR) if 'train_ima' in x]

print("Number of Train Image files: ", len(train_image_files))



test_image_files = [x for x in os.listdir(BASE_DIR) if 'test_ima' in x]

print("Number of Train Image files: ", len(test_image_files))
#Credits: https://www.kaggle.com/phoenix9032/pytorch-efficientnet-starter-code/data



SIZE = 128



def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=SIZE, pad=16):

    

    #crop a box around pixels large than the threshold 

    #some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    

    #cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < ORIGINAL_WIDTH - 13) else ORIGINAL_WIDTH

    ymax = ymax + 10 if (ymax < ORIGINAL_HEIGHT - 10) else ORIGINAL_HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    

    #remove lo intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    

    #make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return cv2.resize(img,(size,size))



def Resize(df,size=128):

    resized = {} 

    

    for i in tqdm(range(df.shape[0])): 

        image0 = 255 - df.loc[df.index[i]].values.reshape(137,236).astype(np.uint8)

        

        #normalize each image by its max val

        img = (image0*(255.0/image0.max())).astype(np.uint8)

        image = crop_resize(img)

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized
ORIGINAL_HEIGHT = 137

ORIGINAL_WIDTH = 236



def load_npa(file):

    df = pd.read_parquet(file)

    return df.iloc[:, 1:].values.reshape(-1, ORIGINAL_HEIGHT, ORIGINAL_WIDTH)



def load_pa_df(file):

    df = pd.read_parquet(file)

    df = df.set_index('image_id')

    return df



# loading one of the parquest file for analysis

train_image_files.sort()

train_images = [load_pa_df(os.path.join(BASE_DIR, x)) for x in train_image_files]

for images in train_images:

    print("Number of images in loaded files: ", images.shape[0])

    

print("Number of columns in image df: ", images.shape[1])

for i, images in enumerate(train_images):

    print("Number of images in loaded file {}: {}\n".format(i, images.shape[0]))

    print("Images in loaded file {}: {}\n\n".format(i, images.index.values))

#     print(images.head())



def get_image_from_dfrow(df, row_id):

    df_row = df.iloc[row_id]

    pixel_values = df_row.values[1:]

    return df_row.values.reshape(ORIGINAL_HEIGHT, ORIGINAL_WIDTH).astype('int')



f, ax = plt.subplots(5, 5, figsize=(16, 8))

for i in range(5):

    for j in range(5):

        ax[i][j].imshow(get_image_from_dfrow(train_images[0], i*5+j), cmap='Greys')



from tqdm import tqdm

import cv2

for i, images in enumerate(train_images):

    train_images[i] = Resize(train_images[i])

    

# save for faster training

train_images[0].to_feather('train-images0.feather')

train_images[1].to_feather('train-images1.feather')

train_images[2].to_feather('train-images2.feather')

train_images[3].to_feather('train-images3.feather')
# train_images = [None, None, None, None]

# train_images[0] = pd.read_feather('train-images0.feather')

# train_images[1] = pd.read_feather('train-images1.feather')

# train_images[2] = pd.read_feather('train-images2.feather')

# train_images[3] = pd.read_feather('train-images3.feather')
def get_image_from_dfrow(df, row_id):

    df_row = df.iloc[row_id]

    pixel_values = df_row.values[1:]

    return pixel_values.reshape(128, 128).astype('int')



f, ax = plt.subplots(5, 5, figsize=(16, 8))

for i in range(5):

    for j in range(5):

        ax[i][j].imshow(get_image_from_dfrow(train_images[0], i*5+j), cmap='Greys')
data_full = pd.concat(train_images,ignore_index=True)



del train_images

gc.collect()
class GraphemeDataset(Dataset):

    def __init__(self,df,label=None,_type='train',transform =True,aug=None):

        df = df.set_index('image_id')

        self.df = df

        self.label = label

        self.aug = aug

        self.transform = transform

        self.type=_type

        

    def __len__(self):

        return len(self.label)

    

    def __getitem__(self,idx):

        

        if self.type=='train':

            label1 = self.label.vowel_diacritic.values[idx]

            label2 = self.label.grapheme_root.values[idx]

            label3 = self.label.consonant_diacritic.values[idx]

            name = self.label.image_id.values[idx]

            image = self.df.loc[name].values.reshape(SIZE,SIZE).astype(np.float)

                        

#             augment = self.aug(image =image)

#             image = augment['image']

            img_ = image.reshape(1, 128, 128)



            return img_, [label1, label2, label3]

        else:

            image = self.df.loc[name].values.reshape(SIZE,SIZE).astype(np.float)

            return image
HEIGHT = 128

WIDTH = 128



BATCH = 16



train, test = train_test_split(train_df, test_size=0.1)



train.reset_index(inplace=True)

test.reset_index(inplace=True)



split_train_df = train

split_val_df = test



print("Train Data Shape: ", split_train_df.shape)

print("Test Data Shape: ", split_val_df.shape)



train_dataset = GraphemeDataset(data_full ,split_train_df,transform = False)

val_dataset = GraphemeDataset(data_full , split_val_df,transform = False)



train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=2)

val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ResidualBlock(nn.Module):

    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):

        super(ResidualBlock,self).__init__()

        self.cnn1 =nn.Sequential(

            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(True)

        )

        self.cnn2 = nn.Sequential(

            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),

            nn.BatchNorm2d(out_channels)

        )

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),

                nn.BatchNorm2d(out_channels)

            )

        else:

            self.shortcut = nn.Sequential()

            

    def forward(self,x):

        residual = x

        x = self.cnn1(x)

        x = self.cnn2(x)

        x += self.shortcut(residual)

        x = nn.ReLU(True)(x)

        return x
class ResNet34(nn.Module):

    def __init__(self):

        super(ResNet34,self).__init__()

        

        self.block1 = nn.Sequential(

            nn.Conv2d(1,64,kernel_size=2,stride=2,padding=3,bias=False),

            nn.BatchNorm2d(64),

            nn.ReLU(True)

        )

        

        self.block2 = nn.Sequential(

            nn.MaxPool2d(1,1),

            ResidualBlock(64,64),

            ResidualBlock(64,64,2)

        )

        

        self.block3 = nn.Sequential(

            ResidualBlock(64,128),

            ResidualBlock(128,128,2)

        )

        

        self.block4 = nn.Sequential(

            ResidualBlock(128,256),

            ResidualBlock(256,256,2)

        )

        self.block5 = nn.Sequential(

            ResidualBlock(256,512),

            ResidualBlock(512,512,2)

        )

        

        self.avgpool = nn.AvgPool2d(2)

        # vowel_diacritic

        self.fc = nn.Linear(2048,512)

        self.fc1 = nn.Linear(512,11)

        # grapheme_root

        self.fc2 = nn.Linear(512,168)

        # consonant_diacritic

        self.fc3 = nn.Linear(512,7)

        

    def forward(self,x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = self.avgpool(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        x1 = self.fc1(x)

        x2 = self.fc2(x)

        x3 = self.fc3(x)

        return x1,x2,x3

    

model = ResNet34()

model.to(device)
# Model Test

# model(torch.Tensor(np.zeros((2, 1, 128, 128))).to(device))

# os.listdir('/kaggle/input/')
## Loaidng on CPU

# model.load_state_dict(torch.load('/kaggle/input/v5-data/model_v3_39.pth', map_location=torch.device('cpu')))

## Loading on GPU

model.load_state_dict(torch.load('/kaggle/input/v5-data/model_v3_39.pth'))
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
def accuracy(prediction, labels):

    ans = 0

    for pred, y in zip(prediction, labels):

        _, pred = torch.max(pred.data, 1)

        ans += (pred == y).sum().item()

    return ans
## Running for 1 Epoch for demonstration

for epoch in range(1):  # loop over the dataset multiple times



    train_loss = 0

    test_loss = 0

    train_accuracy = 0

    test_accuracy = 0

    count = 0

    model.train()

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data

        

        inputs = inputs.to(device, dtype=torch.float)

        labels = [x.to(device) for x in labels]

        

        optimizer.zero_grad()

        outputs = model(inputs)

        

        loss = criterion(outputs[0], labels[0]) + criterion(outputs[1], labels[1]) + criterion(outputs[2], labels[2])

        loss.backward()

        optimizer.step()

        

        train_accuracy += accuracy(outputs, labels)

        train_loss += loss.item()



        if (i+1) % 2000 == 0:

            print("Step: {}, TrainAccuracy: {}, TrainLoss: {}".format(i+1, train_accuracy/((i+1)*BATCH*3), train_loss))

            





    

    model.eval()

    with torch.no_grad():

        for i, data in enumerate(val_loader, 0):

            inputs, labels = data



            inputs = inputs.to(device, dtype=torch.float)

            labels = [x.to(device) for x in labels]

            outputs = model(inputs)

            

            loss = criterion(outputs[0], labels[0]) + criterion(outputs[1], labels[1]) + criterion(outputs[2], labels[2])



            test_accuracy += accuracy(outputs, labels)

            test_loss += loss.item()



        

        

        

        

        

        

    print('Epoch: {}, TrainLoss: {}, TestLoss: {}, TrainAccuracy: {}, TestAccuracy: {}'.format(epoch + 1,

                                                                                        train_loss, test_loss, 

                                                                                        train_accuracy/(len(train_dataset)*3), test_accuracy/(len(val_dataset)*3)))

    if (epoch) % 2 == 0:

        torch.save(model.state_dict(), 'model_v3_{}.pth'.format(epoch+35))

del data_full, train_loader, val_loader

gc.collect()
test_df = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

print('Shape of test_df: ', test_df.shape)

print(test_df.head(), '\n\n')



sample_submission_df = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))

print('Shape of sample submission: ', sample_submission_df.shape)

print(sample_submission_df.head())
# loading one of the parquest file for analysis

test_image_files.sort()

test_images = [load_pa_df(os.path.join(BASE_DIR, x)) for x in test_image_files]

for images in test_images:

    print("Number of images in loaded files: ", images.shape[0])
from tqdm import tqdm

import cv2

for i, images in enumerate(test_images):

    test_images[i] = Resize(test_images[i])
test_data_full = pd.concat(test_images,ignore_index=True)



class GraphemeDatasetTest(Dataset):

    def __init__(self,df,label=None,_type='train',transform =True,aug=None):

        df = df.set_index('image_id')

        self.df = df

        self.label = label

#         self.aug = aug

        self.transform = transform

        self.type=_type

        

    def __len__(self):

        return len(self.label)

    

    def __getitem__(self,idx):

        name = self.label[idx]

        image = self.df.loc[name].values.reshape(SIZE,SIZE).astype(np.float)



        img_ = image.reshape(1, 128, 128)



        return img_, name



test_dataset = GraphemeDatasetTest(test_data_full , test_df.image_id.unique(),transform = False)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
model.eval()

predictions = []

batch_size=1

names = []

with torch.no_grad():

    for idx, (inputs, tag) in enumerate(test_loader):

        inputs = inputs.to(device, dtype=torch.float)



        outputs1,outputs2,outputs3 = model(inputs)

        

        predictions.append(outputs3.argmax(1).cpu().detach().numpy()[0])

        predictions.append(outputs2.argmax(1).cpu().detach().numpy()[0])

        predictions.append(outputs1.argmax(1).cpu().detach().numpy()[0])
submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')

submission.target = np.hstack(predictions)

submission.head(10)
submission.to_csv('submission.csv',index=False)