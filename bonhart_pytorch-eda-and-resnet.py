# Libraries

import os

from glob import glob



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

# All data

print(os.listdir('../input'))
# Data path

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')

train_dir = '../input/train_images'

test_dir = '../input/test_images'
print('Total images for train {0}'.format(len(os.listdir(train_dir))))

print('Total images for test {0}'.format(len(os.listdir(test_dir))))
train_df.iloc[100:110]
test_df.iloc[100:110]
test_df.iloc[100:110]
# code from https://www.kaggle.com/gpreda/iwildcam-2019-eda



classes_wild = {0: 'empty', 1: 'deer', 2: 'moose', 3: 'squirrel', 4: 'rodent', 5: 'small_mammal', \

                6: 'elk', 7: 'pronghorn_antelope', 8: 'rabbit', 9: 'bighorn_sheep', 10: 'fox', 11: 'coyote', \

                12: 'black_bear', 13: 'raccoon', 14: 'skunk', 15: 'wolf', 16: 'bobcat', 17: 'cat',\

                18: 'dog', 19: 'opossum', 20: 'bison', 21: 'mountain_goat', 22: 'mountain_lion'}



train_df['classes_wild'] = train_df['category_id'].apply(lambda cw: classes_wild[cw])
# Category distribution

train_df['classes_wild'].value_counts()
plt.figure(figsize=(10,5))

train_df['classes_wild'].value_counts().plot(kind='bar',  title="Category distribution",);

plt.show()
def image_plotting(df, category, data_dir=train_dir):

    data_dir = data_dir

    df = train_df[train_df['classes_wild']== category]

    df = df[['classes_wild', 'file_name']]

    plt.rcParams['figure.figsize'] = (15, 15)

    plt.subplots_adjust(wspace=0, hspace=0)

    i_ = 0

    

    

    for l in range(25):

        cat, img_name = df.sample(1).values[0]

        path = os.path.join(train_dir, img_name)



        img = cv2.imread(path)

        img = cv2.resize(img, (256, 256)) 



        plt.subplot(5, 5, i_+1) #.set_title(l)

        plt.imshow(img); plt.axis('off')

        i_ += 1

    print(cat)
image_plotting(data_dir=train_dir,category='bobcat',df=train_df)
image_plotting(train_df, 'cat')
image_plotting(train_df, 'coyote')
image_plotting(train_df, 'deer')
image_plotting(train_df, 'dog')
image_plotting(train_df, 'empty')
image_plotting(train_df, 'fox')
image_plotting(train_df, 'mountain_lion' )
image_plotting(train_df, 'opossum' )
image_plotting(train_df, 'rabbit')
image_plotting(train_df, 'raccoon' )
image_plotting(train_df, 'rodent' )
image_plotting(train_df, 'skunk' )
image_plotting(train_df, 'squirrel' )
# Libraries

import torch

import torch.nn as nn

import torchvision

from torchvision.transforms import transforms

from torch.utils.data import Dataset, DataLoader

from torchvision import models



from sklearn.model_selection import train_test_split
train_df = train_df[['file_name','category_id']]

train_df.head()

# code lightly modified from https://www.kaggle.com/ateplyuk/iwildcam2019-pytorch-starter

category = train_df['category_id'].unique()



encoder = dict([(v, k) for v, k in zip(category, range(len(category)))])

decoder = dict([(v, k) for k, v in encoder.items()])





print( pd.DataFrame({

    'Before encoding': list(encoder.keys()),

    'After encoding': list(encoder.values())}).to_string(index=False))





def encoding(labels):

        return encoder[int(labels)]
train_df['category_id'] = train_df['category_id'].apply(encoding)

train_df['category_id'].value_counts()
# Custom data generator

class WildDataset(Dataset):

    def __init__(self, df, img_dir, transforms=None):

        self.df = df

        self.img_dir = img_dir

        self.transforms = transforms

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        img_name = os.path.join(self.img_dir,

                               self.df.iloc[idx, 0])

        image = cv2.imread(img_name)

        label = self.df.iloc[idx, 1]

        if self.transforms is not None:

            image = self.transforms(image)

        return image, label
train, val = train_test_split(train_df, stratify=train_df.category_id, test_size=0.1)

len(train), len(val)
# Augmentations for data



aug = transforms.Compose([transforms.ToPILImage(),                          

                          transforms.Resize((32, 32)),

                          transforms.ToTensor(),

                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                             ])





# iWildCam dataset

dataset_train = WildDataset(df=train,

                            img_dir=train_dir,

                            transforms=aug)



dataset_valid = WildDataset(df=val,

                           img_dir=train_dir,

                           transforms=aug)



# Data loader

train_loader = DataLoader(dataset=dataset_train, batch_size=24, shuffle=True)

val_loader = DataLoader(dataset_valid, batch_size=24, shuffle=False, num_workers=0)
# Aug for data img

def show_aug(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.figure(figsize=(20,15))

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated



    

# Get a batch of training data

inputs, _ = next(iter(train_loader))



# Make a grid from batch

out = torchvision.utils.make_grid(inputs, 4)  



show_aug(out)
_
_.shape
## Parameters for model



# Hyper parameters

num_epochs = 2

num_classes = 14

learning_rate = 0.02



# Device configuration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 3x3 convolution

def conv3x3(in_channels, out_channels, stride=1):

    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 

                     stride=stride, padding=1, bias=False)



  

# Residual block

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

        

    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample:

            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

      

      

      

# ResNet

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=14):

        super(ResNet, self).__init__()

        self.in_channels = 16

        self.conv = conv3x3(3, 16)

        self.bn = nn.BatchNorm2d(16)

        self.relu = nn.LeakyReLU(inplace=True)

        self.layer1 = self.make_layer(block, 16, layers[0])

        self.layer2 = self.make_layer(block, 32, layers[1], 2)

        self.layer3 = self.make_layer(block, 64, layers[2], 2)

        self.layer4 = self.make_layer(block, 128, layers[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d(4)

        self.fc = nn.Linear(128, num_classes)

        

    def make_layer(self, block, out_channels, blocks, stride=1):

        downsample = None

        if (stride != 1) or (self.in_channels != out_channels):

            downsample = nn.Sequential(

                conv3x3(self.in_channels, out_channels, stride=stride),

                nn.BatchNorm2d(out_channels))

        layers = []

        layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels

        for i in range(1, blocks):

            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    

    def forward(self, x):

        out = self.conv(x)

        out = self.bn(out)

        out = self.relu(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.avg_pool(out)

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out

      

      

      

def create_resnet_model(output_dim: int = 1) -> nn.Module:

    model = ResNet(ResidualBlock, [2, 2, 2, 2])

    in_features = model.fc.in_features

    model.avg_pool = nn.AdaptiveAvgPool2d(1)

    model.fc = nn.Linear(in_features, output_dim)

    model = model.to(device)

    return model



model = create_resnet_model(output_dim=num_classes)



# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)


# Train the model

total_step = len(train_loader)

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)

        labels = labels.to(device)

        

        # Forward pass

        outputs = model(images)

        loss = criterion(outputs, labels)

        

        # Backward and optimize

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if (i+1) % 100 == 0:

            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"

                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model

model.eval()

with torch.no_grad():

    correct = 0

    total = 0

    for images, labels in val_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



    print('Accuracy of the model on the 19630 test images: {} %'.format(100 * correct / total))

sub = pd.read_csv('../input/sample_submission.csv')

sub['Id'] = sub['Id'] + '.jpg'

sub.head()
# Dataset for test img

dataset_valid = WildDataset(df=sub,

                           img_dir=test_dir,

                           transforms=aug)



# Data loader

test_loader = DataLoader(dataset_valid, batch_size=24, shuffle=False)
# Test the model

model.eval()

preds = []

#with torch.no_grad():

for images, labels in test_loader:

    images = images.to(device)

    labels = labels.to(device)

    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)

    predicted

    for i in predicted.detach().cpu().numpy():

        preds.append(i)

sub['Predicted'] =  preds

sub['Id'] = sub['Id'].str[:-4]

sub.head()



def decoding(labels):

        return decoder[int(labels)]
sub['Predicted'] = sub['Predicted'].apply(decoding)

sub.head()



sub.to_csv('submission.csv', index=False)
sub['Predicted'].value_counts()