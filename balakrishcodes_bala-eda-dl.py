# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/fashion-product-images-small/styles.csv", nrows=6000, error_bad_lines=False)

df = df.sample(frac=1).reset_index(drop=True)

df.head(10)
[print(i, df[i].unique(), end="\n\n") for i in df.columns if df[i].dtype=='object']
df.info()

df = df.dropna(axis=0)

df.info()
df.describe()
numerical_data = ['year']

categorical_data = ['masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
sns.distplot(df['year'], norm_hist=False, kde=False, hist_kws={"alpha": 1})#.set(xlabel='Sale Price', ylabel='Count');
print(df.columns)

f, axes = plt.subplots(3, 3, figsize=(20,20))

sns.countplot(x="gender", data=df, ax=axes[0,0])

sns.countplot(x="masterCategory", data=df, ax=axes[0,1])

sns.countplot(y="baseColour", data=df, ax=axes[0,2])



sns.countplot(y="subCategory", data=df, ax=axes[1,0])

sns.countplot(x="season", data=df, ax=axes[1,1])

sns.countplot(x="year", data=df, ax=axes[1,2])



sns.countplot(y="articleType", data=df, ax=axes[2,0])

sns.countplot(x="usage", data=df, ax=axes[2,1])

f, axes = plt.subplots(2, 2, figsize=(20,20))

sns.countplot(x="gender", hue="masterCategory", data=df, ax=axes[0,0])

sns.countplot(x="masterCategory", hue="gender", data=df, ax=axes[0,1])



sns.countplot(x="season", hue="masterCategory", data=df, ax=axes[1,0])

sns.countplot(x="usage", hue="masterCategory", data=df,ax=axes[1,1])
f, ax = plt.subplots(2, 3, figsize=(15, 15))

for var, subplot in zip(categorical_data, ax.flatten()):

    sns.boxplot(x=var, y='year', data=df, ax=subplot)
f, ax = plt.subplots(4, 3, figsize=(15, 15))

for i, subplot in (zip(sorted(df.year.unique()),ax.flatten())):

    df[df.year == i].groupby('season').count().plot(kind="bar",title=i, ax=subplot)

    plt.tight_layout()
f, ax = plt.subplots(2, 2, figsize=(15, 15))

for i, subplot in (zip(sorted(df.season.unique()),ax.flatten())):

    df[df.season == i].groupby('masterCategory').count().plot(kind="bar",title=i, ax=subplot)

    plt.tight_layout()
f, ax = plt.subplots(2, 2, figsize=(15, 15))

for i, subplot in (zip(sorted(df.season.unique()),ax.flatten())):

    df[df.season == i].groupby('year').count().plot(kind="bar",title=i, ax=subplot)

    plt.tight_layout()
import numpy as np

import torch 

import matplotlib.pylab as plt

import numpy as np

import pandas as pd

import time, os, random

import h5py

from torch.utils.data import Dataset, DataLoader

from keras.utils import to_categorical

from torchvision import transforms

print(torch.__version__)

import nibabel as nib

from torch.autograd import Variable

import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models

from torch.utils.data import Dataset, DataLoader

import torchvision

from torchvision import transforms, utils

#!pip install torchsummary --quiet

!pip install torchsummaryX  --quiet

from torchsummaryX import summary
df = pd.read_csv('../input/fashion-product-images-small/styles.csv',error_bad_lines=False)

df['image_path'] = df.apply(lambda x : os.path.join("/kaggle/input/fashion-product-images-small/myntradataset/images",str(x.id)+".jpg"), axis=1)

df.head()
mapper = {}

for i,cat in enumerate(list(df.masterCategory.unique())):

    mapper[cat] = i

print(mapper)

df['targets'] = df.masterCategory.map(mapper)

df.head()
img = plt.imread('/kaggle/input/fashion-product-images-small/myntradataset/images/4711.jpg')

plt.imshow(img)
for i in range(6):

    print("label {} - Total Count {}".format(i,df.targets[df.targets==i].count()))
fold = ['train']*(int(len(df)*0.9)) + ['valid']*(len(df) - int(len(df)*0.9))

random.shuffle(fold)

df['fold'] = fold

df.head()
sns.countplot(df['fold'])
df.image_path[0]
NUM_SAMP=5

fig = plt.figure(figsize=(25, 16))

import cv2

for jj in range(5):

    for i, (idx, row) in enumerate(df.sample(NUM_SAMP,random_state=123+jj).iterrows()):

        ax = fig.add_subplot(5, NUM_SAMP, jj * NUM_SAMP + i + 1, xticks=[], yticks=[])

        path=f"/kaggle/input/fashion-product-images-small/myntradataset/images/{row['id']}.jpg"

        image = plt.imread(path)

        plt.imshow(image)

        ax.set_title('%d-%s' % (idx, row['id']) )
classifier = True # input as False makes the model regressor.



# Flag for feature extracting. When False, we finetune the whole model,

#   when True we only update the reshaped layer params

feature_extract = False



if classifier:

    num_classes = 5 # Classifier

    criterion =  nn.CrossEntropyLoss() 

else:

    num_classes = 1 # Regressor

    criterion =  nn.MSELoss() 







def set_parameter_requires_grad(model, feature_extracting):

    if feature_extracting:

        for param in model.parameters():

            param.requires_grad = False



def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

    # Initialize these variables which will be set in this if statement. Each of these

    #   variables is model specific.

    model_ft = None

    input_size = 0



    if model_name == "resnet":

        """ Resnet18

        """

        model_ft = models.resnet18(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.fc.in_features

        model_ft.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        input_size = 224



    elif model_name == "alexnet":

        """ Alexnet

        """

        model_ft = models.alexnet(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier[6].in_features

        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

        input_size = 224



    elif model_name == "vgg":

        """ VGG11_bn

        """

        model_ft = models.vgg11_bn(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier[6].in_features

        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

        input_size = 224



    elif model_name == "squeezenet":

        """ Squeezenet

        """

        model_ft = models.squeezenet1_0(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

        model_ft.num_classes = num_classes

        input_size = 224



    elif model_name == "densenet":

        """ Densenet

        """

        model_ft = models.densenet121(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier.in_features

        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

        input_size = 224



    elif model_name == "inception":

        """ Inception v3

        Be careful, expects (299,299) sized images and has auxiliary output

        """

        model_ft = models.inception_v3(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        # Handle the auxilary net

        num_ftrs = model_ft.AuxLogits.fc.in_features

        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        # Handle the primary net

        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs,num_classes)

        input_size = 299



    else:

        print("Invalid model name, exiting...")

        exit()



    return model_ft, input_size

# Initialize the model for this run

model_name = "resnet" # Models to choose ["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"]

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)



# Print the model we just instantiated

print(model_ft)

print()

print("Input image size format",(input_size,input_size))
summary(model_ft, torch.zeros((1, 3, 224, 224)))
feature_extract = True



BATCH_SIZE =  16 # Desired batch size

SAMPLE = 0 # Increase the sample size if you want to train only on a specific number of samples, otherwise to train on entire datset, set sample = 0

img_size = input_size # This sets the input image size based on the model's you choose

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



print("Running on",device)

model_ft = model_ft.to(device)



params_to_update = model_ft.parameters()

print("Params to learn:")

if feature_extract:

    params_to_update = []

    for name,param in model_ft.named_parameters():

        if param.requires_grad == True:

            params_to_update.append(param)

            print("\t",name)

else:

    for name,param in model_ft.named_parameters():

        if param.requires_grad == True:

            print("\t",name)





learning_rate=0.01

# optimizer = optim.Adam(params_to_update, lr=learning_rate)

optimizer = optim.SGD(params_to_update, lr=learning_rate , momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.85, patience=2, verbose=True)
from PIL import Image



class FDataset(Dataset):

    # Constructor

    def __init__(self, df, fold , img_size, transform=True):

        # Image directory

        self.transform = transform

        self.img_size = img_size

        self.fold = fold

        self.df = df

        self.df = self.df[self.df['fold'] == fold]

        #print(self.df.head())

        if transform is None:

            transform = torchvision.transforms.Compose([

                torchvision.transforms.Resize((224, 224)),

                torchvision.transforms.ToTensor()

            ])

        self.transform = transform



    # Get the length

    def __len__(self):

        return len(self.df)

    

    # Getter

    def __getitem__(self, idx):

        img_path = self.df.image_path[idx]

        #print(img_path)

        img = Image.open(img_path).convert('RGB')

        img_tensor = self.transform(img)

        

        label = self.df.targets[idx]        

        print(label)

        return image, label
transformed_datasets = {}

transformed_datasets['train'] = FDataset(df,  fold="train" ,img_size = img_size)

transformed_datasets['valid'] = FDataset(df,  fold="valid" ,img_size = img_size)

 

dataloaders = {}

dataloaders['train'] = torch.utils.data.DataLoader(transformed_datasets['train'],batch_size=BATCH_SIZE,shuffle=True)

dataloaders['valid'] = torch.utils.data.DataLoader(transformed_datasets['valid'],batch_size=BATCH_SIZE,shuffle=True)  

print()

print(len(dataloaders['train']))

print(len(dataloaders['valid']))