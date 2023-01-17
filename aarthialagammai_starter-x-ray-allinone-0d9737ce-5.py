from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets

import pathlib
print(os.listdir('../input'))
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

transform = transforms.Compose([

    transforms.ToTensor()

])

data = datasets.ImageFolder('../input/x_ray_images_per_class/images_per_class',transform=transform)

dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)

images, labels = iter(dataloader).next()

numpy_images = images.numpy()

per_image_mean = np.mean(numpy_images, axis=(2,3)) 

per_image_std = np.std(numpy_images, axis=(2,3)) 

pop_channel_mean = np.mean(per_image_mean, axis=0) 

pop_channel_std = np.mean(per_image_std, axis=0) 

mean=(pop_channel_mean )

std=(pop_channel_std)
# how many samples per batch to load

batch_size = 20

# percentage of training set to use as validation and test

valid_size = 0.166

test_size=0.16

train_transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize(256),

                                     transforms.RandomHorizontalFlip(),transforms.RandomRotation(20),

                                    transforms.ColorJitter(),transforms.ToTensor(), transforms.Normalize(mean, std)])

valid_transform = transforms.Compose([transforms.Resize(256),transforms.ToTensor(),

                                        transforms.Normalize(mean, std)])   

    

test_transform = transforms.Compose([transforms.Resize(256),transforms.ToTensor(),

                                     transforms.Normalize(mean, std)])

    



train_data = datasets.ImageFolder('../input/x_ray_images_per_class/images_per_class',transform=train_transform)

valid_data = datasets.ImageFolder('../input/x_ray_images_per_class/images_per_class', transform=valid_transform)

test_data = datasets.ImageFolder('../input/x_ray_images_per_class/images_per_class', transform=test_transform)

num_train = len(train_data)

print("total_examples",num_train)

indices = list(range(num_train))

np.random.shuffle(indices)

valid_split = int(np.floor(valid_size * num_train))

test_split = int(np.floor(test_size * num_train))

valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:(test_split+valid_split)], indices[(test_split+valid_split):]

print("valid",len(valid_idx))

print("test",len(test_idx))

print("train",len(train_idx))
train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)

test_sampler= SubsetRandomSampler(test_idx)

# prepare data loaders (combine dataset and sampler)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,

    sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 

    sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 

    sampler=test_sampler)



root = pathlib.Path('../input/x_ray_images_per_class/images_per_class/')

classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)

print("No_of_classes",len(classes))