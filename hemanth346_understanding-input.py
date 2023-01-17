import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import gc # garbage collector



%matplotlib inline
project_dir = '/kaggle/input/bengaliai-cv19/'
import glob

csv_files = [file for file in glob.glob(project_dir+"*.csv")]

train_parquet_files =  [file for file in glob.glob(project_dir+"train*.parquet")]

test_parquet_files =  [file for file in glob.glob(project_dir+"test*.parquet")]
csv_files
sample_submission = '/kaggle/input/bengaliai-cv19/sample_submission.csv'

class_maps = '/kaggle/input/bengaliai-cv19/class_map.csv'

test = '/kaggle/input/bengaliai-cv19/test.csv'

train = '/kaggle/input/bengaliai-cv19/train.csv'



# For some strange reason, when I commit csv_files order is changing



# sample_submission = csv_files[0]

# class_maps = csv_files[1]

# test = csv_files[2]

# train = csv_files[3]
def csv_overview(csv_file, name='', head=3, tail=3, columns=False, describe=False, info=True):

    print('file :', csv_file)

    df = pd.read_csv(csv_file)

    print('{} Shape : '.format(name),df.shape)

    print('-'*36)

    if columns:

        print('{} Columns : '.format(name),df.columns)

        print('-'*36)

    if describe:

        print('{} Distribution :\n'.format(name),df.describe().T)

        print('-'*36)

    if info:

        print('{} Summary :\n'.format(name))

#         print(df.info())

        df.info()

        print('-'*36)

    print('{} Unique values :\n'.format(name),df.nunique())

    print('-'*36)

    print('Sample data')

    print('-'*12)

    print('head')

    print(df.head(head))

    print('-'*12)

    print('tail')

    print(df.tail(tail))
csv_overview(train, 'train_df', columns=True)
csv_overview(test, 'test_df', head=5, tail=5, columns=True)
csv_overview(class_maps, 'Class Maps', columns=True)
csv_overview(sample_submission, 'Sample Submissions', head=5, columns=True)
def explore_parquet(file, name='', head=3, tail=3, columns=False, describe=False, unique=False, info=True):

    print('file : {}'.format(file))

    df = pd.read_parquet(file)

    print('{} Shape : '.format(name),df.shape)

    print('-'*36)

    if columns:

        print('{} Columns : '.format(name),df.columns)

        print('-'*36)

    if describe:

        print('{} Distribution :\n'.format(name),df.describe().T)

        print('-'*36)

    if info:

        print('{} Summary :\n'.format(name))

#         print(df.info())

        df.info()

        print('-'*36)

    if unique:

        print('{} Unique values :\n'.format(name),df.nunique())

        print('-'*36)

    print('Sample data')

    print('-'*12)

    print('head')

    print(df.head(head))

    print('-'*12)

    print('tail')

    print(df.tail(tail))

def visualize_parquet(file, shape=(137, 236), cmap=None):

    # shape - (height, width)

    df = pd.read_parquet(file)

    df1 = df.head(25)

    labels, images = df1.iloc[:, 0], df1.iloc[:, 1:].values.reshape(-1, *shape) 

    

    f, ax = plt.subplots(5, 4, figsize=(20, 20))

    ax = ax.flatten()

    f.suptitle(file) #super title

    

    for i in range(20):

        ax[i].set_title(labels[i])

        ax[i].imshow(images[i], cmap=cmap)
train_parquet_files
for file in train_parquet_files:

    explore_parquet(file)

    print('==='*18)
for file in train_parquet_files:

    visualize_parquet(file, cmap='Blues')
test_parquet_files
for file in test_parquet_files:

    explore_parquet(file)

    print('==='*18)

        
shape=(137, 236)

for file in test_parquet_files:

    print('file', file)

    df = pd.read_parquet(file)

    labels, images = df.iloc[:, 0], df.iloc[:, 1:].values.reshape(-1, *shape) 



    f, ax = plt.subplots(1, 3, figsize=(10, 10))

    ax = ax.flatten()

#     f.suptitle(file) #super title



    for i in range(3):

        ax[i].set_title(labels[i])

        ax[i].imshow(images[i], cmap='Blues')