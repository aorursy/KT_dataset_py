# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
train_features.head(5)
train_features.describe()
train_features.dtypes
test_features.head(5)
test_features.describe()
train_features.dtypes
train_targets_scored.head(5)
train_targets_scored.shape
train_targets_scored.describe()
train_targets_scored.dtypes
train_targets_nonscored.head(5)
train_targets_nonscored.shape
train_targets_nonscored.describe()
train_targets_nonscored.dtypes
train_features.isnull().sum()
test_features.isnull().sum()
train_targets_scored.isnull().sum()
train_targets_nonscored.isnull().sum()
train_features.columns.str.startswith('g-').sum()
train_features.columns.str.startswith('c-').sum()
plt.figure(figsize=(16, 16))

cols = [

    'c-1', 'c-2', 'c-3', 'c-4',

    'c-5', 'c-6', 'c-7', 'c-8',

    'c-92', 'c-93', 'c-94', 'c-95', 

    'c-96', 'c-97', 'c-98', 'c-99']

for i, col in enumerate(cols):

    plt.subplot(4, 4, i + 1)

    plt.hist(train_features.loc[:, col], bins=100, alpha=1,color='#66bfbf');

    plt.title(col)
plt.figure(figsize=(16, 16))

cols = [

    'g-1', 'g-2', 'g-3', 'g-4',

    'g-5', 'g-6', 'g-7', 'g-8',

    'g-92', 'g-93', 'g-94', 'g-95', 

    'g-96', 'g-97', 'g-98', 'g-99']

for i, col in enumerate(cols):

    plt.subplot(4, 4, i + 1)

    plt.hist(train_features.loc[:, col], bins=100, alpha=1,color='#66bfbf');

    plt.title(col)