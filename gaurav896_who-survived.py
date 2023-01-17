# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
train.head()
print(train.shape)

print('\n')

print(test.shape)
train.columns
train.info()
sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis',cbar=False)
sns.heatmap(test.isnull(),yticklabels=False,cmap='viridis',cbar=False)
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='RdBu_r',)
sns.set_style('whitegrid')

sns.catplot(x='Sex',y='Survived',hue='Pclass',kind='bar',data=train)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')