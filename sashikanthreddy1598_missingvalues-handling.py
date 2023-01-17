# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Import libraries

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Load Data



train = pd.read_csv('../input/train.csv')



test = pd.read_csv('../input/test.csv')

#Concatenate train & test

train_objs_num = len(train)

train_objs_num
y = train['Survived']

y
dataset = pd.concat(objs=[train.drop(columns=['Survived']), test], axis=0)
dataset.info()
total = dataset.isnull().sum().sort_values(ascending = False)

percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys = ['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y= missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel("Percent of missing values", fontsize=15)

plt.title('Percent of missing data by feature', fontsize=15)

missing_data.head()
dataset.head()
#Plot histogram using seaborn

plt.figure(figsize=(15,8))

sns.distplot(dataset.Age, bins = 30)
df = dataset

#Will drop all features with missing values

df.dropna(inplace = True)

df.isnull().sum()
df1 = dataset

#will drop the rows only if all the values in the row are missing

df1.dropna(how = 'all', inplace = True)
df1.isnull().sum()
df = dataset

#Will drop the feature that has some missing values.

df.dropna(axis = 1,inplace = True)
df = dataset

# Keep only rows with atleat 4 non-na values

df.dropna(thresh = 4, inplace = True)

df = dataset

df.fillna(method = 'bfill', inplace = True)#for back fill

df.fillna(method = 'ffill', inplace = True)# for forward fill

#MEAN: Suitable for continuous data without outliers

df3 = train

df3['Age'].isnull().sum()
df3['Age'].mean()
df3['Age'].replace(np.NaN,df3['Age'].mean()).head(15)
df4 = train

df4['Age'].fillna(df4['Age'].median(), inplace = True)

df4.head()
#Mode: For categorical feature we can select to fill in the missing values 

#with the most common value(mode) as illustrated below

data_cat = train

data_cat['Embarked'].fillna(data_cat['Embarked'].mode()[0], inplace = True)

data_cat.head()
data_unique = train

data_unique['Cabin'].head(10)
data_unique['Cabin'].fillna('U').head(10)