# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
print(df_train.shape)

print(df_test.shape)
df_train.sample(3)
df_train.dtypes.unique()
df_train.select_dtypes(include='O').columns
df_train.select_dtypes(include='float64').columns
df_train.select_dtypes(include='int64').columns
#proportion of null values per columns

print('proportion of null values in train set : ')

print(df_train.isnull().sum(axis = 0).sort_values(ascending = False).head(10)/len(df_train))

print('\n')

print('proportion of null values in test set : ')

print(df_test.isnull().sum(axis = 0).sort_values(ascending = False).head(10)/len(df_test))
df = df_train.isnull().sum()[df_train.isnull().sum() !=0]/len(df_train)

df=pd.DataFrame(df.reset_index())

df.head(3)

total = len(df_train)

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))
#Exploring missing values

train_missing= df_train.isnull().sum()[df_train.isnull().sum() !=0]

train_missing=pd.DataFrame(train_missing.reset_index())

train_missing.rename(columns={'index':'features',0:'missing_count'},inplace=True)

train_missing['missing_count_percentage']=((train_missing['missing_count'])/len(df_train))*100

plt.figure(figsize=(15,7))

#train_missing

splot = sns.barplot(y=train_missing['features'],x=train_missing['missing_count_percentage'])

for p in splot.patches:

    splot.annotate(str(format(p.get_width(), '.2f')+'%'), (p.get_width()+3,p.get_y() + p.get_height() ), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
df_train.head(3)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df_train['Product_Info_2']=le.fit_transform(data_train['Product_Info_2'])

df_test['Product_Info_2']=le.transform(data_test['Product_Info_2'])
df_train.describe()
#Function for normalization

def normalization(data):

    return (data - data.min())/(data.max() - data.min())