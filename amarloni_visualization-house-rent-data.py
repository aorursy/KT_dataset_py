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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('ggplot')
data = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

data.head()
data.isnull().any()
data.info()
data.isna().sum()
data.describe()
data.columns.unique()
Q1 = data.quantile(0.05)

Q3 = data.quantile(0.80)

IQR = Q3 - Q1

print(IQR)
data_out = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

data_out.shape


data_out.plot.box(grid='True', figsize = (20,10))

plt.show()
data1 = data_out.drop(['city', 'area', 'rooms', 'bathroom', 'parking spaces', 'floor','animal', 'furniture', 'fire insurance (R$)', 'property tax (R$)'], axis = 1)

data2 = data_out.drop(['hoa (R$)', 'rent amount (R$)','total (R$)', 'rooms', 'parking spaces', 'bathroom'], axis = 1)

data3 = data_out.drop(['city','area', 'floor','animal', 'furniture', 'hoa (R$)', 'rent amount (R$)',

       'property tax (R$)', 'fire insurance (R$)', 'total (R$)'], axis = 1)
#plt.figure(figsize = (10,5))

fig, axs = plt.subplots(ncols=3, figsize=(15,5))

data1.plot.box(grid='True',  ax = axs[0])

data2.plot.box(grid='True',  ax = axs[1])

data3.plot.box(grid='True',  ax = axs[2])

plt.show()
fig, axs = plt.subplots(1,3, figsize=(18,5))

#plt.figure(figsize = (10,5))

sns.barplot(x='city', y = 'area', data = data_out, ax= axs[0])

sns.barplot(x='city', y = 'parking spaces', data = data_out, ax= axs[1])

sns.barplot(x='city', y = 'rooms', data = data_out, ax= axs[2])

plt.show()

data.city.unique()
#plt.style.use('black_background')

#plt.figure(figsize=(10,5))

#sns.barplot(x = 'rooms', y ='area', hue = 'rent amount (R$)', data = data_out) 

#plt.show()
fig, axs = plt.subplots(1,3, figsize=(20,5))



sns.barplot(x = 'rooms', y ='rent amount (R$)', data = data_out, ax = axs[0]) 

plt.xticks(rotation=90)

sns.barplot(x = 'city', y ='hoa (R$)', data = data_out, ax= axs[1])

plt.xticks(rotation=90)

sns.barplot(x = 'city', y ='fire insurance (R$)', data = data_out, ax= axs[2])

plt.xticks(rotation=90)

plt.show()
x= data_out['total (R$)']

x1 = data_out['hoa (R$)']

x2 = data_out['fire insurance (R$)']

x3 = data_out['property tax (R$)']

#x3 = data_out['property tax (R$)']

fig, axs = plt.subplots(1,4, figsize=(20,5))



sns.distplot(x, ax=axs[0])

sns.distplot(x1, ax= axs[1], color = 'b')

sns.distplot(x2, ax= axs[2], color ='g')

sns.distplot(x3, ax= axs[3], color ='r')
fig, axs = plt.subplots(1,2, figsize=(11,5))



sns.countplot(x = 'animal', data = data_out, ax = axs[0]) 

sns.countplot(x = 'furniture', data = data_out, ax= axs[1])

plt.show()
data.columns.unique()
fig, axs = plt.subplots(1,2, figsize=(20,5))

#plt.figure(figsize=(10,6))

sns.scatterplot(x= 'area', y='rent amount (R$)', data = data_out, ax= axs[0])

sns.barplot(x= 'floor', y='rent amount (R$)', data = data_out, ax= axs[1])

plt.xticks(rotation=90)

#sns.scatterplot(x= 'rooms', y='rent amount (R$)', data = data_out)
plt.figure(figsize=(15,7))

sns.scatterplot( x='area' , y = 'rent amount (R$)', hue ='city', data = data_out)
plt.figure(figsize=(15,7))

sns.swarmplot(x='city',y='rent amount (R$)', hue= 'animal', data = data_out )

plt.figure(figsize=(15,7))

sns.violinplot(x='city',y='rent amount (R$)', hue= 'furniture', data = data_out)
df_sao = data_out.loc[(data_out.city =='São Paulo')]



df_sao
fig, axs = plt.subplots(ncols=4, figsize=(20,5))

sns.scatterplot(x= 'area', y ='property tax (R$)', data = df_sao, ax= axs[0])

sns.scatterplot(x='area', y='total (R$)', data = df_sao, ax=axs[1], color='c')

sns.scatterplot(x='area', y='hoa (R$)', data = df_sao, ax=axs[2], color='b')

sns.scatterplot(x='area', y='fire insurance (R$)', data = df_sao, ax=axs[3], color='g')
#sns.scatterplot(x='area', y='total (R$)', data = df_sao)
fig, axs = plt.subplots(ncols=3, figsize=(20,5))

sns.countplot(x = 'animal', data = df_sao, ax = axs[0])

sns.countplot(x = 'furniture', data = df_sao, ax = axs[1])

sns.countplot(x = 'floor', data = df_sao, ax = axs[2])

plt.xticks(rotation=90)

plt.show
plt.figure(figsize=(10,5))

sns.barplot(x='rooms', y='bathroom', data=df_sao)