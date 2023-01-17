# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import seaborn as sns

sns.set(style='whitegrid')

import matplotlib.pyplot as plt

from matplotlib import style

#sta matplotlib to inline and displays graphs below the corresponding cell.

%matplotlib inline

import os

from sklearn.datasets import *

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Automobile.csv")

df.head()
df.shape
#data cleaning

df.isnull().sum()
df.describe().T
df.info()
df.columns
cols=['symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration',

       'number_of_doors', 'body_style', 'drive_wheels', 'engine_location',

       'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_type',

       'number_of_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke',

       'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg',

       'highway_mpg', 'price']

for i in cols:

    def check(data):

        t=data[i].loc[data[i]=='?']

        return t

    



    g=check(df)

    print(g)
#what are the columns which are object

obj=list(df.select_dtypes(include=['object']))

obj
#what are the columns which are float and int

flint=list(df.select_dtypes(include=['int64','float64']))

flint
#checking for outliers

plt.figure(figsize=(15,8))

sns.boxplot(data=df)
df[['engine_size','peak_rpm','curb_weight','horsepower','price']].hist(figsize=(10,8),bins=6,color='Y')

plt.tight_layout()

plt.show()
print('the minimum price of car: %0.2d, the maximum price of the car: %0.2d'%(df['price'].min(),df['price'].max()))
df['make'][df['price']>=30000].count()
d=df['make'][df['price']>=30000].value_counts().count()

print(d)

df['make'][df['price']>=30000].value_counts()
df.aspiration.value_counts()
fig,a=plt.subplots(1,2,figsize=(10,5))

df.groupby('aspiration')['price'].agg(['mean','median','max']).plot.bar(rot=0,ax=a[0])

df.aspiration.value_counts().plot.bar(rot=0,ax=a[1])
plt.figure(1)

plt.subplot(221)

df['engine_type'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='red')

plt.title("Number of Engine Type frequency diagram")

plt.ylabel('Number of Engine Type')

plt.xlabel('engine-type');



plt.subplot(222)

df['body_style'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='orange')

plt.title("Number of Body Style frequency diagram")

plt.ylabel('Number of vehicles')

plt.xlabel('body-style');



plt.subplot(223)

df['number_of_doors'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='green')

plt.title("Number of Door frequency diagram")

plt.ylabel('Number of Doors')

plt.xlabel('num-of-doors');



plt.subplot(224)

df['fuel_type'].value_counts(normalize= True).plot(figsize=(10,8),kind='bar',color='purple')

plt.title("Number of Fuel Type frequency diagram")

plt.ylabel('Number of vehicles')

plt.xlabel('fuel-type');





plt.tight_layout()

plt.subplots_adjust(wspace=0.3,hspace=0.5)

plt.show()
fig,a=plt.subplots(1,2,figsize=(10,2))

df.body_style.value_counts().plot.pie(explode=(0.03,0,0,0,0),autopct='%0.2f%%',figsize=(10,5),ax=a[0])

a[0].set_title('No. of cars sold')





df.groupby('body_style')['price'].agg(['mean','median','max']).sort_values(by='median',ascending=False).plot.bar(ax=a[1])

a[1].set_title('Price of each body_style')

plt.tight_layout()

plt.show()
sns.catplot(data=df, y="normalized_losses", x="symboling"  ,kind="point")
sns.lmplot('engine_size','highway_mpg',hue='make',data=df,fit_reg=False)

plt.title('Engine_size Vs highway_mpg')

plt.show()

sns.lmplot('engine_size','city_mpg',hue='make',data=df,fit_reg=False)

plt.title('Engine_size Vs city_mpg')
df[['make','fuel_type','aspiration','number_of_doors','body_style','drive_wheels','engine_location']][df['engine_size']>=300]
fig,ax=plt.subplots(2,1,figsize=(15,5))

sns.countplot(x='drive_wheels',data=df,ax=ax[0])

df.groupby(['drive_wheels','make'])['price'].mean().plot.bar(ax=ax[1])

plt.grid()

plt.show()



fig,ax=plt.subplots(2,1,figsize=(15,5))

sns.countplot(x='drive_wheels',data=df,ax=ax[0])

df.groupby(['drive_wheels','body_style'])['price'].mean().plot.bar(ax=ax[1])

ax[1].set_ylabel('Price')

plt.grid()

plt.show()
dff=pd.pivot_table(df,index=['body_style'],columns=['drive_wheels'],values=['engine_size'],

                   aggfunc=['mean'],fill_value=0)



dff.plot.bar(figsize=(15,5),rot=45)

plt.show()

dff
sns.kdeplot(df['price'],shade=True)
df.price.max()
df.groupby('drive_wheels')[['city_mpg','highway_mpg']].agg('sum').plot.bar()

df.groupby('drive_wheels')[['city_mpg','highway_mpg']].agg('sum')
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(),annot=True)
sns.pairplot(df,aspect=1.5)
sns.pairplot(data=df,x_vars=['city_mpg','highway_mpg','horsepower','engine_size','curb_weight'],y_vars=['price'],kind='reg',size=2.8)
sns.pairplot(data=df,x_vars=['city_mpg','highway_mpg'],y_vars=['horsepower'],hue='price',size=4)
sns.pairplot(data=df,x_vars=['city_mpg','highway_mpg'],y_vars=['horsepower'],kind='reg',size=4)