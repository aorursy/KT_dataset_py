# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

import statsmodels.formula.api as smf

import folium

from folium.plugins import HeatMap



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
x = np.linspace(-5,5,10)

y = 2*x+1

plt.plot(x, y, '-r', label='y=2x+1')

plt.title('Graph of y=2x+1')

plt.xlabel('x', color='#1C2833')

plt.ylabel('y', color='#1C2833')

plt.legend(loc='upper left')

plt.grid()

plt.show()
df=pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")

df.head(5)
df.columns
df['view'].value_counts()
print("The maximum price of a house is {}".format(df['price'].max()))

print("The minimum price of a house is {}".format(df['price'].min()))

print("The mean price of houses is {}".format(df['price'].mean()))

print("The median price of houses is {}".format(df['price'].median()))
print(df['yr_built'].max())

print(df['yr_built'].min())
df['age']=2015-df['yr_built']
print(df['yr_renovated'].value_counts())

print(df['yr_renovated'].max())

print(df['yr_renovated'].min())
df['renovation']=2015-df['yr_renovated']
df.drop(columns=["id",'date','yr_built','yr_renovated'], inplace=True)

df.describe()
df.isna().sum()
df.corr()
def correlation_heatmap(df1):

    _, ax = plt.subplots(figsize = (15, 10))

    colormap= sns.diverging_palette(220, 10, as_cmap = True)

    sns.heatmap(df.corr(), annot=True, cmap = colormap)



correlation_heatmap(df)  
g = sns.pairplot(df[['sqft_above','price','sqft_living','bedrooms', 'age']], 

                 hue='bedrooms', palette='husl',height=6,)
sns.set(style="whitegrid", font_scale=1)

f, axes = plt.subplots(1, 2,figsize=(18,5))

sns.boxplot(x=df['bedrooms'],y=df['price'], ax=axes[0])

sns.boxplot(x=df['floors'],y=df['price'], ax=axes[1])

sns.despine(left=True, bottom=True)

axes[0].set(xlabel='Bedrooms', ylabel='Price')

axes[0].yaxis.tick_left()

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].set(xlabel='Floors', ylabel='Price')



f, axe = plt.subplots(1, 1,figsize=(18,5))

sns.despine(left=True, bottom=True)

sns.boxplot(x=df['bathrooms'],y=df['price'], ax=axe)

axe.yaxis.tick_left()

axe.set(xlabel='Bathrooms', ylabel='Price')



f, axes = plt.subplots(1, 2,figsize=(18,5))

sns.boxplot(x=df['waterfront'],y=df['price'], ax=axes[0])

sns.boxplot(x=df['view'],y=df['price'], ax=axes[1])

sns.despine(left=True, bottom=True)

axes[0].set(xlabel='waterfront', ylabel='Price')

axes[0].yaxis.tick_left()

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].set(xlabel='view', ylabel='Price')



f, axes = plt.subplots(1, 2,figsize=(18,5))

sns.boxplot(x=df['condition'],y=df['price'], ax=axes[0])

sns.boxplot(x=df['grade'],y=df['price'], ax=axes[1])

sns.despine(left=True, bottom=True)

axes[0].set(xlabel='condition', ylabel='Price')

axes[0].yaxis.tick_left()

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].set(xlabel='grade', ylabel='Price')
f, ax = plt.subplots(figsize=(16, 12))

sns.regplot(x="price", y="sqft_lot", data=df)
f, ax = plt.subplots(figsize=(16, 12))

sns.regplot(x="price", y="sqft_lot", data=df.iloc[0:100,:])
feature_cols = ['sqft_living']

X = df[feature_cols]

y = df.price



reg = LinearRegression()

hp=reg.fit(X, y)



print(hp.intercept_)

print(hp.coef_)
hp.score(X, y)
feature_cols1 = ['sqft_living15', 'sqft_above', 'grade', 'sqft_living', 'bathrooms']

X = df[feature_cols1]

y = df.price



hp1 = LinearRegression()

hp1.fit(X, y)



print(hp1.intercept_)

print(hp1.coef_)

hp1.score(X, y)
feature_cols2 = ['sqft_living15', 'sqft_above', 'grade', 'sqft_living', 'bathrooms','lat', 'sqft_basement', 'view' , 'waterfront', 'floors', 'bedrooms']

X = df[feature_cols2]

y = df.price



hp1 = LinearRegression()

hp1.fit(X, y)





print(hp1.intercept_)

print(hp1.coef_)

hp1.score(X, y)
feature_cols3 = ['sqft_living15', 'sqft_above', 'grade', 'sqft_living', 'bathrooms','lat', 'sqft_basement', 'view' , 'waterfront', 'floors', 'bedrooms', 'age']

X = df[feature_cols3]

y = df.price





hp1 = LinearRegression()

hp1.fit(X, y)



print(hp1.intercept_)

print(hp1.coef_)

hp1.score(X, y)
from sklearn.model_selection import train_test_split

train_data,test_data = train_test_split(df,train_size = 0.7,random_state=0)



reg = linear_model.LinearRegression()



X_train = np.array(train_data['sqft_living'], dtype=pd.Series).reshape(-1, 1)

Y_train = np.array(train_data['price'], dtype=pd.Series).reshape(-1,1)

X_test=np.array(test_data['sqft_living'], dtype=pd.Series).reshape(-1, 1)

Y_test=np.array(test_data['price'], dtype=pd.Series).reshape(-1, 1)



reg.fit(X_train,Y_train)
LinearRegression()

print(reg.coef_,reg.intercept_)