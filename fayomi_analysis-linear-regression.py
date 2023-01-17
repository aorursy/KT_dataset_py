# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy import stats #to call a function that removes anomalies



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/kc_house_data.csv')
df.head()
df.drop(['id','date','sqft_lot','sqft_above','lat', 'long','zipcode', 'sqft_living15', 'sqft_lot15','waterfront','view'],axis=1,inplace=True)
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)] #to remove anomalies

df.head()
df.info()
plt.figure(figsize=(16,6))

sns.distplot(df['price'],kde=False,bins=50)
plt.figure(figsize=(16,6))

sns.distplot(df['price'].dropna(),kde=False,bins=50)

plt.figure(figsize=(16,6))

sns.countplot(df['bedrooms'])
plt.figure(figsize=(16,6))

sns.countplot(df['bathrooms'])
plt.figure(figsize=(16,6))

sns.distplot(df['sqft_living'].dropna(),kde=False,bins=50)
sns.pairplot(df)
sns.jointplot(x='bedrooms',y='price',data=df)
sns.jointplot(x='price',y='sqft_living',data=df,kind='reg')
sns.jointplot(x='floors',y='price',data=df)
sns.jointplot(x='grade',y='price',data=df, kind='reg')
sns.jointplot(x='yr_built',y='price',data=df)
sns.jointplot(x='sqft_basement',y='price',data=df)
sns.jointplot(x='bathrooms',y='price',data=df, kind='reg')
sns.jointplot(x='condition',y='price',data=df)
sns.heatmap(df.corr(),cmap='coolwarm', annot=True)


df.columns
#selected inputs

x = df[['bathrooms','grade','sqft_living']]

#expected output

y = df['price']
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
#to train the data

lm.fit(x_train,y_train)
#to calculate teh coefficients

lm.coef_
#to create a table with the coefs

cdf = pd.DataFrame(lm.coef_,x.columns,columns=['coefs'])
cdf
#to get the predictions of test set

pred = lm.predict(x_test)
#to plot predictions and actual result

#This shows an accurate preditction

plt.scatter(y_test, pred)