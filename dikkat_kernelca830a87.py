# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/HeartDisease.csv")
data.head()
# Get the number of missing data points per column. This will show up in variable explorer

missing_values_count = data.isnull().sum()

print(missing_values_count)


#Taking care of missing values

data = data.dropna()
data.tail()
df=data.groupby('Sex').mean()

df
b=df.drop(['ID','Age','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak'],axis=1)

b
from matplotlib import pyplot as plt
plt.figure(figsize=(16, 8))

b.plot(kind='bar')
c=df.drop(['ID','Age','cp','trestbps','fbs','restecg','thalach','exang','oldpeak','num'],axis=1)

c
plt.figure(figsize=(10,5))

c.plot(kind='bar')
d=df.drop(['ID','Age','cp','trestbps','chol','restecg','thalach','exang','oldpeak','num'],axis=1)

d
plt.figure(figsize=(10,5))

d.plot(kind='bar')
data.describe()
#Plot only the values of num- the value to be predicted/Label

data["num"].value_counts().sort_index().plot.bar()
a=data['Age'].value_counts()

a
plt.figure(figsize=(20,10))

a.plot(kind='bar')
import seaborn as sb
plt.figure(figsize=(20,10))

sb.barplot(x='Age', y='chol', data=data)
plt.figure(figsize=(20,10))

sb.barplot(x='Age', y='num', data=data)
#Heat map to see the coreelation between variables, use annot if you want to see the values in the heatmap

plt.subplots(figsize=(12,8))

sb.heatmap(data.corr(),robust=True,annot=True)
#Detect outliers

plt.subplots(figsize=(15,6))

data.boxplot(patch_artist=True, sym="k.")
data.info()
data.columns
#linear model

X=data[['chol', 'Age', 'cp', 'fbs']]

y=data['num']
#training set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

X_test.tail()

#random_state-> same random splits
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
#coefficient

print(reg.intercept_)
reg.coef_
X_train.columns
m=pd.DataFrame(reg.coef_,X.columns, columns=['Coeff'])

m

predictions=reg.predict(X_test)
predictions
y_test.head()
plt.scatter(y_test, predictions)
reg.score(X_test,y_test)