# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

data.info()
data.head()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
fig = plt.figure(figsize = (18,6))

sns.barplot(x = 'quality', y = 'alcohol', data = data)

plt.title("Effect of alcohol on wine quality")

plt.show()
fig = plt.figure(figsize = (18,6))

sns.barplot(x = 'quality', y = 'sulphates', data = data)

plt.title("Effect of sulphates on wine quality")

plt.show()
fig = plt.figure(figsize = (18,6))

sns.barplot(x = 'quality', y = 'citric acid', data = data)

plt.title("Effect of citric acid on wine quality")

plt.show()
fig = plt.figure(figsize = (18,6))

sns.barplot(x = 'quality', y = 'volatile acidity', data = data)

plt.title("Effect of volatile acidity on wine quality")

plt.show()
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

x=data.quality.values.reshape(-1,1)

y=data.alcohol.values.reshape(-1,1)

x.shape,y.shape
lr.fit(x,y)
array=np.arange(15).reshape(-1,1)

y_head=lr.predict(array)
plt.scatter(x,y)

plt.show()
plt.figure(figsize=(10,10))

plt.scatter(x,y)

plt.plot(array,y_head)

plt.show()
y2=data.density.values.reshape(-1,1)
lr.fit(y,y2)
array2=np.arange(8,16).reshape(-1,1)

yy_head=lr.predict(array2)
plt.figure(figsize=(10,10))

plt.scatter(y,y2)

plt.plot(array2,yy_head)

plt.show()