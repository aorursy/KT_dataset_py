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
import matplotlib.pyplot as plt

import seaborn as seabornInstance

import matplotlib.cm as cm

from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn import svm, tree

import time

%matplotlib inline
heart = pd.read_csv("../input/heart-disease-uci/heart.csv")

heart.shape
heart.describe()
heart.plot(x='age',y='cp',style='o')

plt.title('age VS cp')

plt.xlabel('age')

plt.ylabel('cp')

plt.show()
plt.figure(figsize=(5,5))

plt.tight_layout()

seabornInstance.distplot(heart['cp'])
a=heart['age'].values.reshape(-1,1)

b=heart['cp'].values.reshape(-1,1)
print(type(heart))

a=heart.iloc[0:5000,1:]

b=heart.iloc[0:5000,:1]

train_a, test_a, train_b, test_b=train_test_split(a,b,test_size=0.2,random_state=0)
print(train_a.shape,train_b.shape)

print(test_a.shape,test_b.shape)
lm=linear_model.LinearRegression()

model=lm.fit(train_a,train_b)

predictions=lm.predict(test_a)

predictions[0:10]
plt.scatter(test_b, predictions)
model.score(test_a,test_b)