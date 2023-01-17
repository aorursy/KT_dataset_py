# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

plt.style.use('seaborn-whitegrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/insurance.csv")
train_data.head()
fig = plt.figure()

plt.scatter(train_data['age'], train_data['charges']);

plt.show()
model = LinearRegression()
X = pd.DataFrame(train_data['age'])

Y = train_data['charges']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33, random_state=42)
model.fit(X_train, Y_train)
model.score(X_test, Y_test)