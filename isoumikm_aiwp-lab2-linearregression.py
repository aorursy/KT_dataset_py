# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Submitted by: 16BIS0095 Soumik Mitra

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/heart.csv")

dataset.head()
x = dataset[['trestbps','chol','thalach']]

y = dataset['oldpeak']
lr = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1)
lr.fit(X_train,y_train)
y_predict = lr.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_predict)))