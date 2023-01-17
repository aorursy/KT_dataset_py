# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pure_train_set = pd.read_csv("../input/random-linear-regression/train.csv")

pure_test_set = pd.read_csv("../input/random-linear-regression/test.csv")

train_set = pure_train_set.dropna()

test_set = pure_test_set.dropna()
print("Number of rows before cleaning: ",pure_train_set.size,"\n")

print("Number of rows after cleaning: ",train_set.size,"\n")

print("Number of NaN value in dataset : ",pure_train_set.size-train_set.size)
train_set.head()
train_x = train_set.x.values.reshape(-1,1)

train_y = train_set.y.values.reshape(-1,1)



test_x = test_set.x.values.reshape(-1,1)

test_y = test_set.y.values.reshape(-1,1)
plt.scatter(train_x, train_y,  color='black',label="Train_set")

plt.scatter(test_x, test_y,  color='blue',label = "Test_set")
model_linear_reg = LinearRegression()



model_linear_reg.fit(train_x,train_y)
plt.scatter(test_x, test_y,  color='black')



predict_y = model_linear_reg.predict(test_x)



plt.plot(predict_y,test_x,color = "red")

plt.show()