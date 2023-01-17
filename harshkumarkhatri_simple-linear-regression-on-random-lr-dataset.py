# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# This is the first problem which i slved from the kaggle dataset.
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the data from the data set
data_test=pd.read_csv("/kaggle/input/random-linear-regression/train.csv")
data_test2=pd.read_csv("/kaggle/input/random-linear-regression/test.csv")

# print("shape is ")
# print(np.shape(data_test))
# data_train_x=np.array(data_test['x'])
# data_train_y=np.array(data_test['y'])

# Finding the inappropriate value
print("the value is inappropriate")
print(data_test[data_test['x']>500])
print('illocing')
data_test.iloc[213]
data_test.drop(213,axis=0,inplace=True)

# data_train_x=data_train_x.reshape(1,-1)
# print(data_train_x)
# data_train_y=data_train_y.reshape(1,-1)
# print(data_train_y)
# print(data_train_y)
# print(np.shape(data_train_y))
# print(data_train_x)
# print(data_test.head())
# print(data_test['x'].head())
# plt.scatter(data_test['x'],data_test['y'])
# plt.xlabel("x coord")
# plt.ylabel("Y coord")
# plt.show()

# Defining the model
reg=linear_model.LinearRegression()

# data_test['x'].reshape(1,-1)
# data_test['x']=np.array(data_test['x'])
# print(data_test['x'])
# np.array(data_test['y'])
# data_test['x'].reshape(1,-1)
# print("shape is ")
# print(np.shape(data_train_x))
# data_train_x=data_train_x[:,np.newaxis,2]

# Fitting the model
reg.fit(data_test[['x']],data_test[['y']])
print("Score is ")
print(reg.score(data_test[['x']],data_test[['y']]))

# For predictions
y_prdict=reg.predict(data_test2[['x']])
print(np.shape(y_prdict))

# Plotting the output
plt.scatter(data_test['x'],data_test['y'])
plt.plot(data_test2[['x']],y_prdict,'r',linewidth=2)
plt.xlabel("x coord")
plt.ylabel("Y coord")
plt.show()

# Printing the value of coefficients
print("coeff is ")
print(reg.coef_)

# Printing the value of intercepts
print("Intercept is ")
print(reg.intercept_)
