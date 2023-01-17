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



import pandas as pd

import numpy as np

import matplotlib.pyplot as py

from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

X=pd.read_csv('../input/random-linear-regression/test.csv')

ytest=pd.read_csv('../input/random-linear-regression/test.csv')

# x_train,x_test=train_test_split(X,test_size=0.2)



# print(x_train.shape,x_test.shape)

x1=(X).values

x1_train=x1[:,0].reshape(-1,1)

x2_train=x1[:,1]

x2=ytest.values

x1_test=x2[:,0].reshape(-1,1)

x2_test=x2[:,1]

# print(x1_train)

# print(x2_train)

# reg = LinearRegression.fit(x,y,sample_weight=None)

# plt.scatter(x,y)

model = LinearRegression()

model.fit(x1_train,x2_train)

model.score(x1_train,x2_train)

predict1=model.predict(x1_test)

# plt.scatter(x1_train,x2_train,color='red')

print('Coefficients: \n',model.coef_)

print('Mean Squared Error: \n',mean_squared_error(x2_test,predict1))



# plt.scatter(x1_test,x2_test,color='black')

# plt.plot(x1_test,predict1,color='blue', linewidth=3)

# plt.xticks()

# plt.yticks()

# plt.show()





fig, (ax1, ax2) = py.subplots(1, 2)

ax1.scatter(x1_train,x2_train,color='magenta')

ax1.set_title("Training Set")

ax2.scatter(x1_test,x2_test,color='yellow')

ax2.plot(x1_test,predict1,color='blue', linewidth=3)

ax2.set_title("Test Set")




