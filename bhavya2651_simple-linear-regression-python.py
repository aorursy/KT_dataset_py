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
#import library
import matplotlib.pyplot as plt
#import dataset
dataset=pd.read_csv("../input/linear-regression-dataset/Linear Regression - Sheet1.csv")
#read first 5 entries of datset
dataset.head()
#data preprocessing
# 1) getting info of data
dataset.info()
#there are no missing values
#statistical analysis of data
dataset.describe()
#separating input and output variables
x=dataset[["X"]]
y=dataset[['Y']]

#SPLIT DATSET INTRO TRAINING AND TESTING
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=47)
# create model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
#predict the test set result
y_pred=lr.predict(x_test)
#visualize training set result
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,lr.predict(x_train),color='blue')
plt.title("LINEAR REGRESSION")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#visualize test set
plt.scatter(x_test,y_pred,color='red')
plt.plot(x_train,lr.predict(x_train),color='blue')
plt.title("LINEAR REGRESSION")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#create linear regression equation
print(lr.coef_)
print(lr.intercept_)
# y=0.64904167+5.11351805x