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
#iporting the datasets and and converting into a dataframe

dataset1=pd.read_csv("../input/random-linear-regression/train.csv")

dataset2=pd.read_csv("../input/random-linear-regression/test.csv")
dataset1.head()
dataset2.head()
#to check if there is any missing value in the training set

dataset1.isnull().sum()
#there is one missing value in y so we can just drop it using dropna

dataset1.dropna(inplace=True)

#to check if there is any missing value in the test set

dataset2.isnull().sum()


dataset1.describe()
#now to check if there is a linear relationship between x and y

%matplotlib inline

import matplotlib.pyplot as plt
plt.scatter(dataset1.x,dataset1.y,edgecolor="orange")

plt.xlabel("value of x")

plt.ylabel("value of y")

plt.title("relationship between x and y")

plt.show()
#from the graph we can say that there is a strong linear relationship between x and y 

#so now we can apply the linear model

X_train = dataset1.x.values.reshape(-1,1)

y_train = dataset1.y.values.reshape(-1,1)

X_test  = dataset2.x.values.reshape(-1,1)

y_test  = dataset2.y.values.reshape(-1,1)
from sklearn.linear_model import LinearRegression



logmodel=LinearRegression()

logmodel.fit(X_train,y_train)

y_pred_train=logmodel.predict(X_train)
#plotting the y_pred values against the X_train

plt.scatter(X_train,y_train)

plt.plot(X_train,y_pred_train,color="orange",linewidth=3)

plt.title("train_data_prediction")

plt.show()
#let us now check the accuracy of the training set

logmodel.score(X_train,y_train)
#let us check the accuracy of the test set

logmodel.score(X_test,y_test)