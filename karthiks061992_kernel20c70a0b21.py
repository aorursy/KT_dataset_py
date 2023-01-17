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
train=pd.read_csv("/kaggle/input/random-linear-regression/train.csv")

test=pd.read_csv("/kaggle/input/random-linear-regression/test.csv")

train.head()

test.head()
train.isnull().sum()

train.dropna(inplace=True)

test.dropna(inplace=True)

train.isnull().sum()

test.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sn

plt.figure(figsize=(14,6))

sn.regplot(x=train.x,y=train.y,color="orange")
plt.figure(figsize=(14,6))

sn.regplot(x=test.x,y=test.y,color="red")
train.head()

X=train.iloc[:,[0]]

Y=train.iloc[:,[1]]

X=np.array(X)

X=X.reshape(-1,1)

X=pd.DataFrame(X)

Y=np.array(Y)

Y=Y.reshape(-1,1)

Y=pd.DataFrame(Y)
xtest=test.iloc[:,[0]]

ytest=test.iloc[:,[1]]

xtest=np.array(xtest)

xtest=xtest.reshape(-1,1)

xtest=pd.DataFrame(xtest)

ytest=np.array(ytest)

ytest=ytest.reshape(-1,1)

ytest=pd.DataFrame(ytest)
print(X)
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X,Y)

ypred=regressor.predict(xtest)
#r2_score

from sklearn.metrics import r2_score

print(r2_score(ytest,ypred)*100)
from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=regressor,X=X,y=Y,cv=10)

print(accuracies)
print("accuracies:{:.2f}".format(accuracies.mean()*100))