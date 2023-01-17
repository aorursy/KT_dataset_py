# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

data=pd.read_csv('../input/Father_Son_height_C.csv')
data.head()
data.info()
X=data['Father'].values[:,None]

X.shape
y=data.iloc[:,2].values

y.shape
from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(X_train,y_train)
y_test=lm.predict(X_test)

print(y_test)
plt.scatter(X,y,color='b')

plt.plot(X_test,y_test,color='black',linewidth=3)

plt.xlabel('Father height in inches')

plt.ylabel('Son height in inches')

plt.show()
y_train_pred=lm.predict(X_train).ravel()

y_test_pred=lm.predict(X_test).ravel()
from sklearn.metrics import mean_squared_error as mse,r2_score
print("The Mean Squared Error on Train set is:\t{:0.1f}".format(mse(y_train,y_train_pred)))

print("The Mean Squared Error on Test set is:\t{:0.1f}".format(mse(y_test,y_test_pred)))
print("The R2 score on the Train set is:\t{:0.1f}".format(r2_score(y_train,y_train_pred)))

print("The R2 score on the Test set is:\t{:0.1f}".format(r2_score(y_test,y_test_pred)))