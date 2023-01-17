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
#read train dataset
dataset = pd.read_csv('../input/train.csv')
dataset = dataset.dropna(axis=0)
X=dataset.iloc[:,:1].values
y=dataset.iloc[:,1].values
from sklearn.linear_model import LinearRegression
X_train_mod=X.reshape(-1,1)
y_train_mod=y.reshape(-1,1)
regressor=LinearRegression()
regressor.fit(X_train_mod,y_train_mod)
dataset2=pd.read_csv('../input/test.csv')
X_test=dataset2.iloc[:,:1].values
y_test=dataset2.iloc[:,1].values
X_test_mod=X_test.reshape(-1,1)
y_test_mod=y_test.reshape(-1,1)
y_pred=regressor.predict(X_test_mod)
import matplotlib.pyplot as plt
plt.scatter(X_test_mod,y_test_mod,color='red')
plt.plot(X_test_mod,y_pred, color='blue')
plt.title('X vs Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
from sklearn.metrics import accuracy_score
for i in range(len(X_test_mod)):
    print('predicted value is:',y_pred[i],'actual value is:',y_test_mod[i],'and error is',y_test_mod[i]-y_pred[i],'\n')
