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
from sklearn.datasets import load_boston#import the dataset from sklearn for linear regression
data1=load_boston()
print(data1.data.shape)
print(data1.feature_names)#these are features of the model this means input to the model
print(data1.target)#these are actual y values this means output to the model
print(data1.DESCR) #this imformation about the dataset
import pandas as pd
bos = pd.DataFrame(data1.data)
print(bos)
bos['PRICE'] = data1.target
X = bos.iloc[:,:13]
Y = bos.iloc[:,13]
print(X)#input to the model
print(Y) #output of the model
#if you want bulid a model we need train data and test data so we need to divide the dataset in to train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#now train and test data is ready then we need build a model
from sklearn.linear_model import LinearRegression
s=LinearRegression()
s.fit(x_train,y_train)
y_pred=s.predict(x_test)
print(y_pred)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_pred,sample_weight=None, multioutput='uniform_average', squared=True)
print(np.sqrt(mse))#here error is very high so we need to minimize the error 
#when error is less then only the line is best fit line.


