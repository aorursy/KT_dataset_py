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
dataset=pd.read_csv("../input/slaarydataforlinearregression/Salary.csv")
dataset
dataset.head()
dataset.head(3)
dataset.info()
dataset.describe()
dataset.isnull()
dataset.isnull().any()
x=dataset.iloc[:,:1]
x
y=dataset.iloc[:,1:2]
y
x=dataset.iloc[:,:1].values
print(x)
x
y=dataset.iloc[:,1:2].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train
x_test
y_train
y_test
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)
from sklearn.linear_model import LinearRegression
le=LinearRegression()
le.fit(x_train,y_train)
#y_pred
x_test
y_pred=le.predict(x_test)
y_pred
y_test
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
randomvalue = le.predict([[2.6]])
randomvalue
randomvalue = le.predict([[12.6]])
randomvalue
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,le.predict(x_train))
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,le.predict(x_train))

plt.scatter(x_test,y_test,color="red")

plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,le.predict(x_test))