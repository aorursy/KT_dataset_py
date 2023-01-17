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
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
df=pd.read_csv('../input/iris-data-1/iris.csv')
df.head()
x=df[['petal_length']]

y=df['sepal_length']

plt.scatter(x,y,color='blue')

plt.xlabel('length of petal')

plt.ylabel('length of sepal')

plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
x_train.head()
reg=LinearRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
y_test.head()
y_predict[0:5],y_test.head()
plt.scatter(x,y,color='blue')

plt.plot(x_test,y_predict,color='red')

plt.xlabel('length of petal')

plt.ylabel('length of sepal')

plt.show()
mean_squared_error(y_test,y_predict)
reg.intercept_,reg.coef_
y_test.head()
x_test.head()
reg.intercept_,reg.coef_
4.7*0.41998496+4.216915383615326
y_predict[0:1]