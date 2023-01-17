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
import matplotlib.pyplot as plt

d=pd.read_csv('../input/random-salary-data-of-employes-age-wise/Salary_Data.csv')
x=d.iloc[:,:-1].values

y=d.iloc[:,1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=0)



from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


plt.scatter(x_train,y_train,color="red")

plt.plot(x_train,lr.predict(x_train),color="blue")

plt.title("salary vs experience(training set)")

plt.xlabel("year of experience")

plt.ylabel("salary")

plt.show()


plt.scatter(x_test,y_test,color="red")

plt.plot(x_test,lr.predict(x_test),color="blue")

plt.title("salary vs experience(test set)")

plt.xlabel("year of experience")

plt.ylabel("salary")

plt.show()
from sklearn.metrics import mean_absolute_error

a=mean_absolute_error(y_test, y_pred)

print(a)
from sklearn.metrics import mean_squared_error

b=mean_squared_error(y_test, y_pred)

print(b)

print(np.sqrt(b))