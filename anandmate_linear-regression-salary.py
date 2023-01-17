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
import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
salary=pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv")
salary.head()
salary.dtypes
salary.isnull().sum()
salary.corr()
salary.info()
salary.describe()
sns.regplot(x="YearsExperience",y="Salary",data=salary)

plt.show()
X=salary.iloc[:,:-1].values

y=salary.iloc[:,1:2].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
#Creating and fitting model

linereg=LinearRegression()
linereg.fit(X_train,y_train)
linereg.coef_
linereg.intercept_
# Prediction

y_predict=linereg.predict(X_test)
y_predict
# Lets see what was y_test

y_test
plt.scatter(X_train,y_train,color='red')

plt.plot(X_train,linereg.predict(X_train),color='green')

plt.title("Years Experience Vs Salary(Train Data)")

plt.xlabel("Years Experience")

plt.ylabel("Salary")

plt.show()
plt.scatter(X_test,y_predict,color='red')

plt.plot(X_train,linereg.predict(X_train),color='blue')

plt.title('Years Experience Vs Salary (Test Data)')

plt.xlabel('Years Experience')

plt.ylabel('Salary')

plt.show()
print('Mean Abs Error:          ',metrics.mean_absolute_error(y_test,y_predict))

print('Mean Squared Error:      ',metrics.mean_squared_error(y_test,y_predict))

print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))

print('R squared value:         ',metrics.r2_score(y_test,y_predict))