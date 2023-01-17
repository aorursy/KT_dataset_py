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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pandas import Series,DataFrame

import seaborn as sns

data=pd.read_csv("../input/grade-prediction/data1.csv")
data.head()
data.describe()

data.info()
data.isnull().any()
plt.subplots(figsize=(10,15))

grade_counts = data['CGPA'].value_counts().sort_values().plot.barh(width=.9,color=sns.color_palette('inferno',50))

grade_counts.axes.set_title('Number of students who scored a particular grade',fontsize=30)

grade_counts.set_xlabel('Number of students', fontsize=30)

grade_counts.set_ylabel('CGPA', fontsize=30)

plt.show()
b = sns.countplot(data['address'])

b.axes.set_title('Urban and rural students', fontsize = 30)

b.set_xlabel('Address', fontsize = 20)

b.set_ylabel('Count', fontsize = 20)

plt.show()


X= data.iloc[:,4:5].values

Y= data.iloc[:,5:6].values
#Splitting the data

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size= 1/3)



#Fitting Simple Linear Regression ipynb

#This is called Model 

from sklearn.linear_model import LinearRegression

regressor= LinearRegression()

regressor.fit(X_train,Y_train)
##Predicting the test results

Y_pred= regressor.predict(X_test)



#Visualising the training set Results



plt.scatter(X_train, Y_train, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title('Student Grade Prediction)')

plt.xlabel('SGPA')

plt.ylabel('CGPA')

plt.show()

plt.scatter(X_test, Y_test, color='red')

plt.plot(X_test, regressor.predict(X_test), color='blue')

plt.title('Student Grade Prediction(Test set)')

plt.xlabel('SGPA')

plt.ylabel('CGPA')

plt.show()
print(regressor.predict([[1002]]))
a=float(input("What is SGPA "))

print('The CGPA  is', regressor.predict([[a]]))
regressor.intercept_
regressor.coef_
from sklearn import metrics
r2score=metrics.r2_score(Y_test,Y_pred) #frequency score

r2score
regressor.score(X,Y)