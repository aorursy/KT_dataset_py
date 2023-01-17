# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv")
#assigning the dependent and independent variables

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,1:2].values

#checking the null values in the dataset

dataset.isnull().sum().sort_values(ascending=False)
#dataset.plot.scatter(y='Salary',x='YearsExperience')

sns.regplot(y="Salary", x="YearsExperience", data=dataset);
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
#fitting simple linear regression to the training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
y_pred
# visualising the training set results

plt.scatter(X_train, y_train,color ='red')

plt.plot(X_train, regressor.predict(X_train),color = 'green')

plt.title('Salary Vs Experience (Training set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
# visualising the test set results

plt.scatter(X_test, y_test,color ='red')

plt.plot(X_train, regressor.predict(X_train),color = 'blue')

plt.title('Salary Vs Experience (Test set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()