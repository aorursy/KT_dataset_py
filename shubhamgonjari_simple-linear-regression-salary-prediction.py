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
# Reading the Dataset

data=pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')

data.head()
%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Scatter plot shows the Linear Relationship between Dependent Variable 'YearsExperiance' and independent variable 'Salary' 

sns.set()

sns.scatterplot(x='YearsExperience',y='Salary',data=data)
from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(data.YearsExperience,data.Salary)
'''What Is R-squared?

R-squared is a statistical measure of how close the data are to the fitted regression line. 

It is also known as the coefficient of determination, or the coefficient of multiple 

determination for multiple regression.'''

r_value ** 2
# Visualizing how line fits the data

def predict(x):

    return slope * x + intercept



fitLine = predict(data.YearsExperience)



plt.scatter(data.YearsExperience, data.Salary)

plt.plot(data.YearsExperience, fitLine, c='r')

plt.title('Salary VS YearsExperience')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
from sklearn.model_selection import train_test_split

# Creating Dependent and Independent Variables 

X=data['YearsExperience']

y=data['Salary']
# import the Linear Regression Model from sklearn 

from sklearn.linear_model import LinearRegression

lr=LinearRegression()



#Splitting Data into train test split for model Evaluation

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.30,random_state=2018)



import numpy as np

X_train=X_train[:,np.newaxis]

X_test=X_test[:,np.newaxis]
# Train model on training set 

lr.fit(X_train,y_train)
# Make Predictions on Testing Set

y_pred=lr.predict(X_test)
from sklearn.metrics import mean_squared_error,r2_score

print('Mean Sqared Error :',mean_squared_error(y_test,y_pred))

print('R-Square Value :',r2_score(y_test,y_pred))
print('Intercept :',lr.intercept_)

print('Coefficient :',lr.coef_)
plt.scatter(X_train,y_train,color='r')

plt.plot(X_train,lr.predict(X_train),color='green')

plt.title('Salary VS Years of Experience(Training set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
plt.scatter(X_test,y_test,color='r')

plt.plot(X_train,lr.predict(X_train),color='green')

plt.title('Salary VS Years of Experience(Testing set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()