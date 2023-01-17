import pandas as pd

df = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')

df.head()
import numpy as np 

import matplotlib.pyplot as plt
df.describe()
x = df[['YearsExperience','Salary']]

x.head()
viz = x[['YearsExperience','Salary']]

viz.hist()

plt.show()
plt.scatter(x.YearsExperience,x.Salary, color = 'red')

plt.xlabel('YearsExperience')

plt.ylabel('Salary')

plt.show()
msk = np.random.rand(len(df))>0.8

train = x[msk]

test  = x[~msk]
from sklearn import linear_model

regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['YearsExperience']])

train_y = np.asanyarray(train[['Salary']])

regr.fit(train_x,train_y)

print('Coeffecient',regr.coef_)

print('Intercept',regr.intercept_)
plt.scatter(train.YearsExperience,train.Salary,color = 'green')

plt.plot(train_x,regr.coef_[0][0]*train_x+regr.intercept_[0],'r')

plt.xlabel('YearsExperience')

plt.ylabel('Salary')

plt.show()
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['YearsExperience']])

text_y = np.asanyarray(test[['Salary']])

test_y = regr.predict(test_x)

print('mean absolute error:%.2f'%np.mean(np.absolute(test_y-test_y)))

print('R2-score:%2f'%r2_score(test_y,test_y))