



import pandas as pd

import sklearn

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from matplotlib import pyplot as plt







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')
df.info()
df.head()
y = df[['Salary']]

x = df[['YearsExperience']]
plt.scatter(x,y,alpha=0.4)

plt.title('Salary vs Experience')

plt.xlabel('Experience in Years',fontsize=15)

plt.ylabel('Salary in ₹',fontsize=15)



plt.show()
x_train, x_test, y_train, y_test = train_test_split(x,y)

x_train.shape

x_test.shape

y_train.shape

y_test.shape
model = LinearRegression()

model.fit(x_train,y_train)
model.coef_ 
model.intercept_ 
y_predicted = model.predict(x_test)

dfy = pd.DataFrame(y_predicted)

dfy
plt.scatter(x,y,alpha=0.4)

plt.xlabel('Experience in Years',fontsize=15)

plt.ylabel('Salary in ₹',fontsize=15)



plt.plot(x_test,y_predicted)



plt.show()
print(metrics.r2_score(y_test,y_predicted))