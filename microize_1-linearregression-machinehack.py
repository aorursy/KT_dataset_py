# importing Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Reading the Libraries

df=pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')

display(df.head())

print('Dataframe has {} rows and {} columns.'.format(df.shape[0],df.shape[1]))
#relationnship between salary and years of experience

df.plot.scatter('YearsExperience','Salary')
#independent and dependennt variables

X=df[['YearsExperience']]

y=df[['Salary']]



#splitting the X and y

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

print('Number of rows in trainig data: {}'.format(X_train.shape[0]))

print('Number of rows in test data: {}'.format(X_test.shape[0]))
# building Linear model

lr=LinearRegression()

lr.fit(X_train,y_train)

predict_test=lr.predict(X_test)

y_predict=lr.predict(X)

y_predict
# mean squared error

err2=mean_squared_error(y,y_predict)

print(np.sqrt(err2))
# actual vs predicted value

plt.scatter(X,y,s = 70, label='Actual',alpha=0.70)

plt.scatter(X,y_predict,marker='D',s = 70, label='Predicted',alpha=0.70)

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.title("Salary VS Years of Experience")

plt.legend()

plt.show()