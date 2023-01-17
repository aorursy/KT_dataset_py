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

import matplotlib as plt

import numpy as np 

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score 

from sklearn.metrics import mean_squared_error 

from sklearn.metrics import mean_absolute_error

%matplotlib inline 



df = pd.read_csv('//kaggle//input//salary-data//Salary_Data.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
print(df.corr())
df.plot.scatter(x='YearsExperience' ,y = 'Salary')
sns.regplot('YearsExperience' , 'Salary',data = df)
train,test  = train_test_split(df,test_size= 0.2,random_state = 142)
print("train shape ",train.shape)

print("test shape ",test.shape)
X_train = train.drop(['Salary'], axis = 1)

Y_train = train["Salary"]

X_test = test.drop(['Salary'], axis = 1)

Y_test = test["Salary"]

print("X_train shape :",X_train.shape)

print("y_train shape :",Y_train.shape)

print("X_testshape :",X_test.shape)

print("y_test shape :",Y_test.shape)





lr = LinearRegression()

lr.fit(X_train,Y_train)
print("Coefficents :",lr.coef_)

print("Intercept :",lr.intercept_)
y_pred = lr.predict(X_test)

y_pred
print("MSME:",mean_squared_error(Y_test,y_pred))

print("RMSE:",np.sqrt(mean_squared_error(Y_test,y_pred)))

print("MAE:",mean_absolute_error(Y_test,y_pred))

print("r2 SCORE:",r2_score(Y_test,y_pred))
y_pred_train = lr.predict(X_train)

print("MSME:",mean_squared_error(Y_test,y_pred))

print("RMSE:",np.sqrt(mean_squared_error(Y_test,y_pred)))

print("MAE:",mean_absolute_error(Y_test,y_pred))

print("r2 SCORE:",r2_score(Y_test,y_pred))