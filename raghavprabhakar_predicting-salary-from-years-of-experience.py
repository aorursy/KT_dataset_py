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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')

data.head()
plt.scatter(data['YearsExperience'], data['Salary'], color = 'red')

plt.xlabel('YearsExperience')

plt.ylabel('Salary')

plt.title('Prediction of Salary from years of experience')
def normalize(x):

    min = x.min()

    max = x.max()

    y = (x-min)/(max-min)

    return y
data_final = normalize(data)
X = pd.DataFrame(data_final['YearsExperience'])

y = pd.DataFrame(data_final['Salary'])
X = data_final['YearsExperience'].values.reshape(-1,1)

y = data_final['Salary'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 30)
lr = LinearRegression()

lr.fit(X_train, y_train)
print(lr.intercept_)

print(lr.coef_)
y_pred = lr.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df
plt.plot(X_test, y_pred, color='black', linewidth=2, label='Predicted Values')

plt.scatter(X_test, y_test,  color='red', label='Test Values')

plt.scatter(X_train,y_train, color='grey', label='Training Values')

plt.xlabel('YearsExperience')

plt.ylabel('Salary')

plt.title('Prediction of Salary from years of experience')

leg = plt.legend();

plt.show()
mse=mean_squared_error(y_test,y_pred)

r2_square = r2_score(y_test,y_pred)
print('Mean Squared Error : ',mse)

print('R2_Square : ',r2_square)

print('Accurarcy of model : ',r2_square*100)