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
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')

df.head(5)
df['sex'] = df['sex'].map({'female': 1, 'male': 0})

df.head(5)
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

df.head(5)
df['region'] = df['region'].map({'southwest':1,'southeast':2,'northwest':3,'northeast':4})

df.region.unique()
df.head(4)
features = [

    u'age', u'sex', u'bmi', u'children', u'smoker', u'region',

]

x = df[features]

y = df['charges']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=3)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)
accuracy = regressor.score(x_test, y_test)

"Accuracy: {}%".format(int(round(accuracy * 100)))
#To retrieve the intercept:

print(regressor.intercept_)

#For retrieving the slope:

print(regressor.coef_)
y_pred = regressor.predict(x_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1 = df1.head(30)

print(df1)
import matplotlib.pyplot as plt

df1.plot(kind='bar',figsize=(20,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()