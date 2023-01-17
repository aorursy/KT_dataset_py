# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
file = pd.read_csv('/kaggle/input/weight-height/weight-height.csv')
file.head()
sns.pairplot(file)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
gnd = pd.get_dummies(file['Gender'],drop_first=True)
gnd.head()
file.drop('Gender',axis=1,inplace=True)
file = pd.concat([file,gnd],axis=1)
file.shape
X = file[['Height','Male']]

y = file['Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=432)
lmodel = LinearRegression()

lmodel.fit(X_train,y_train)
X_test.head()
pred = lmodel.predict(X_test)
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, pred))

print('MSE:', metrics.mean_squared_error(y_test, pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
plt.scatter(X_test['Height'],y_test,color = 'blue',alpha='0.3')

plt.scatter(X_test['Height'],pred,color = 'red',alpha='0.1')
file.head()
df2 = pd.DataFrame(columns=file.columns)    
df2.head()
df2 = df2.drop('Weight',axis=1)
df2 =df2.append({'Height':70.07874,'Male': 0},ignore_index = True)
print(lmodel.predict(df2)[0])