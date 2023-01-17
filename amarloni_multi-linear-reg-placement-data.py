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

%matplotlib inline

import seaborn as sns

plt.style.use('ggplot')
df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

df.head()
df.columns
df.isna().any()
df = df.fillna(0)
df1 = df.copy()
df.info()
X  = df.drop(['sl_no', 'gender', 'ssc_b', 'hsc_b', 'hsc_s',

       'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p',

       'status', 'salary'], axis = 1)

y= df['mba_p']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression

classifier = LinearRegression()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df
from sklearn import metrics

from sklearn.metrics import r2_score



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred).round(3))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred).round(3))  

print('Root Mean Squared:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))

print('r2_score:', r2_score(y_test, y_pred).round(3))
X  = df1.drop(['sl_no', 'gender', 'ssc_b', 'hsc_b', 'hsc_s','degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p','status', 'salary'], axis = 1)

y= df1['mba_p']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

classifier = LinearRegression()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

result.head()
from sklearn import metrics

from sklearn.metrics import r2_score

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred).round(3))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred).round(3))  

print('Root Mean Squared:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))

print('r2_score:', r2_score(y_test, y_pred).round(3))