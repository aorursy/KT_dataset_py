# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import sklearn.utils

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.metrics import mean_squared_error

from scipy.stats.mstats import zscore

from sklearn.learning_curve import learning_curve

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



plt.style.use('ggplot')

%matplotlib inline
store = pd.read_csv('../input/store.csv').dropna()

store.shape
store.head()
store.tail()
sns.pairplot(store)
train = pd.read_csv('../input/train.csv')

state_hol = pd.get_dummies(train[['StateHoliday']].replace(0, '0'), prefix='StateHoliday')

train = pd.concat([train, state_hol], axis=1).drop(['StateHoliday'], axis=1)
train.head()
train.tail()
train = train[train.Sales != 0]

train.shape
sns.distplot(train[['Sales']].dropna())
avg_per_store = train[['Sales', 'Store']].groupby('Store').mean()

avg_per_store.reset_index().plot(kind='scatter', x='Store', y='Sales')
avg_per_weekday = train[['Sales', 'DayOfWeek']].groupby('DayOfWeek').mean()

avg_per_weekday.reset_index().plot(kind='bar', x='DayOfWeek', y='Sales')
avg_hist_by_month = train[['Sales', 'Customers', 'Promo']].groupby(['Promo']).mean()

sns.barplot(x="DayOfWeek", y="Sales", hue="Promo", order=[0, 1, 2, 3, 4, 5, 6], data=train)
train['MonthDay'] = train['Date'].map(lambda x: x[8:])



avg_hist_by_month = train[['Sales', 'Customers', 'MonthDay']].groupby('MonthDay').mean()

avg_hist_by_month.plot(kind='bar')
train[['Customers', 'Sales']].plot(kind='scatter', x='Customers', y='Sales')
np.log(train[['Customers', 'Sales']]).plot(kind='scatter', x='Customers', y='Sales')
train = train[train.Sales != 0]

train.shape
train[['Customers', 'Sales']].plot(kind='scatter', x='Customers', y='Sales')
np.log(train[['Customers', 'Sales']]).plot(kind='scatter', x='Customers', y='Sales')
avg_promotion = train[['Sales', 'Customers', 'Promo']].groupby('Promo').mean()

avg_promotion.plot(kind='bar')
avg_stateholiday = train[['Sales', 'Customers', 'StateHoliday_0']].groupby('StateHoliday_0').mean()

avg_stateholiday.plot(kind='bar')
test = pd.read_csv('../input/test.csv').fillna(0)

state_hol = pd.get_dummies(test[['StateHoliday']].replace(0, '0'), prefix='StateHoliday')

test = pd.concat([test, state_hol], axis=1).drop(['StateHoliday'], axis=1)

test.shape
test.head()
test.tail()
X = train[['Store','DayOfWeek','Open','Promo', 'SchoolHoliday', \

           'StateHoliday_0', 'StateHoliday_a']].values

y = np.ravel(np.array(train[['Sales']]))



#print y.shape



X_TEST = test[['Store','DayOfWeek','Open', 'Promo', 'SchoolHoliday', 'StateHoliday_0', 'StateHoliday_a']].values

#print type(X)

#print y

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



#parameters = {'max_depth':[4, 50, 100]}

rf = RandomForestRegressor(n_estimators=300)#linear_model.LinearRegression()

#regr = GridSearchCV(rf, parameters)

#print X_TEST



rf.fit(X, y)

#y_pred = list(rf.predict(X))#[:,np.newaxis]

y_pred_TEST = list(int(e) for e in rf.predict(X_TEST))
Id = [i for i in range(1, len(y_pred_TEST) + 1)]

res = pd.DataFrame(np.matrix([Id, y_pred_TEST]).transpose(), columns=['Id', 'Sales'])

res.to_csv('res.csv', index=False)