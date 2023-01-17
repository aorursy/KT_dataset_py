# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/input.csv')
df.columns
df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')

df.head()
df.info()
numtrain = df.select_dtypes(include=[np.number])

corr = numtrain.corr()

plt.figure(figsize=(26,26))

sns.heatmap(corr,vmax=1,square=True,annot=True)
corr = df.corr().stack().abs()

corr = corr[corr.index.get_level_values(0) != corr.index.get_level_values(1)]

corr.sort_values(ascending=False)

#plt(hst, corr)
del df['NG']

del df['YYYYMMDD']

del df['TXH']
plt.plot(df['SQ'])
df.shape
X=df.values[:,0:36]

y=df.values[:,36]
print('Shape of X = ', X.shape)

print('Shape of y = ', y.shape)
y= y/10
from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 100)

from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(max_depth=8)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
plt.plot(y_pred[:50], label = 'prediction')

plt.plot(y_test[:50], label = 'data')

plt.legend()

plt.show()
from sklearn.cross_validation import cross_val_score

scores = -cross_val_score(reg, X_test, y_test, scoring='neg_mean_absolute_error', cv=1000)

scores.mean()
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, reg.predict(X_test))
X_today = [[220,20,27,50,9,10,21,90,9,181,126,3,99,6,64,41,1827,0,-1,-1,4,10132,10146,22,10113,3,67,24,83,3,74,97,23,56,15,32]]
y_today = reg.predict(X_today)

y_today
df = pd.read_csv('../input/input.csv')

df = df[['FG', 'TG', 'SQ', 'SP', 'Q', 'RH', 'PG', 'UG', 'EV24', 'TX']]

df.head()
df.shape
X=df.values[:,0:9]

y=df.values[:,9]
y = y/10
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 100)
from sklearn.tree import DecisionTreeRegressor

reg_lessF = DecisionTreeRegressor(max_depth=8)
reg_lessF.fit(X_train, y_train)
y_pred_lessF = reg_lessF.predict(X_test)
plt.plot(y_pred_lessF[:75], label = 'prediction')

plt.plot(y_test[:75], label = 'data')

plt.legend()

plt.show()
from sklearn.cross_validation import cross_val_score

scores = -cross_val_score(reg_lessF, X_test, y_test, scoring='neg_mean_absolute_error', cv=1000)

scores.mean()
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, reg_lessF.predict(X_test))