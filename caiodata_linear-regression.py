# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv", index_col=0)

data.head()
data.head(100).info()
data.columns
# filter variables

new_columns = ['Overall', 'Potential', 'Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle','Release Clause']
data[new_columns].head(500).info()
data = data[new_columns].head(500)

data.dropna(inplace=True)

data.info()
data.head()
data['Release Clause'].head()
data['Release Clause'].tail()
## convert to float dtype

data['Release Clause'] = data['Release Clause'].str.strip('€M').astype(float)

data['Release Clause'].tail()
data.info()
data.corr().iloc[-1]
data.corr().iloc[-1][:-1].max()
data.corr().iloc[-1][:-1].min()
import seaborn as sns

import matplotlib.pylab as plt



fig, ax = plt.subplots(figsize=(14,10))         # Sample figsize in inches

sns.heatmap(data.corr(),  linewidths=.5, ax=ax, square=True, vmin=-1, vmax=1)

plt.show()
data.plot.scatter(x='Overall', y='Release Clause')
data.plot.scatter(x='Potential', y='Release Clause')
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = data['Potential']

y = data['Release Clause']
model.fit(X.values.reshape(-1,1), y)
model.predict(X.values.reshape(-1,1))
data.plot.scatter(x='Potential', y='Release Clause', figsize=[14,10])

plt.plot(X , model.predict(X.values.reshape(-1,1)), c='r')
model.coef_
model.intercept_
f = lambda x : model.intercept_ + model.coef_*x
f(85)
y_predict = model.predict(X.values.reshape(-1,1))
y[0]
y_predict[0]
y[0] - y_predict[0]
(y[0] - y_predict[0])**2
y[1] - y_predict[1]
(y[1] - y_predict[1])**2
(y - y_predict)[:2]
((y - y_predict)**2)[:2]
((y - y_predict)**2)
((y - y_predict)**2).sum()
ss_r = ((y - y_predict)**2).sum()
ss_t = ((y - y.mean())**2).sum()
r_squared = 1 - (ss_r / ss_t)

r_squared
model.score(X.values.reshape(-1,1), y)
data.iloc[:,:-1].head()
new_X = data.iloc[:,:-1]
np.dot(np.linalg.inv(np.dot(new_X.values.T, new_X.values)), np.dot(new_X.T, y))
np.dot(np.linalg.pinv(new_X.values), y)
np.linalg.lstsq(new_X, y)
## using scikit



model_2 = LinearRegression()
model_2.fit(new_X, y)
model_2.score(new_X, y)
model_2.predict(new_X)
model_2.intercept_ + (model_2.coef_ * new_X.iloc[0].values).sum()
model_2.intercept_
for i in range(10):

    print((new_X.iloc[i].values * np.dot(np.linalg.pinv(new_X.head(31).values), y.head(31))).sum(), y[i])
_x = np.array([0, 1, 2, 3, 4, 5])



_y = np.array([0, .8, .9, .1, -.8, -1])
p1 = np.polyfit(_x, _y, 1)

p2 = np.polyfit(_x, _y, 2)

p3 = np.polyfit(_x, _y, 3)

p4 = np.polyfit(_x, _y, 4)
plt.figure(figsize=(14,10))

plt.plot(_x, _y, 'o')

_x = np.linspace(-2,6, 100)

plt.plot(_x, np.polyval(p1, _x), 'r-')

plt.plot(_x, np.polyval(p2, _x), 'b--')

plt.plot(_x, np.polyval(p3, _x), 'm:')

plt.plot(_x, np.polyval(p4, _x))

plt.legend(['dado',1,2,3,4])
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(new_X, y, random_state=1)
# .fit()

X_train.shape
# .score()

X_test.shape
new_X.shape
model_3 = LinearRegression()

scores = cross_val_score(model_3, new_X, y, cv=4, scoring="mean_squared_error")
scores
sklearn.metrics.SCORERS.keys()
import sklearn