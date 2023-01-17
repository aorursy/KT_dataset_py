# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv('../input/train.csv')

dft = pd.read_csv('../input/test.csv')
plt.scatter(range(df.shape[0]), np.sort(df.trip_duration.values))
q = df.trip_duration.quantile(0.99)

dfTemp = df

dfTemp = dfTemp[dfTemp.trip_duration < q]

plt.scatter(range(dfTemp.shape[0]), np.sort(dfTemp.trip_duration.values))
df = df[df.trip_duration < q]
from geopy.distance import great_circle

df['pickup'] = [(a,b) for a, b in zip(df.pickup_latitude, df.pickup_longitude)]

df['dropoff'] = [(a,b) for a, b in zip(df.dropoff_latitude, df.dropoff_longitude)]

df['kilometers'] = [great_circle(a,b).kilometers for a, b in zip(df.pickup, df.dropoff)]

#df['velocity'] = df.kilometers/(df.trip_duration/(3600))

df.head()
dft['pickup'] = [(a,b) for a, b in zip(dft.pickup_latitude, dft.pickup_longitude)]

dft['dropoff'] = [(a,b) for a, b in zip(dft.dropoff_latitude, dft.dropoff_longitude)]

dft['kilometers'] = [great_circle(a,b).kilometers for a, b in zip(dft.pickup, dft.dropoff)]

dft.head()
%%time

kmeans = KMeans(n_clusters=15, random_state=2).fit(df[['pickup_longitude','pickup_latitude']])

kmeans.cluster_centers_
kmeans = KMeans(n_clusters=15, random_state=2).fit(dft[['pickup_longitude','pickup_latitude']])

kmeans.cluster_centers_
df['cluster'] = kmeans.predict(df[['pickup_longitude','pickup_latitude']])

dft['cluster'] = kmeans.predict(dft[['pickup_longitude','pickup_latitude']])
df.pickup_datetime=pd.to_datetime(df.pickup_datetime)

df['hour'] = df.pickup_datetime.dt.hour

df['month'] = df.pickup_datetime.dt.month

df['day'] = df.pickup_datetime.dt.dayofweek
dft.pickup_datetime=pd.to_datetime(dft.pickup_datetime)

dft['hour'] = dft.pickup_datetime.dt.hour

dft['month'] = dft.pickup_datetime.dt.month

dft['day'] = dft.pickup_datetime.dt.dayofweek
X = df.as_matrix([['passenger_count', 'vendor_id', 'pickup_latitude', 'pickup_longitude',

                   'dropoff_latitude', 'dropoff_longitude',

                   'kilometers',  'hour', 'month', 'day', 'cluster']])

y = df.as_matrix([['trip_duration']])
y.shape
y.ravel(1444051,)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV 



xgb_reg = XGBRegressor()

clf = GridSearchCV(xgb_reg,

                  {'max_depth': [1,2,3]},

                  verbose=1)

clf.fit(X_train, y_train)



print(clf.best_score_)

print(clf.best_params_)
xgb_reg
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

xgb_reg = XGBRegressor(max_depth=5, min_child_weight=14)

xgb_reg.fit(X_train, y_train)

print(cross_val_score(xgb_reg, X_test, y_test))

print('mse for XGB = ', mean_squared_error(y_test, xgb_reg.predict(X_test)))
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

reg = DecisionTreeRegressor(max_depth=22, min_samples_leaf=5) #12 = 429

reg.fit(X_train, y_train)

print(cross_val_score(reg, X_test, y_test))
from sklearn import linear_model

from sklearn.model_selection import cross_val_score

linear_reg = linear_model.LinearRegression()

linear_reg.fit(X_train, y_train)

print(cross_val_score(linear_reg, X_test, y_test))
#from sklearn import linear_model

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

#from sklearn.model_selection import cross_val_score

alphas = np.array([1000, 100, 10, 1, 0.1, 0.01, 0.001])



model = Lasso()

grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))

grid.fit(X_train, y_train)

print(grid)



print(grid.best_score_)

print(grid.best_estimator_.alpha)
from sklearn import linear_model

#from sklearn.model_selection import cross_val_score

lasso_reg = linear_model.Lasso(alpha=10.0, copy_X=True, fit_intercept=True,

                              max_iter=50000, selection='random',

                              random_state=5)





lasso_reg.fit(X_train, y_train)

print(cross_val_score(lasso_reg, X_test, y_test))
y_linreg = linear_reg.predict(X_test)

y_tree = reg.predict(X_test)

y_lasso = lasso_reg.predict(X_test)

y_xgb = xgb_reg.predict(X_test)
from sklearn.metrics import mean_absolute_error

print('mae for linreg = ', mean_absolute_error(y_test, linear_reg.predict(X_test)))

print('mae for lassoreg = ', mean_absolute_error(y_test, lasso_reg.predict(X_test)))

print('mae for Tree = ', mean_absolute_error(y_test, reg.predict(X_test)))

print('mae for XGB = ', mean_absolute_error(y_test, xgb_reg.predict(X_test)))
#plt.plot(y_linreg[:50], label='linear')

#plt.plot(y_tree[:50], label='tree')

plt.plot(y_test[:100], label='Data')

#plt.plot(y_lasso[:50], label='lasso')

plt.plot(y_xgb[:100], label='XGB')

plt.legend()

plt.show()
tfeatures = dft.as_matrix([['passenger_count', 'vendor_id', 'pickup_latitude', 'pickup_longitude',

                   'dropoff_latitude', 'dropoff_longitude',

                   'kilometers',  'hour', 'month', 'day', 'cluster']])
#dft['trip_duration_linear'] = linear_reg.predict(tfeatures)

#dft['trip_duration_tree'] = reg.predict(tfeatures)

#dft['trip_duration_lasso'] = lasso_reg.predict(tfeatures)

dft['trip_duration'] = xgb_reg.predict(tfeatures)
dft.head(30)
out = dft[['id', 'trip_duration']]
out.to_csv('dsloet_submission.csv')
out.head()
X_train = X_train[:1000]

y_train = y_train[:1000]
from tpot import TPOTRegressor

auto_classifier = TPOTRegressor(generations=3, population_size=9, verbosity=2)

auto_classifier.fit(X_train, y_train)