 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/bike-sharing-demand/train.csv')

df_test = pd.read_csv('../input/bike-sharing-demand/test.csv')
df.head()
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
grid = sns.FacetGrid(df, col = 'season', row = 'holiday')

grid.map(plt.scatter, 'temp', 'casual', alpha = 0.5)

grid.add_legend();
plt.scatter(x = df['windspeed'], y = df['count'], alpha= .5)

plt.show()
df[['count', 'holiday']].groupby(['holiday'], as_index = False).mean().sort_values(by = 'count')
df[['count', 'workingday']].groupby(['workingday'], as_index = False).mean().sort_values(by = 'count')
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn import model_selection, preprocessing
df.isnull().sum()
df.datetime = df.datetime.apply(pd.to_datetime)

df['month'] = df.datetime.apply(lambda x: x.month)

df['hour'] = df.datetime.apply(lambda x: x.hour)

df['day'] = df.datetime.apply(lambda x: x.day)

#df['year'] = df.datetime.apply(lambda x: x.year)

df.drop(['datetime'], 1, inplace = True)

df.head()
plt.scatter(x = df['casual'] + df['registered'], y = df['count'])

plt.show()

X = np.array(df.drop(df[['casual', 'registered', 'count']], 1))

y = np.array(df['count'])



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
def rmsle(y, y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
clf = LinearRegression(normalize= True)

clf.fit(X_train, y_train)

print("The accuracy of linear regression is: ", clf.score(X_test, y_test), "\nThe RMSLE is: ", rmsle(y_test, clf.predict(X_test)))
clf = DecisionTreeRegressor()

clf.fit(X_train, y_train)

print("The accuracy of Decision Tree Regressor is: ", clf.score(X_test, y_test), "\nThe RMSLE is: ", rmsle(y_test, clf.predict(X_test)))
clf = RandomForestRegressor(n_estimators= 200)

clf.fit(X_train, y_train)

print('The accuracy of Random Forest Regressor is: ', clf.score(X_test, y_test), "\nThe RMSLE is: ", rmsle(y_test, clf.predict(X_test)))