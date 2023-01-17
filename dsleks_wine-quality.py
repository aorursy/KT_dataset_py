!pip install scipy==1.2 --upgrade
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer 

import statsmodels.api as sm
sns.set(color_codes=True)

wines = pd.read_csv("../input/winequality.csv")

wines.info()
wines.head()
wines.describe()
high_rated_wines = wines.loc[wines['quality'] == 8]

high_rated_wines.describe()
sns.pairplot(high_rated_wines)
for feat in high_rated_wines.columns.drop('quality') :

    print(feat)

    print(high_rated_wines[feat].min())

    print(high_rated_wines[feat].max())
wines.isnull().any().any()
transformed = Normalizer().transform(wines).reshape(1599,12)

transformed_df = pd.DataFrame(transformed, columns = wines.columns)

transformed_df.describe()
q2 ='quality'

Y = transformed_df.loc[:,'quality']

for q1 in transformed_df.columns :

    plt.figure()

    sns.regplot(x = q1, y = q2, data = transformed_df, color='green')

    plt.xlabel(q1)

    plt.ylabel(q2)
    for f in ['fixedacidity', 'totalsulfurdioxide', 'sulphates', 'alcohol', 'pH'] :

        X = transformed_df.loc[:, f]

        X = sm.add_constant(X)

        model = sm.OLS(Y, X).fit()

        print(f, "vs Quality")

        print(model.summary())

        print("\n\n\n")
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import statistics
Y = transformed_df.quality

X = transformed_df.drop('quality', axis = 1)
scores = []

r2 = []

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X,Y) :

    x_train, x_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    RegModel = LinearRegression()

    RegModel.fit(x_train, y_train)

    y_pred = RegModel.predict(x_test)

    scores.append(metrics.mean_squared_error(y_test,y_pred))

    r2.append(metrics.r2_score(y_test,y_pred))

print(statistics.mean(scores))

print(statistics.mean(r2))

features = ['fixedacidity', 'totalsulfurdioxide', 'sulphates', 'alcohol', 'pH']

Y = transformed_df.quality

X = transformed_df[features]
feat_scores = []

feat_r2 = []

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X,Y) :

    x_train, x_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    RegModel = LinearRegression()

    RegModel.fit(x_train, y_train)

    y_pred = RegModel.predict(x_test)

    feat_scores.append(metrics.mean_squared_error(y_test,y_pred))

    feat_r2.append(metrics.r2_score(y_test, y_pred))

print(statistics.mean(feat_scores))

print(statistics.mean(feat_r2))
from sklearn.model_selection import train_test_split

import pylab
for i in range(0,10) :

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

    reg = LinearRegression()

    reg.fit(X_train,y_train)

    pred_vals = reg.predict(X_test)

    true_vals = y_test

    residuals = true_vals - pred_vals

    sm.qqplot(residuals, line='s')

    pylab.show()

    print("\n")
from sklearn.tree import DecisionTreeRegressor
Y = transformed_df.quality

X = transformed_df.drop('quality', axis = 1)
tree_scores = []

tree_r2 = []

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X,Y) :

    x_train, x_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    RegTree = DecisionTreeRegressor()

    RegTree.fit(x_train, y_train)

    y_pred = RegTree.predict(x_test)

    tree_scores.append(metrics.mean_squared_error(y_test,y_pred))

    tree_r2.append(RegTree.score(x_test, y_test))

print(statistics.mean(feat_scores))

print(statistics.mean(feat_r2))
from sklearn.ensemble import RandomForestRegressor
Y = transformed_df.quality

X = transformed_df.drop('quality', axis = 1)
ranfores_scores = []

ranfores_r2 = []

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X,Y) :

    x_train, x_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    RanForest = RandomForestRegressor(n_estimators=100)

    RanForest.fit(x_train, y_train)

    y_pred = RanForest.predict(x_test)

    ranfores_scores.append(metrics.mean_squared_error(y_test,y_pred))

    ranfores_r2.append(RegTree.score(x_test, y_test))

print(statistics.mean(ranfores_scores))

print(statistics.mean(ranfores_r2))
features = ['fixedacidity', 'totalsulfurdioxide', 'sulphates', 'alcohol', 'pH']

Y1 = transformed_df.quality

X1 = transformed_df[features]
pranfores_scores = []

pranfores_r2 = []

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X1,Y1) :

    x_train, x_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    pRanForest = RandomForestRegressor(n_estimators=100)

    pRanForest.fit(x_train, y_train)

    y_pred = RanForest.predict(x_test)

    pranfores_scores.append(metrics.mean_squared_error(y_test,y_pred))

    pranfores_r2.append(RegTree.score(x_test, y_test))

print(statistics.mean(pranfores_scores))

print(statistics.mean(pranfores_r2))