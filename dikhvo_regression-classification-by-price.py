import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt



plt.rc('figure', figsize=(15, 12))



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read the data

dmds = pd.read_csv('../input/diamonds.csv')

dmds.drop('Unnamed: 0', axis=1, inplace=True)

dmds.head(3)
# encode cut, color and clarity

categorical_cols = ['cut', 'color', 'clarity']

for c in categorical_cols:

    dmds[c] = pd.factorize(dmds[c])[0]

dmds.head(3)
# graphs

# use a reduced set without categorical columns

dmds_reduced = dmds[dmds.columns.difference(['cut', 'color', 'clarity'])]



# correlations

plt.matshow(dmds_reduced.corr())

plt.colorbar()

tick_marks = dmds_reduced.columns.values

plt.xticks(np.arange(tick_marks.size), tick_marks)

plt.yticks(np.arange(tick_marks.size), tick_marks)

plt.show()
# let's compare the correlation visually

from pandas.tools.plotting import scatter_matrix



scatter_matrix(dmds_reduced, diagonal='kde')

plt.grid()

plt.show()
# split

from sklearn.model_selection import train_test_split



X, y = dmds.iloc[:, dmds.columns != 'price'].values, dmds.iloc[:, dmds.columns == 'price'].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor



models = [('LR', LinearRegression(n_jobs=-1)),

          ('RF', RandomForestRegressor(n_estimators=100, criterion='mse', random_state=1, n_jobs=-1)),

#           ('SVR-lin', SVR(kernel='linear', C=1e3))

#           ('SVR-rbf', SVR(kernel='rbf', C=1e3)),

#           ('SVR-poly', SVR(kernel='poly', C=1e3, degree=2))

         ]
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import cross_val_score, KFold



# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=5, random_state=123)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, n_jobs=-1)    

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# compute ms-error and R^2

for name, model in models:

    model.fit(X_train, y_train)



    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)



    print('%s: MSE train: %.4f, test: %.4f' % (name, mean_squared_error(y_train, y_train_pred),

                                           mean_squared_error(y_test, y_test_pred)))

    print('%s: R^2 train: %.4f, test: %.4f' % (name, r2_score(y_train, y_train_pred),

                                           r2_score(y_test, y_test_pred)))
# Compare Algorithms

fig = plt.figure(figsize=(16, 8))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results, vert=False)

ax.set_yticklabels(names)

plt.grid()

plt.show()
# convert price data to classes

n_classes = 10



y_classes = np.linspace(0, y.max(), n_classes)

y_train_cl = np.digitize(y_train, bins=y_classes)

y_test_cl = np.digitize(y_test, bins=y_classes)



print('Price classes: %s' % (y_classes))
# test different models on the data

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.neural_network import MLPClassifier



models = [('LR', LogisticRegression()),

          ('KNN', KNeighborsClassifier()),

          ('CART', DecisionTreeClassifier()),

          ('NB', GaussianNB()),

          ('SVM-lin', SVC(kernel='linear')),

          ('SVM-rbf', SVC(kernel='rbf')),

          ('RF', RandomForestClassifier()),

          ('MLP', MLPClassifier(alpha=1)),

          ('ADA', AdaBoostClassifier())]
# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=5, random_state=42)

    cv_results = cross_val_score(model, X_train, y_train_cl, cv=kfold, n_jobs=-1)    

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Compare Algorithms

fig = plt.figure(figsize=(16, 8))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results, vert=False)

ax.set_yticklabels(names)

plt.grid()

plt.show()
# convert price data to classes

n_classes = 5



y_classes = np.linspace(0, y.max(), n_classes)

y_train_cl = np.digitize(y_train, bins=y_classes)

y_test_cl = np.digitize(y_test, bins=y_classes)



print('Price classes: %s' % (y_classes))
# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=5, random_state=42)

    cv_results = cross_val_score(model, X_train, y_train_cl, cv=kfold, n_jobs=-1)    

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Compare Algorithms

fig = plt.figure(figsize=(16, 8))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results, vert=False)

ax.set_yticklabels(names)

plt.grid()

plt.show()