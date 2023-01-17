import pandas as pd

import numpy as np

foods = pd.read_csv("../input/FAO database.csv", encoding='latin1')

pd.set_option('max_column', None)

usa_foods = foods.query('Area == "United States of America"')

usa_foods.head(3)
from sklearn.preprocessing import normalize



X = usa_foods.loc[:, ['Y{0}'.format(y) for y in range(1961, 2014)]].replace(0, np.nan)

X = X.dropna(thresh=30)

X = X.fillna(method='backfill')

X = normalize(X)



# Remove some erronous entries.

X = X[(np.max(X, axis=1) - np.min(X, axis=1)) / np.min(X, axis=1) < 10**2, :]
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



fig = plt.figure(figsize=(12, 4))

for n in np.random.choice(range(X.shape[1]), size=20):

    plt.plot(X[n, :])

    

plt.title('US Food Consumption, Normalized, 20 Categories')

pass
X, y = X[:, :-1], X[:, -1:]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
from sklearn.linear_model import LinearRegression



clf = LinearRegression()

clf.fit(X_train, y_train)

y_test_hat = clf.predict(X_test)



from sklearn.metrics import median_absolute_error

median_absolute_error(y_test_hat, y_test)
median_absolute_error(clf.predict(X_train), y_train)
from sklearn.model_selection import KFold



kf = KFold(n_splits=4)



scores = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    scores.append(

        median_absolute_error(clf.fit(X_train, y_train).predict(X_test), y_test)

    )

    

np.mean(scores)
from sklearn.model_selection import RepeatedKFold



rkf = RepeatedKFold(n_splits=4, n_repeats=2)



scores = []

for train_index, test_index in rkf.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    scores.append(

        median_absolute_error(clf.fit(X_train, y_train).predict(X_test), y_test)

    )

    

np.mean(scores)
from sklearn.model_selection import LeaveOneOut



loo = LeaveOneOut()



scores = []

for train_index, test_index in loo.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    scores.append(

        median_absolute_error(clf.fit(X_train, y_train).predict(X_test), y_test)

    )

    

scores = np.array(scores)

np.mean(scores)
from sklearn.model_selection import LeavePOut



lpo = LeavePOut(p=2)



scores = []

for train_index, test_index in lpo.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    scores.append(

        median_absolute_error(clf.fit(X_train, y_train).predict(X_test), y_test)

    )

    

scores = np.array(scores)

np.mean(scores)
from sklearn.model_selection import StratifiedKFold



skf = StratifiedKFold(n_splits=4)



scores = []



t = (np.random.random(size=100) > 0.1).astype(int)[:, np.newaxis]

for train_index, test_index in skf.split(

    np.sort((np.random.random(size=100) > 0.1)).astype(int), 

    np.sort(np.random.random(size=100) > 0.1).astype(int)

):

    X_train, X_test = t[train_index], t[test_index]

    y_train, y_test = t[train_index], t[test_index]

    

    scores.append(

        median_absolute_error(clf.fit(X_train, y_train).predict(X_test), y_test)

    )

    

scores = np.array(scores)

np.mean(scores)