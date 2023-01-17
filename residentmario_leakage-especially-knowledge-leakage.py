import numpy as np

import pandas as pd

import scipy.stats as st



np.random.seed(0)

df = np.random.randint(0,10,size=[100,10000])

y = np.random.randint(0,2,size=100)

df = pd.DataFrame(df)

X = df.values



corr = np.abs(

    np.array([st.pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])

)

corrmax_indices = np.argpartition(np.abs(corr), -2)[-2:]



X_selected = X[:, corrmax_indices]



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



clf = LogisticRegression()

clf.fit(X_selected, y)

mse = cross_val_score(clf, X_selected, y, cv=10, scoring='neg_mean_squared_error')



pd.Series(mse).abs().mean()
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)



corr = np.abs(

    np.array([st.pearsonr(X_train[:, i], y_train)[0] for i in range(df.shape[1])])

)

corrmax_indices = np.argpartition(np.abs(corr), -2)[-2:]

X_selected = X_train[:, corrmax_indices]



clf = LogisticRegression()

clf.fit(X_selected, y_train)

y_hat = clf.predict(X_test[:, corrmax_indices])

mean_squared_error(y_test, y_hat)
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import StratifiedKFold



kf = StratifiedKFold(n_splits=10)



mse_results = []



for train_index, test_index in kf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]



    corr = np.abs(

        np.array([st.pearsonr(X_train[:, i], y_train)[0] for i in range(df.shape[1])])

    )

    corrmax_indices = np.argpartition(np.abs(corr[:-1]), -2)[-2:]



    X_train_selected = X_train[:, corrmax_indices]



    clf = LogisticRegression()

    clf.fit(X_train_selected, y_train)

    mse = mean_squared_error(clf.predict(X_test[:, corrmax_indices]), y_test)

    mse_results.append(mse)

    

mse = pd.Series(mse).mean()

mse