import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, f_classif

from matplotlib import cm



df = pd.read_csv('../input/creditcard.csv')

df.head()
X = df[df.columns[:len(df.columns) - 1]]

y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y)



np.set_printoptions(precision=3)

test = SelectKBest(score_func=f_classif, k=4)

fit = test.fit(X_train, y_train)

for property, value in vars(fit).items():

    print(property, ": ", value)

features = fit.transform(X)

X_train = features[0:5,:]

print(X_train)
cmap = cm.get_cmap('gnuplot')

scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)