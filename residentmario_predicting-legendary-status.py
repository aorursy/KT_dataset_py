import numpy as np

import pandas as pd

from sklearn import linear_model



pokemon = pd.read_csv('../input/Pokemon.csv')
pokemon.groupby('Legendary').apply(np.mean)
import seaborn as sns

%matplotlib inline



sns.boxplot(x='Legendary', y='Total', data=pokemon)
sns.swarmplot(x='Legendary', y='Total', data=pokemon.query('550 < Total < 800'))
pokemon_sorted = pokemon.sort_values(by='Total')



X = pokemon_sorted['Total'].values.reshape((800, 1))

y = pokemon_sorted['Legendary'].values.reshape((800, 1))

clf = linear_model.LogisticRegression(C=1e5)

clf.fit(X, y)
totals = X.flatten()

y_predicted = clf.predict(X).flatten()

y_actual = y.flatten()



prediction_correct = (y_actual == y_predicted)
import matplotlib.pyplot as plt



plt.figure(1, figsize=(8, 4))

plt.scatter(X, y_predicted, color=['steelblue' if p else 'darkred' for p in prediction_correct])



X_test = np.linspace(150, 900, 300)

decision_boundary = (1/(1 + np.exp(-(clf.intercept_ + X_test*clf.coef_)))).ravel()

plt.plot(X_test, decision_boundary, color='black')
from sklearn import metrics



metrics.confusion_matrix(y_actual, y_predicted)
clf = linear_model.LogisticRegression(C=1e5, class_weight={0: 0.2, 1: 1})

clf.fit(X, y)
totals = X.flatten()

y_predicted = clf.predict(X).flatten()

y_actual = y.flatten()



prediction_correct = (y_actual == y_predicted)



plt.figure(1, figsize=(8, 4))

plt.scatter(X, y_predicted, color=['steelblue' if p else 'darkred' for p in prediction_correct])



X_test = np.linspace(150, 900, 300)

decision_boundary = (1/(1 + np.exp(-(clf.intercept_ + X_test*clf.coef_)))).ravel()

plt.plot(X_test, decision_boundary, color='black')
metrics.confusion_matrix(y_actual, y_predicted)