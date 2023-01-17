# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import datetime

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn')



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from sklearn.model_selection import StratifiedKFold

from sklearn.manifold import TSNE



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
X = pd.read_csv('/kaggle/input/predicting-real-time-delay-status-part-1/X.csv', index_col=0)

y = pd.read_csv('/kaggle/input/predicting-real-time-delay-status-part-1/y.csv', index_col=0)
col_to_keep = ['last_status','avg_station_same', 'avg_station_opp', 'avg_sys']

pair_plot_df = X[col_to_keep]

sns.pairplot(pair_plot_df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)
X = []

y = []
standardize = StandardScaler()

X_train = standardize.fit_transform(X_train)
X_test = standardize.transform(X_test)
pca = PCA(n_components=51)

pca.fit(X_train)

pc_vs_variance = np.cumsum(pca.explained_variance_ratio_)



fig, ax = plt.subplots(figsize=[8, 8])

plt.plot(pc_vs_variance)

ax.set_xlabel('Num of components')

ax.set_ylabel('Cummulative variance')

plt.show()
print(pc_vs_variance[46])

print('According to the plot, n = 47 should be enough to capture 100% of variance')
pca = PCA(n_components=47)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print('mean squared error on train sets:', mean_squared_error(y_train, lr.predict(X_train)))

print('mean squared error on test sets:', mean_squared_error(y_test, y_pred))

print('r2 on train sets:', r2_score(y_train, lr.predict(X_train)))

print('r2 on test sets:', r2_score(y_test, y_pred))
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_train_np = np.array(y_train)
cvscores = []



for train, test in kfold.split(X_train, y_train_np):

  model = Sequential()

  model.add(Dense(40, input_dim=47, kernel_initializer='normal', activation='relu'))

  model.add(Dense(40, kernel_initializer='normal', activation='relu'))

  model.add(Dense(40, kernel_initializer='normal', activation='relu'))

  model.add(Dense(1, kernel_initializer='normal'))    

  # Compile model

  model.compile(loss='mse', optimizer='adam')

  print('-------------')

  # Fit the model

  model.fit(X_train[train], y_train_np[train], epochs=10, batch_size=128)

  print('-------------')

  # evaluate the model

  scores = model.evaluate(X_train[test], y_train_np[test])

  cvscores.append(scores)
print('loss: %.4f' % np.mean(cvscores), '(+/-%.3f)' % np.std(cvscores))
cvscores2 = []



for train, test in kfold.split(X_train, y_train_np):

  model = Sequential()

  model.add(Dense(40, input_dim=47, kernel_initializer='normal', activation='relu'))

  model.add(Dense(40, kernel_initializer='normal', activation='relu'))

  model.add(Dense(1, kernel_initializer='normal'))    

  # Compile model

  model.compile(loss='mse', optimizer='adam')

  # Fit the model

  print('-------------')

  model.fit(X_train[train], y_train_np[train], epochs=10, batch_size=128)

	# evaluate the model

  print('-------------')

  scores = model.evaluate(X_train[test], y_train_np[test])

  cvscores2.append(scores)
print('loss: %.4f' % np.mean(cvscores2), '(+/-%.3f)' % np.std(cvscores2))
cvscores3 = []



for train, test in kfold.split(X_train, y_train_np):

  model = Sequential()

  model.add(Dense(30, input_dim=47, kernel_initializer='normal', activation='relu'))

  model.add(Dense(30, kernel_initializer='normal', activation='relu'))

  model.add(Dense(1, kernel_initializer='normal'))    

  # Compile model

  model.compile(loss='mse', optimizer='adam')

  # Fit the model

  print('-------------')

  model.fit(X_train[train], y_train_np[train], epochs=10, batch_size=128)

	# evaluate the model

  print('-------------')

  scores = model.evaluate(X_train[test], y_train_np[test])

  cvscores3.append(scores)
print('loss: %.4f' % np.mean(cvscores3), '(+/-%.3f)' % np.std(cvscores3))
model = Sequential()

model.add(Dense(40, input_dim=47, kernel_initializer='normal', activation='relu'))

model.add(Dense(40, kernel_initializer='normal', activation='relu'))

model.add(Dense(1, kernel_initializer='normal'))    

# Compile model

model.compile(loss='mse', optimizer='adam')

# Fit the model

model.fit(X_train, y_train_np, epochs=10, batch_size=128)
y_pred = model.predict(X_test)

y_pred_train = model.predict(X_train)



print('mean squared error on train sets:', mean_squared_error(y_train, y_pred_train))

print('mean squared error on test sets:', mean_squared_error(y_test, y_pred))

print('r2 on train sets:', r2_score(y_train, y_pred_train))

print('r2 on test sets:', r2_score(y_test, y_pred))