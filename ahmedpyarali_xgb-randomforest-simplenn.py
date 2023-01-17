%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

import xgboost as xgb

from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.regularizers import l2, activity_l2
plt.style.use('ggplot')

pd.set_option('display.max_colwidth', -1, 'display.max_columns', 0)
data = pd.read_csv('../input/voice.csv')
data.head()
data['label'] = data['label'].map({'male' : 0, 'female' : 1})
X = data.ix[:, 0:-1]

y = data['label']
data.head()
plt.plot(data[data.label == 0]['meanfun'], label='male')

plt.plot(data[data.label == 1]['meanfun'], label='female')

plt.legend()
plt.plot(data[data.label == 0]['centroid'], label='male')

plt.plot(data[data.label == 1]['centroid'], label='female')

plt.legend()
plt.plot(data[data.label == 0]['meanfreq'], label='male')

plt.plot(data[data.label == 1]['meanfreq'], label='female')

plt.legend()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
xgbc = xgb.XGBClassifier(learning_rate=0.03, n_estimators=50, seed=1)
xgbc.fit(X_train, y_train)
xgb_preds = xgbc.predict(X_test)
accuracy_score(y_test, xgb_preds)
rf = RandomForestClassifier(n_estimators=50, criterion='entropy', min_impurity_split=1e-3, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

accuracy_score(y_test, rf_preds)
np.random.seed(1)
model = Sequential()

model.add(Dense(output_dim=64, input_dim=X_train.shape[1]))

model.add(Activation('relu'))

model.add(Dense(output_dim=128, input_dim=64, activity_regularizer=activity_l2(0.01), W_regularizer=l2(0.01)))

model.add(Activation('relu'))

model.add(Dense(output_dim=1))

model.add(Activation('sigmoid'))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train.values, y_train.values, nb_epoch=500, batch_size=50, verbose=0)

keras_preds = model.predict(X_test.values)

keras_preds[keras_preds >= 0.5] = 1

keras_preds[keras_preds < 0.5] = 0

accuracy_score(y_test, keras_preds)