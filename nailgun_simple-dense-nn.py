import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import tensorflow as tf

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

import gc
X = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')

X.head()
X_test = pd.concat([X_test, pd.get_dummies(X['wheezy-copper-turtle-magic'], prefix='magic', drop_first=True)], axis=1).drop(['wheezy-copper-turtle-magic'], axis=1)



y = X.target

X = pd.concat([X, pd.get_dummies(X['wheezy-copper-turtle-magic'], prefix='magic', drop_first=True)], axis=1).drop(['wheezy-copper-turtle-magic'], axis=1)

X.drop('target', axis=1, inplace=True)

X.drop('id', axis=1, inplace=True)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
model = None

gc.collect()

np.random.seed(42)

model = Sequential()

model.add(Dense(830, input_dim=len(X.columns), kernel_initializer="normal", activation="relu"))

model.add(Dropout(0.1))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['mae'])

hist=model.fit(X_train, y_train, epochs = 5, batch_size=896, verbose=1, validation_data=(X_valid, y_valid))
y_pred = model.predict_proba(X_valid)

print(roc_auc_score(y_valid, y_pred))



plt.plot(hist.history['mean_absolute_error'], label = 'mae')

plt.plot(hist.history['val_mean_absolute_error'], label = 'val_mae')

plt.plot(hist.history['loss'], label = 'loss')

plt.plot(hist.history['val_loss'], label = 'val_loss')

plt.legend()

plt.show()
hist=model.fit(X, y, epochs = 15, batch_size=896, verbose=0)

pred = model.predict(X_test.drop('id', axis=1))



plt.plot(hist.history['mean_absolute_error'], label = 'mae')

plt.plot(hist.history['loss'], label = 'loss')

plt.legend()

plt.show()
import seaborn as sns

plt.title('results distribution')

sns.distplot(pred)
submission = pd.DataFrame({"id" : X_test['id'].values, "target" : pred[:,0]})

submission.to_csv('submission.csv', index = False, header = True)