# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
from matplotlib import pyplot as plt
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv', header=0, sep=',')
print (data.head())
test = pd.read_csv('../input/test.csv', header=0, sep=',')
print (test.head())
X = data.drop('SalePrice', axis=1).drop('Id',axis=1)
test = test.drop('Id',axis=1)
y = data['SalePrice']

y = pd.Series(np.log(y))
X = pd.get_dummies(X)
test = pd.get_dummies(test)
X_1, test_1 = X.align(test, join='left',axis=1)
print (X_1.head())
print (test_1.head())
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X_2 = my_imputer.fit_transform(X_1)
test_2 = my_imputer.transform(test_1)
print (X_2.shape)
print (test_2.shape)
meansX2 = X_2.mean(axis=0)
meanst2 = test_2.mean(axis=0)
stdX2 = X_2.std(axis=0)
stdt2 = test_2.std(axis=0)
X_3 = (X_2 - meansX2)/stdX2
test_3 = (test_2 - meanst2)/stdt2
print (X_3.shape)
print (test_3.shape)
from sklearn import ensemble
clf = ensemble.RandomForestRegressor(n_estimators=100,  n_jobs=-1, random_state=1)
from sklearn import model_selection
RF_scoring = model_selection.cross_val_score(clf, X_3, y, n_jobs=-1)
RF_scoring.mean()
clf.fit(X_3,y)
ans = clf.predict(test_3)
ans
ans1 = np.exp(ans)
ans1
otvet = pd.read_csv('../input/sample_submission.csv', header=0, sep=',')
otvet.head()
otvet.iloc[:,1] = ans1
otvet.head()
metrics.mean_squared_error(y,clf.predict(X_3))
import tensorflow as tf
print(tf.__version__)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=160, activation='relu', input_shape=[288]),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(units=70, activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(units=30, activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(units=1, activation=None),
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.005),loss='mean_squared_error', metrics=['mse'])
#model.summary()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
def lr_decay(epoch):
  return 0.005 * math.pow(0.6, epoch)

lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=False)
ep = 600
history = model.fit(X_3,y,epochs=ep, batch_size=32, verbose=False, validation_split=0.1,callbacks=[early_stop,lr_decay_callback])
metrics.mean_squared_error(y,model.predict(X_3))
plt.figure(figsize=(20,7))
plt.subplot(1,2,1)
plt.plot(history.epoch,history.history['mse'], label='train')
plt.plot(history.epoch,history.history['val_mse'], label='test')
plt.ylim(0,5)
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.epoch,history.history['loss'], label='train')
plt.plot(history.epoch,history.history['val_loss'], label='test')
plt.legend()

#history.history['val_loss']
ans = pd.read_csv('../input/sample_submission.csv', header=0, sep=',')
ans.iloc[:,1] = np.exp(model.predict(test_3))
ans.to_csv('submission_TF_earlystop2.csv',index=None)
