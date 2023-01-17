# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# open training dataset (1)

import csv

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission.head()
train.shape, test.shape, sample_submission.shape
cols=["target","id"]

X = train.drop(cols,axis=1)

y = train["target"]



X_test  = test.drop("id",axis=1)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)
import keras

keras.__version__
from keras import models

from keras import layers



original_model = models.Sequential()

original_model.add(layers.Dense(16, activation='relu', input_shape=(300,)))

original_model.add(layers.Dense(16, activation='relu'))

original_model.add(layers.Dense(1, activation='sigmoid'))



original_model.compile(optimizer='rmsprop',

                       loss='binary_crossentropy',

                       metrics=['acc'])
original_hist = original_model.fit(X_train, y_train,

                                   epochs=30,

                                   batch_size=512,

                                   validation_data=(X_val, y_val))
epochs = range(1, 31)

original_val_loss = original_hist.history['val_loss']
import matplotlib.pyplot as plt



# b+ is for "blue cross"

plt.plot(epochs, original_val_loss, 'b+', label='Original model')

plt.xlabel('Epochs')

plt.ylabel('Validation loss')

plt.legend()



plt.show()
from keras import regularizers



l2_model = models.Sequential()

l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),

                          activation='relu', input_shape=(300,)))

l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),

                          activation='relu'))

l2_model.add(layers.Dense(1, activation='sigmoid'))
l2_model.compile(optimizer='rmsprop',

                 loss='binary_crossentropy',

                 metrics=['acc'])
l2_model_hist = l2_model.fit(X_train, y_train,

                             epochs=30,

                             batch_size=512,

                             validation_data=(X_val, y_val))
l2_model_val_loss = l2_model_hist.history['val_loss']



plt.plot(epochs, original_val_loss, 'b+', label='Original model')

plt.plot(epochs, l2_model_val_loss, 'bo', label='L2-regularized model')

plt.xlabel('Epochs')

plt.ylabel('Validation loss')

plt.legend()



plt.show()
dpt_model = models.Sequential()

dpt_model.add(layers.Dense(300,activation='relu', input_shape=(300,)))

dpt_model.add(layers.Dropout(0.66))

dpt_model.add(layers.Dense(16,activation='relu'))

dpt_model.add(layers.Dropout(0.2))

dpt_model.add(layers.Dense(16,activation='relu'))

dpt_model.add(layers.Dense(16,activation='relu'))

dpt_model.add(layers.Dropout(0.1))

dpt_model.add(layers.Dense(1, activation='sigmoid'))



dpt_model.compile(optimizer='rmsprop',

                  loss='binary_crossentropy',

                  metrics=['acc'])
dpt_model_hist = dpt_model.fit(X_train, y_train,

                               epochs=30,

                               batch_size=512,

                               validation_data=(X_val, y_val))
dpt_model_val_loss = dpt_model_hist.history['val_loss']



plt.plot(epochs, original_val_loss, 'b+', label='Original model')

plt.plot(epochs, dpt_model_val_loss, 'bo', label='Dropout-regularized model')

plt.xlabel('Epochs')

plt.ylabel('Validation loss')

plt.legend()



plt.show()
import matplotlib.pyplot as plt

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support)

predictions = dpt_model.predict(X_val)

predictions_2 = original_model.predict(X_val)

fpr, tpr, thresholds = roc_curve(y_val, predictions)

fpr2, tpr2, thresholds2 = roc_curve(y_val, predictions_2)

roc_auc = auc(fpr, tpr)

roc_auc2 = auc(fpr2, tpr2)

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)

plt.plot(fpr2, tpr2, label='AUC2 = %0.4f'% roc_auc2)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show();
def column(matrix, i):

    return [row[i] for row in matrix]
predictions_test = dpt_model.predict(X_test)

y_predictions = column(predictions_test, 0)

submission_wyatt = pd.DataFrame({

        "id": test["id"],

        "target": y_predictions

    })

submission_wyatt.to_csv('submission_wyatt.csv', index=False)