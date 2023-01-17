# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import  StandardScaler

from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv1D, MaxPooling1D

from keras.models import Model

import keras.backend as K

from keras import metrics

from keras.optimizers import Adam

from keras import regularizers

from keras.losses import binary_crossentropy

from sklearn.metrics import roc_auc_score

# LOAD DATA

train=pd.read_csv("./../input/train.csv")

test=pd.read_csv("./../input/test.csv")
train.head()
train.describe(include='all')
sns.countplot(train["target"])
train["target"].value_counts()
df_train = train.drop(["ID_code","target"],axis=1)

y_train = train["target"]

df_test = test.drop(["ID_code"],axis=1)
from imblearn.under_sampling import RandomUnderSampler



rus = RandomUnderSampler()

df_train_rus, y_train_rus = rus.fit_sample(df_train, y_train)
unique, counts = np.unique(y_train_rus, return_counts=True)

dict(zip(unique, counts))
type(y_train_rus)
sc = StandardScaler()

df_train = sc.fit_transform(df_train_rus)

df_test = sc.fit_transform(df_test)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
type(df_train)
df_train = pd.DataFrame(df_train)

y_train = pd.DataFrame(y_train_rus)

df_test = pd.DataFrame(df_test)
def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall
def _Model():

    inp = Input(shape=(200, 1))

    d1 = Dense(128, activation='sigmoid')(inp)

    d2 = Conv1D(64, 2, activation="relu", kernel_initializer="uniform")(d1)

    d3 = Dense(32, activation='sigmoid')(d2)

    d4 = Dense(16, activation='relu')(d3)

    f1 = Flatten()(d4)

    preds = Dense(1, activation='sigmoid')(f1)

    model = Model(inputs=inp, outputs=preds)

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy',recall])

    return model
print(_Model().summary())
preds = []

c = 0

oof_preds = np.zeros((len(df_train), 1))

for train, valid in cv.split(df_train, y_train):

    print("VAL %s" % c)

    X_train = np.reshape(df_train.iloc[train].values, (-1, 200, 1))

    y_train_ = y_train.iloc[train].values

    X_valid = np.reshape(df_train.iloc[valid].values, (-1, 200, 1))

    y_valid = y_train.iloc[valid].values

    model = _Model()

    history=model.fit(X_train, y_train_, validation_data=(X_valid, y_valid), epochs=20, verbose=2, batch_size=256)

    #print(model.evaluate(X_valid, y_valid))

    #model.load_weights('cv_{}.h5'.format(c))

    

    X_test = np.reshape(df_test.values, (200000, 200, 1))

    curr_preds = model.predict(X_test, batch_size=2048)

    oof_preds[valid] = model.predict(X_valid)

    preds.append(curr_preds)

    c += 1

auc = roc_auc_score(y_train, oof_preds)

print("CV_AUC: {}".format(auc))
# SAVE DATA

preds = np.asarray(preds)

preds = preds.reshape((5, 200000))

preds_final = np.mean(preds.T, axis=1)

submission = pd.read_csv('./../input/sample_submission.csv')

submission['target'] = preds_final
history_dict = history.history

history_dict.keys()
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

recall = history.history['recall']

val_recall = history.history['val_recall']

epochs = range(1, len(acc) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_recall, 'g-', label='Validation Recall')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall
def _Model_1():

    inp = Input(shape=(200, 1))

    d1 = Dense(64, activation='sigmoid')(inp)

    d2 = Conv1D(128, 2, activation="relu", kernel_initializer="uniform")(d1)

    d3 = Dense(32, activation='sigmoid')(d2)

    d4 = Dense(16, activation='relu')(d3)

    f1 = Flatten()(d4)

    preds = Dense(1, activation='sigmoid')(f1)

    model = Model(inputs=inp, outputs=preds)

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy', recall])

    return model
print(_Model_1().summary())
preds_1 = []

c = 0

oof_preds = np.zeros((len(df_train), 1))

for train, valid in cv.split(df_train, y_train):

    print("VAL %s" % c)

    X_train = np.reshape(df_train.iloc[train].values, (-1, 200, 1))

    y_train_ = y_train.iloc[train].values

    X_valid = np.reshape(df_train.iloc[valid].values, (-1, 200, 1))

    y_valid = y_train.iloc[valid].values

    model = _Model_1()

    history=model.fit(X_train, y_train_, validation_data=(X_valid, y_valid), epochs=20, verbose=2, batch_size=256)

    #print(model.evaluate(X_valid, y_valid))

    #model.load_weights('cv_{}.h5'.format(c))

    

    X_test = np.reshape(df_test.values, (200000, 200, 1))

    curr_preds = model.predict(X_test, batch_size=2048)

    oof_preds[valid] = model.predict(X_valid)

    preds_1.append(curr_preds)

    c += 1

auc = roc_auc_score(y_train, oof_preds)

print("CV_AUC: {}".format(auc))
# SAVE DATA

preds_1 = np.asarray(preds_1)

preds_1 = preds_1.reshape((5, 200000))

preds_final = np.mean(preds_1.T, axis=1)

submission_1 = pd.read_csv('./../input/sample_submission.csv')

submission_1['target'] = preds_final
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

recall = history.history['recall']

val_recall = history.history['val_recall']

epochs = range(1, len(acc) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_recall, 'g-', label='Validation Recall')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()