# import

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



# read csv data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
# ---preprocessing--

# delete Name & Ticket columns

train = train.drop(["Name", "Ticket"], axis=1)

test = test.drop(["Name", "Ticket"], axis=1)



# delete Cabin column because it has too many NaN

train = train.drop(["Cabin"], axis=1)

test = test.drop(["Cabin"], axis=1)



# Sex: male -> 1, female -> 0

train.loc[train["Sex"]=="male", "Sex"] = 1

train.loc[train["Sex"]=="female", "Sex"] = 0

test.loc[test["Sex"]=="male", "Sex"] = 1

test.loc[test["Sex"]=="female", "Sex"] = 0



# fill empty Embarked units of the train data with "S"

train["Embarked"] = train["Embarked"].fillna("S")



# Embarked: S -> 0, C -> 1, Q -> 2

train.loc[train["Embarked"]=="S", "Embarked"] = 0

train.loc[train["Embarked"]=="C", "Embarked"] = 1

train.loc[train["Embarked"]=="Q", "Embarked"] = 2

test.loc[test["Embarked"]=="S", "Embarked"] = 0

test.loc[test["Embarked"]=="C", "Embarked"] = 1

test.loc[test["Embarked"]=="Q", "Embarked"] = 2



# fill test.Fare[152]

test.loc[152, "Fare"] = test.loc[test["Pclass"]==test.Pclass[152], "Fare"].median()



# calculate Mean, Std and # of Nan for the train data

average_age_train   = train["Age"].mean()

std_age_train       = train["Age"].std()

count_nan_age_train = train["Age"].isnull().sum()



# test data

average_age_test   = test["Age"].mean()

std_age_test       = test["Age"].std()

count_nan_age_test = test["Age"].isnull().sum()



# generate random numbers

rand_1 = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)



# replace NaN with random numbers

train.loc[np.isnan(train["Age"]), "Age"] = rand_1

test.loc[np.isnan(test["Age"]), "Age"] = rand_2



# float -> int

train['Age'] = train['Age'].astype(int)

test['Age']    = test['Age'].astype(int)
train.head()
# Deep Neural Network with Keras



from keras.datasets import mnist

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import Adadelta

from keras.utils import np_utils

from keras.callbacks import Callback



# prepare train & test data

X_train = train.drop(["PassengerId", "Survived"], axis=1)

y_train = train["Survived"]

X_test  = test.drop(["PassengerId"],axis=1)



X_train = np.array(X_train).astype('float32')

y_train = np.array(y_train).astype('int32')

y_train = np_utils.to_categorical(y_train, 2)

X_test = np.array(X_test).astype('float32')
# Sequential

model = Sequential()



# 1st hidden layer

model.add(Dense(128, input_shape=(7, ), init="uniform"))

model.add(Activation('relu'))

model.add(Dropout(0.2))



# 2nd hidden layer

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.2))



# output layer

model.add(Dense(2))

model.add(Activation('sigmoid'))



# compile

model.compile(loss='binary_crossentropy',

              optimizer=Adadelta(),

              metrics=['accuracy'])





# fit

call = Callback()

print("Training...")

hist = model.fit(X_train, y_train,

                 batch_size=128,

                 nb_epoch=3000,

                 verbose=0,

                 validation_split=0.1,

                 callbacks=[call])
# plot loss & accuracy vs epoch number



loss = hist.history['loss']

val_loss = hist.history['val_loss']

acc = hist.history['acc']

val_acc = hist.history['val_acc']



from scipy.ndimage.filters import gaussian_filter as gf

fit_loss = gf(loss, sigma=50)

fit_val_loss = gf(val_loss, sigma=50)

fit_acc = gf(acc, sigma=50)

fit_val_acc = gf(val_acc, sigma=50)



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 20))

nb_epoch = len(loss)



ax[0].plot(range(nb_epoch), loss, alpha=0.4)

ax[0].plot(range(nb_epoch), val_loss, alpha=0.4)

ax[0].plot(range(nb_epoch), fit_loss, label='loss', linewidth=4)

ax[0].plot(range(nb_epoch), fit_val_loss, label='val_loss', linewidth=4)

ax[0].set_xlabel('epoch')

ax[0].set_ylabel('loss')

ax[0].set_ylim([0.2, 0.6])

ax[0].legend()

ax[0].grid()



ax[1].plot(range(nb_epoch), acc, alpha=0.4)

ax[1].plot(range(nb_epoch), val_acc, alpha=0.4)

ax[1].plot(range(nb_epoch), fit_acc, label='acc', linewidth=3)

ax[1].plot(range(nb_epoch), fit_val_acc, label='val_acc', linewidth=4)

ax[1].set_xlabel('epoch')

ax[1].set_ylabel('accuracy')

ax[1].set_ylim([0.6, 0.95])

ax[1].legend()

ax[1].grid()



plt.show()