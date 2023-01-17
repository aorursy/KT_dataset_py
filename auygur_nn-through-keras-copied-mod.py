## Importing standard libraries



%pylab inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
## Importing sklearn libraries



from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import LabelEncoder
## Keras Libraries for Neural Networks



from keras.models import Sequential

from keras.layers import Dense,Dropout,Activation

from keras.utils.np_utils import to_categorical

from keras.callbacks import EarlyStopping
## Read data from the CSV file



data = pd.read_csv('../input/train.csv')

parent_data = data.copy()    ## Always a good idea to keep a copy of original data

ID = data.pop('id')
data.shape
## Since the labels are textual, so we encode them categorically



y = data.pop('species')

y = LabelEncoder().fit(y).transform(y)

print(y.shape)
## Most of the learning algorithms are prone to feature scaling

## Standardising the data to give zero mean =)



X = StandardScaler().fit(data).transform(data)

print(X.shape)
## We will be working with categorical crossentropy function

## It is required to further convert the labels into "one-hot" representation



y_cat = to_categorical(y)

print(y_cat.shape)
2## Developing a layered model for Neural Networks

## Input dimensions should be equal to the number of features

## We used softmax layer to predict a uniform probabilistic distribution of outcomes



model = Sequential()

model.add(Dense(600,input_dim=192,  init='uniform', activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(300, activation='sigmoid'))

model.add(Dropout(0.3))

model.add(Dense(99, activation='softmax'))
## Error is measured as categorical crossentropy or multiclass logloss

## Adagrad, rmsprop, SGD, Adadelta, Adam, Adamax, Nadam

model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])
## Fitting the model on the whole training data

## the validation is literally just the last x% of samples in the input you passed.

## the split is deeply flawed because it should split proportionally for multi-class problems

## this is one of the reasons of low performance of this model

early_stopping = EarlyStopping(monitor='val_loss', patience=280)

history = model.fit(X,y_cat,batch_size=192,

                    nb_epoch=800 ,verbose=0, validation_split=0.1, callbacks=[early_stopping])
## we need to consider the loss for final submission to leaderboard

## print(history.history.keys())

print('val_acc: ',max(history.history['val_acc']))

print('val_loss: ',min(history.history['val_loss']))

print('train_acc: ',max(history.history['acc']))

print('train_loss: ',min(history.history['loss']))



print()

print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))
# summarize history for loss

## Plotting the loss with the number of iterations



plt.semilogy(history.history['loss'])

plt.semilogy(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
## Plotting the error with the number of iterations

## With each iteration the error reduces smoothly

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
## read test file

test = pd.read_csv('../input/test.csv')

index = test.pop('id')

test = StandardScaler().fit(test).transform(test)

yPred = model.predict_proba(test)
## Converting the test predictions in a dataframe as depicted by sample submission

yPred = pd.DataFrame(yPred,index=index,columns=sort(parent_data.species.unique()))
fp = open('submission_nn_kernel.csv','w')

fp.write(yPred.to_csv())