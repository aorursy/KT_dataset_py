# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt



print(os.listdir("../input"))



from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential

from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping



from keras import optimizers



## Load

irisDf = pd.read_csv('../input/Iris.csv')

irisDf.head()
irisDf
## Pre-process
# Normalize data(?)
# Drop unused columns from train



X = irisDf.drop('Id', axis=1)

X = X.drop('Species', axis=1)

X.head()
# Create target dataframe (should it be vector instead?)

targets = pd.DataFrame(irisDf.loc[:, 'Species'])

targets.head()
# Convert species into numerical categories

Y = pd.get_dummies(targets).values

Y.shape
# Test, train split

from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)



X_train.head()
X_train.shape
irisDf.iloc[0,:]
len(irisDf)
X.shape[1]

## Model

# Set hyperparameters

seed = 2019

hiddenUnitCount = 64

hiddenLayerCount = 3

dropoutValue = 0.2

epochs = 150

learningRate = 0.001
def GetModel():

    m = Sequential()



    m.add(Dense(X.shape[1], activation='relu', input_dim=4))

    m.add(Dropout(dropoutValue))



    for i in range(0, hiddenLayerCount):

        m.add(Dense(hiddenUnitCount, activation='relu'))

        m.add(Dropout(dropoutValue))



    m.add(Dense(Y.shape[1], activation='softmax'))

    m.compile(optimizer=optimizers.RMSprop(lr=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])



#     m.summary()

    return m



model = GetModel()

model.summary()

# model.compile(optimizer=optimizers.RMSprop(lr=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])



# checkpointer = ModelCheckpoint(filepath='inputs/weights.best.from_scratch.hdf5', 

#                                verbose=1, save_best_only=True)



history = model.fit(X_train, Y_train,

                    validation_data=(X_test, Y_test),

                    epochs=epochs, batch_size=20, verbose=1)

#                     epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

fig_size = plt.rcParams["figure.figsize"]

fig_size



fig_size[0] = 12

fig_size[1] = 9

plt.rcParams["figure.figsize"] = fig_size
# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
from sklearn.model_selection import StratifiedKFold



kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

accuracyResults = []

lossResults = []

historyResults = []

currentSplitCount = 0



for train, test in kfold.split(X, targets):

    currentSplitCount += 1

    print("Split #%i:" % currentSplitCount)

    

    processedY = pd.get_dummies(targets.iloc[train]).values

    processedYTest = pd.get_dummies(targets.iloc[test]).values

    model = GetModel()

    

    earlyStopping = EarlyStopping(min_delta=100, monitor='loss', patience=10)

    history = model.fit(X.iloc[train], processedY,

                    validation_data=(X.iloc[test], processedYTest),

                    epochs=epochs, batch_size=20,

#                     callbacks=[earlyStopping],

                    verbose=0)

    

    historyResults.append(history)

    scores = model.evaluate(X.iloc[test], processedYTest)

    

#     print(model.metrics_names)

    print("%s: %.2f" % (model.metrics_names[0], scores[0]*100))

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    lossResults.append(scores[0] * 100)

    accuracyResults.append(scores[1] * 100)

    

    # validation??

    

print("mean loss: %.2f (+/- %.2f)" % (np.mean(lossResults), np.std(lossResults)))    

print("mean acc: %.2f%% (+/- %.2f%%)" % (np.mean(accuracyResults), np.std(accuracyResults)))    

    
# Plot loss values

legendLabels = []

i = 1

for h in historyResults:

    plt.plot(h.history['loss'])

#     plt.plot(h.history['val_loss'])

    legendLabels.append("set %i loss" % (i))

#     legendLabels.append("set %i val. loss" % (i))

    i += 1



# print(np.mean(lossResults) / 100)

# plt.plot(np.mean(lossResults) / 100)

plt.axhline(y=np.mean(lossResults) / 100, color='b', linestyle='dotted')

plt.title('Cross-validation loss, seed: %i' % (seed))

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(legendLabels, loc='upper left')

plt.show()
# Plot acc values

legendLabels = []

i = 1

for h in historyResults:

    plt.plot(h.history['acc'])

#     plt.plot(h.history['val_acc'])

    legendLabels.append("set %i acc." % (i))

#     legendLabels.append("set %i val. acc." % (i))

    i += 1



plt.axhline(y=np.mean(accuracyResults) / 100, color='b', linestyle='dotted')

plt.title('Cross-validation accuracy, seed: %i' % (seed))

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(legendLabels, loc='upper left')

plt.show()