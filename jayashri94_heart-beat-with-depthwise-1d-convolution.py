# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import math

import random

import pickle

import itertools



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 



from sklearn.utils import shuffle



from scipy.signal import resample



import matplotlib.pyplot as plt



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error



from mlxtend.plotting import plot_confusion_matrix



from keras.models import Sequential

from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Input, Flatten, SeparableConv1D

from keras.layers import GlobalMaxPooling1D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.models import Model



from keras import backend as K

from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler, ModelCheckpoint



np.random.seed(4)



import pickle

from sklearn.preprocessing import OneHotEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/mitbih_train.csv", header=None)

df2 = pd.read_csv("../input/mitbih_test.csv", header=None)

df3= pd.read_csv("../input/ptbdb_abnormal.csv", header=None)

df = pd.concat([df, df2,df3], axis=0)
df.head()
df.info()
df[187].value_counts()
M = df.values

X = M[:, :-1]

y = M[:, -1].astype(int)
del df

del df2

del M
C0 = np.argwhere(y == 0).flatten()

C1 = np.argwhere(y == 1).flatten()

C2 = np.argwhere(y == 2).flatten()

C3 = np.argwhere(y == 3).flatten()

C4 = np.argwhere(y == 4).flatten()
x = np.arange(0, 187)*8/1000



plt.figure(figsize=(12,12))

plt.subplot(5, 1, 1)



plt.plot(x, X[C0, :][0], label="Cat. N")

plt.title("1-beat ECG for category N", fontsize=20)

plt.ylabel("Amplitude", fontsize=15)

plt.xlabel("Time (ms)", fontsize=15)



plt.subplot(5,1, 2)

plt.plot(x, X[C1, :][0], label="Cat. S")

plt.title("1-beat ECG for category S", fontsize=20)

plt.ylabel("Amplitude", fontsize=15)

plt.xlabel("Time (ms)", fontsize=15)



plt.subplot(5,1,3)

plt.plot(x, X[C2, :][0], label="Cat. V")

plt.title("1-beat ECG for category V", fontsize=20)

plt.ylabel("Amplitude", fontsize=15)

plt.xlabel("Time (ms)", fontsize=15)



plt.subplot(5,1, 4)

plt.plot(x, X[C3, :][0], label="Cat. F")

plt.title("1-beat ECG for category F", fontsize=20)

plt.ylabel("Amplitude", fontsize=15)

plt.xlabel("Time (ms)", fontsize=15)



plt.subplot(5,1,5)

plt.plot(x, X[C4, :][0], label="Cat. Q")

plt.title("1-beat ECG for category Q", fontsize=20)

plt.ylabel("Amplitude", fontsize=15)

plt.xlabel("Time (ms)", fontsize=15)

plt.tight_layout()

plt.show()
def stretch(x):

    l = int(187 * (1 + (random.random()-0.5)/3))

    y = resample(x, l)

    if l < 187:

        y_ = np.zeros(shape=(187, ))

        y_[:l] = y

    else:

        y_ = y[:187]

    return y_



def amplify(x):

    alpha = (random.random()-0.5)

    factor = -alpha*x + (1+alpha)

    return x*factor



def augment(x):

    result = np.zeros(shape= (4, 187))

    for i in range(3):

        if random.random() < 0.33:

            new_y = stretch(x)

        elif random.random() < 0.66:

            new_y = amplify(x)

        else:

            new_y = stretch(x)

            new_y = amplify(new_y)

        result[i, :] = new_y

    return result
plt.plot(X[0, :])

plt.plot(amplify(X[0, :]))

plt.plot(stretch(X[0, :]))

plt.show()
result = np.apply_along_axis(augment, axis=1, arr=X[C3]).reshape(-1, 187)

classe = np.ones(shape=(result.shape[0],), dtype=int)*3

X = np.vstack([X, result])

y = np.hstack([y, classe])
subC0 = np.random.choice(C0, 800)

subC1 = np.random.choice(C1, 800)

subC2 = np.random.choice(C2, 800)

subC3 = np.random.choice(C3, 800)

subC4 = np.random.choice(C4, 800)
X_test = np.vstack([X[subC0], X[subC1], X[subC2], X[subC3], X[subC4]])

y_test = np.hstack([y[subC0], y[subC1], y[subC2], y[subC3], y[subC4]])



X_train = np.delete(X, [subC0, subC1, subC2, subC3, subC4], axis=0)

y_train = np.delete(y, [subC0, subC1, subC2, subC3, subC4], axis=0)



X_train, y_train = shuffle(X_train, y_train, random_state=0)

X_test, y_test = shuffle(X_test, y_test, random_state=0)



del X

del y
X_train = np.expand_dims(X_train, 2)

X_test = np.expand_dims(X_test, 2)
print("X_train", X_train.shape)

print("y_train", y_train.shape)

print("X_test", X_test.shape)

print("y_test", y_test.shape)
ohe = OneHotEncoder()

y_train = ohe.fit_transform(y_train.reshape(-1,1))

y_test = ohe.transform(y_test.reshape(-1,1))
print("X_train", X_train.shape)

print("y_train", y_train.shape)

print("X_test", X_test.shape)

print("y_test", y_test.shape)
n_obs, feature, depth = X_train.shape

batch_size = 500
def build_model():

    input_img = Input(shape=(feature, depth), name='ImageInput')

    x = Conv1D(32, 3, activation='relu', padding='same', name='Conv1_1')(input_img)

    x = Conv1D(32, 3, activation='relu', padding='same', name='Conv1_2')(x)

    x = MaxPooling1D(2, name='pool1')(x)

    

    x = SeparableConv1D(32, 3, activation='relu', padding='same', name='Conv2_1')(x)

    x = SeparableConv1D(32, 3, activation='relu', padding='same', name='Conv2_2')(x)

    x = MaxPooling1D(2, name='pool2')(x)

    

    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='Conv3_1')(x)

    x = BatchNormalization(name='bn1')(x)

    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='Conv3_2')(x)

    x = BatchNormalization(name='bn2')(x)

    

    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='Conv3_3')(x)

    x = MaxPooling1D(2, name='pool3')(x)

    

    x = Flatten(name='flatten')(x)

    x = Dense(128, activation='relu', name='fc1')(x)

    x = Dropout(0.6, name='dropout1')(x)

    x = Dense(128, activation='relu', name='fc2')(x)

    x = Dropout(0.5, name='dropout2')(x)

    x = Dense(5, activation='softmax', name='fc3')(x)

    

    model = Model(inputs=input_img, outputs=x)

    return model



model =  build_model()

model.summary()
def exp_decay(epoch):

    initial_lrate = 0.001

    k = 0.75

    t = n_obs//(10000 * batch_size)  # every epoch we do n_obs/batch_size iteration

    lrate = initial_lrate * math.exp(-k*t)

    return lrate



lrate = LearningRateScheduler(exp_decay)
adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(X_train, y_train, 

                    epochs=75, 

                    batch_size=batch_size, 

                    verbose=2, 

                    validation_data=(X_test, y_test), 

                    callbacks=[lrate])
# Get predictions

preds = model.predict(X_test, batch_size=1000)

preds = np.argmax(preds, axis=-1)



# Original labels

orig_test_labels = np.argmax(y_test, axis=-1)



print(orig_test_labels.shape)

print(preds.shape)
# Get the confusion matrix

cm  = confusion_matrix(orig_test_labels, preds)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()