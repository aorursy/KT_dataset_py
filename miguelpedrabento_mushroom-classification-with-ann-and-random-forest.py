# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
df0 = pd.read_csv('../input/mushrooms.csv')

df0.shape
df0.describe()
df0.columns
for k in range(1,22,2):

    fig, ax =plt.subplots(1,2,figsize=(15, 4))

    sns.countplot(x = df0.columns[k], hue = 'class', data=df0, palette="Set1", ax=ax[0])

    sns.countplot(x = df0.columns[k+1], hue = 'class', data=df0, palette="Set1", ax=ax[1])

    plt.show()
df1 = df0

df1['class'] = df0['class'].replace({'p':1,'e':0})

df1 = pd.get_dummies(df0)
df1.shape
x = df1.values[:,1:].astype(float)

y = df1.values[:,0].astype(float)
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.optimizers import SGD, Adam

from keras.layers.normalization import BatchNormalization

from keras import regularizers
model = Sequential()

model.add(Dense(64, input_shape=(117,),activation = 'tanh',use_bias = True, kernel_regularizer=regularizers.l2(.0005)))

Dropout(.2)

model.add(Dense(16,activation = 'tanh',use_bias = True, kernel_regularizer=regularizers.l2(.0005)))

Dropout(.2)

model.add(Dense(16,activation = 'tanh',use_bias = True, kernel_regularizer=regularizers.l2(.0005)))

Dropout(.2)

model.add(Dense(1,activation = 'sigmoid',use_bias = True, kernel_regularizer=regularizers.l2(.0005)))



model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
history = model.fit(x,y,validation_split=0.2, epochs=20, batch_size=5, verbose=1)
plt.figure(figsize=(15,4))

plt.subplot(121)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Mushrooms')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(122)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Mushrooms')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200)
col1 = list(df1.columns)[1:len(list(df1.columns))]

rf.fit(df1[col1],df1['class'])

rf.score(df1[col1],df1['class'])
feature_importances = pd.DataFrame(rf.feature_importances_,

                                   index = df1[col1].columns,

                                    columns=['importance']).sort_values('importance',      ascending=False)

feature_importances
x1 = list(feature_importances.index)

y1 = feature_importances.values.flatten().tolist()

n_imp = 50

plt.figure(figsize=(15,8))

plt.bar(range(len(x1[0:n_imp])), y1[0:n_imp])

plt.xticks(range(len(x1[0:n_imp])), x1[0:n_imp],rotation=90)

plt.show()
df2 = pd.get_dummies(df0)[['class']+list(feature_importances.index)[0:50]]

df2.shape
x2 = df2.values[:,1:].astype(float)

y2 = df2.values[:,0:1].astype(float)
model_2 = Sequential()

model_2.add(Dense(64, input_shape=(50,),activation = 'tanh',use_bias = True, kernel_regularizer=regularizers.l2(.0005)))

Dropout(.2)

model_2.add(Dense(16,activation = 'tanh',use_bias = True, kernel_regularizer=regularizers.l2(.0005)))

Dropout(.2)

model_2.add(Dense(16,activation = 'tanh',use_bias = True, kernel_regularizer=regularizers.l2(.0005)))

Dropout(.2)

model_2.add(Dense(1,activation = 'sigmoid',use_bias = True, kernel_regularizer=regularizers.l2(.0005)))



model_2.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
history = model_2.fit(x2,y2,validation_split=0.2, epochs=20, batch_size=5, verbose=3)
plt.figure(figsize=(15,4))

plt.subplot(121)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Mushrooms')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(122)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Mushrooms')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
rf_p = rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_pred, y_test)
corr_matrix = df1.corr().abs()
plt.figure(figsize=(100,100))

sns.heatmap(corr_matrix, cmap = "RdBu_r", annot=True, vmin=0, vmax=1)

plt.show()