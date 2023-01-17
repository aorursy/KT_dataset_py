# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import BatchNormalization,Dropout,Dense,Flatten,Conv1D

from tensorflow.keras.optimizers import Adam
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head()
df.shape
df.info()
df.Class.unique()
df.Class.value_counts()
nf = df[df.Class==0]

f = df[df.Class==1]
nf = nf.sample(738)
data = f.append(nf,ignore_index=True)
data.shape
X = data.drop(['Class'],axis=1)

y=data['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y)

X_train.shape,X_test.shape
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)
y_train=y_train.to_numpy()

y_test=y_test.to_numpy()
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)

X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
model=Sequential()

model.add(Conv1D(32,2,activation='relu',input_shape=X_train[0].shape))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Conv1D(64,2,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs=50,validation_data=(X_test,y_test))
def plotLearningCurve(history,epochs):

  epochRange = range(1,epochs+1)

  plt.plot(epochRange,history.history['accuracy'])

  plt.plot(epochRange,history.history['val_accuracy'])

  plt.title('Model Accuracy')

  plt.xlabel('Epoch')

  plt.ylabel('Accuracy')

  plt.legend(['Train','Validation'],loc='upper left')

  plt.show()
plotLearningCurve(history,50)
def plotLearningCurve(history,epochs):

    epochRange = range(1,epochs+1)

    plt.plot(epochRange,history.history['loss'])

    plt.plot(epochRange,history.history['val_loss'])

    plt.title('Model Loss')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.legend(['Train','Validation'],loc='upper left')

    plt.show()



plotLearningCurve(history,50)