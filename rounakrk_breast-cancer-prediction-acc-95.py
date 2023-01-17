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

from tensorflow import keras



from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv1D, MaxPool1D

from tensorflow.keras.optimizers import Adam



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
cancer = pd.read_csv('../input/breastcancer-dataset/data.csv')

cancer.head()
X = cancer.drop(['diagnosis','Unnamed: 32'],axis=1)

y = cancer['diagnosis']
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

y = lb.fit_transform(y)

y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train.shape, X_test.shape
X_train = X_train.reshape(455,31,1)

X_test = X_test.reshape(114,31,1)
epochs = 50

model = Sequential()

model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(31,1)))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer=Adam(lr=0.00005), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)
import warnings

warnings.filterwarnings('ignore')



#plotting training and validation accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train','Val'], loc='upper_left')

plt.show()



#plotting training and validation loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(['Train','Val'], loc='upper_left')

plt.show()