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
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from keras import backend as K
from keras.layers import BatchNormalization
import seaborn as sns
import matplotlib.pyplot as plt
import math
data = pd.read_csv("../input/mushroom-classification/mushrooms.csv") #Reading dataset.
data.head()
#FEATURES ARE STRİNG VALUES
data.info()
#CHECKİNG MİSSİNG VALUES.
for i in data.columns:
  a = data[i].value_counts()
  b = pd.DataFrame({"name":a.name,'feature':a.index, 'count':a.values})
  print(b)
#STALK-ROOT HAS 2480 MİSSİNG VALUES THAT ARE "?" WE SHOULD DROP THİS COLUMN.
data = data.drop('stalk-root', 1)
#CONVERT FEATURES TO BİNARY VALUES.
Y = pd.get_dummies(data.iloc[:,0],  drop_first=False)
X = pd.DataFrame()
for i in data.iloc[:,1:].columns:
    Q = pd.get_dummies(data[i], prefix=i, drop_first=False)
    X = pd.concat([X, Q], axis=1)
#CREATİNG MODEL.
def model():
  model = Sequential()
  model.add(Dense(250, input_dim=X.shape[1], kernel_initializer='uniform', activation='sigmoid'))
  model.add(BatchNormalization())
  model.add(Dropout(0.7))
  model.add(Dense(300, input_dim=250, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.8))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='binary_crossentropy' , optimizer='adamax', metrics=["accuracy"])
  return model
#TRAİNİNG.
model = model()
history = model.fit(X.values, Y.values, validation_split=0.50, epochs=300, batch_size=50, verbose=0)
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % 
      (100*history.history['accuracy'][-1], 100*history.history['val_accuracy'][-1]))
print("Training loss: %.2f%% / Validation loss: %.2f%%" % 
      (100*history.history['loss'][-1], 100*history.history['val_loss'][-1]))
