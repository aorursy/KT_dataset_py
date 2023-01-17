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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

import warnings

warnings.filterwarnings('ignore')

def display_all(df):

    with pd.option_context("display.max_rows", 100, "display.max_columns", 100):

        display(df)
shrooms = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
shrooms.head()
data = shrooms.values
X = data[:, 1:]

y = data[:, 0]
#format to string just bc it could accidentally get converted to nums 

X = X.astype(str)

y = y.reshape((len(y), 1))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#one-hot encode X

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

ohe.fit(X_train)

X_train_enc = ohe.transform(X_train)

X_test_enc = ohe.transform(X_test)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(y_train)

y_train_enc = le.transform(y_train)

y_test_enc = le.transform(y_test)
from keras import models 

from keras import layers



model = models.Sequential()



model.add(layers.Dense(16, activation='relu', input_dim=X_train_enc.shape[1]))

model.add(layers.Dense(8, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_enc, y_train_enc, epochs=10, batch_size=64, validation_data=(X_test_enc, y_test_enc), verbose=2)
history_dict = history.history
history_dict.keys()
val_loss_values = history_dict['val_loss']

loss_values = history_dict['loss']

acc = history_dict['accuracy']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, loss_values, 'bo', label='Training Loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()

acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']

epochs = range(1, len(acc_values) + 1)



plt.plot(epochs, acc_values, 'bo', label='Training Accuracy')

plt.plot(epochs, val_acc_values, 'b', label='Validation Accuracy')



plt.xlabel('Epochs')

plt.ylabel('Loss')



plt.legend()

plt.show()