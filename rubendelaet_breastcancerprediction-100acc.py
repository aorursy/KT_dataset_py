# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

data
data.diagnosis = data.diagnosis.map({'M':0,'B':1})
data.diagnosis.hist()
data = data.drop(data.columns[-1], axis=1)
import matplotlib.pyplot as plt



# Scikit-learn

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.utils import class_weight



# Tensorflow

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout
target = 'diagnosis'

features = list(data.loc[:,data.columns!=target])



X = data[features].values

y = data[target].values



scaler =  MinMaxScaler()

X = scaler.fit_transform(X)



X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=.1, random_state=0)
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
model = Sequential()

model.add(Dense(6, input_dim=X_train.shape[1], activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=.1, class_weight=class_weights)
# Test the neural network on the test set. Use the following metrics: accuracy, recall, 

# precision, f1-score and the ROC/auRoc.



# Accuray 

plt.plot(history.history['accuracy'],'r')

plt.plot(history.history['val_accuracy'],'b')

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()



# Loss 

plt.plot(history.history['loss'],'r')

plt.plot(history.history['val_loss'],'b')

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
# Testing with the test set



y_pred = model.predict_classes(X_test)



print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print('Accuracy:', accuracy_score(y_test, y_pred) * 100)