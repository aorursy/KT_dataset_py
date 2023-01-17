import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from keras.layers import Dense, Activation

from keras.models import Sequential

%matplotlib inline
data = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
data.head()
data.drop(['objid','specobjid'], axis=1, inplace=True)
data.head()
sns.countplot(x=data['class'])
#CONVERT AS CLASSES PARA ONE HOT ENCODING

labels = pd.get_dummies(data['class'], prefix='class')

data = data.drop(columns='class', axis=1)



data.head()
data.drop(['run','rerun','camcol','field'],axis=1,inplace=True)
data.head()
from sklearn import preprocessing



scaler = preprocessing.MinMaxScaler()



normalizado = scaler.fit_transform(data)



train = pd.DataFrame(normalizado)
train.head()
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=128)
y_train.shape
model = Sequential()



model.add(Dense(30, input_dim=X_train.shape[1], activation='sigmoid'))

model.add(Dense(10, activation='sigmoid'))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=12, validation_split=0.10)
y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test, verbose=0) 
score
import matplotlib.pyplot as plt



print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('Acurácia')

plt.xlabel('Época')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('Época')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.summary()
from keras.utils import plot_model



plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
!ls