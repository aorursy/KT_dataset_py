import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
import tensorflow as tf

import keras as keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
data = pd.read_csv("../input/predicting-churn-for-bank-customers/Churn_Modelling.csv")

data.head()
x = data.iloc[:,3:-1]

y = data.iloc[:,-1]

x.head()
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
le = LabelEncoder()

x['Gender'] = le.fit_transform(x['Gender'])
x = pd.get_dummies(x,drop_first=True,columns=['Geography'])

x.head()
x_train,x_test,y_train , y_test = train_test_split(x,y,test_size=.2,stratify=y,random_state=10)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)

x.shape
model = Sequential()

model.add(Dense(x.shape[1],activation='relu',input_dim=x.shape[1]))

model.add(Dense(128,activation='relu'))

model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss="binary_crossentropy",metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=300,epochs=10,verbose=1,validation_split=.2)
model.evaluate(x_test,y_test)
y_pred = model.predict_classes(x_test)
history.history
import matplotlib.pyplot as plt



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(["Train","Test"],loc="upper left")

plt.plot()

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(["Train Loss","Test Loss"],loc="upper left")

plt.plot()

plt.show()
# !pip install mlxtend

class_names = ['still with bank','will go out']

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix



cm = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm, figsize=(6,6),colorbar=True,show_normed=True,class_names=class_names)