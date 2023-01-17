import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/heart_disease.csv")

data.head()
X=data.drop(['target'],axis=1)

y=data['target']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

X_train.shape[1]
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.utils import plot_model
model=Sequential()

model.add(Dense(32,activation='relu',input_shape=(X_train.shape[1],)))

model.add(Dense(16,activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(1,activation='sigmoid'))
optimizer=RMSprop(0.01)

model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

model.summary()

plot_model(model)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=200,batch_size=10,verbose=1)
model.evaluate(X_test, y_test)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['Train', 'Validation'])

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['Train', 'Validation'])

plt.show()