import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train.shape
y_test.shape
#First image

img1=x_train[0]
plt.imshow(img1)

#note- mnist consists of greyscale images, since we are using matplotlib to plot it uses the default color
y_train[0]
from tensorflow.keras.utils import to_categorical
y_train_cat = to_categorical(y_train,10)

y_test_cat = to_categorical(y_test,10)
y_train_cat[0]
x_train=x_train/255

x_test=x_test/255
plt.imshow(x_train[0])
x_train.shape
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
model=Sequential()



model.add(Conv2D(filters=32,kernel_size=(4,4),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(128, activation='relu'))



model.add(Dense(10,activation='softmax'))





model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy']) 
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2)
model.fit(x_train,y_train_cat,epochs=100,validation_data=(x_test,y_test_cat),callbacks=[early_stop])
Trained=pd.DataFrame(model.history.history)
Trained
Trained['accuracy'].plot()
Trained[['loss','val_loss']].plot()
test=x_test[555]

model.predict_classes(test.reshape(1,28,28,1))
y_test[555]
plt.imshow(test.reshape(28,28))
from sklearn.metrics import classification_report
predictions = model.predict_classes(x_test)
print(classification_report(y_test,predictions))