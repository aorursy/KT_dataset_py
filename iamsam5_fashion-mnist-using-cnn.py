from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

import plotly.express as px

%matplotlib inline
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
img=x_train[0]
px.imshow(img)

y_train[0] 

#To crosscheck we check the label coressponding to the given image class

#Output label is 9 - Ankle Boot
x_train.shape
y_train.shape
y_train
from tensorflow.keras.utils import to_categorical
y_train_cat = to_categorical(y_train,10)

y_test_cat = to_categorical(y_test,10)
x_train[0]
x_train=x_train/255

x_test=x_test/255
px.imshow(x_train[1])
y_train[1]

#The output indicates class 0 which is T-Shirt
x_train=x_train.reshape(60000,28,28,1)
x_test.shape
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
import pandas as pd

history=pd.DataFrame(data=model.history.history)
history
history[['loss','val_loss']].plot()
history['accuracy'].plot()
prediction=model.predict_classes(x_test)
sample=x_test[888]

sample.shape
sample=sample.reshape(1,28,28,1)
model.predict_classes(sample)
y_test[888]
px.imshow(sample.reshape(28,28))
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,prediction)
cm