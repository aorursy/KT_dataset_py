import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from sklearn.metrics import classification_report
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
plt.imshow(x_train[0])
print(x_test.shape)
print(x_train.shape)
print(x_train.max())
print(x_test.max())
x_train=x_train/255
x_test=x_test/255
print(x_train.min())
print(x_test.min())
print(x_train.max())
print(x_test.max())
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
y_cat_train=to_categorical(y_train)
y_cat_test=to_categorical(y_test)
model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

model.summary()
model.fit(x_train,y_cat_train,epochs=10)
model.metrics_names
model.evaluate(x_test,y_cat_test)
prediction=model.predict_classes(x_test)
print(y_cat_test[0])
print(prediction[0])
print(classification_report(y_test,prediction))