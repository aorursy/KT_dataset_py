import numpy as np 
import pandas as pd
import os
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,Flatten,MaxPooling2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
#print(os.listdir("../input"))
df_train = pd.read_csv('../input/fashion-mnist_train.csv')
df_test = pd.read_csv('../input/fashion-mnist_test.csv')
train = np.array(df_train,dtype = 'float32')
test  = np.array(df_test,dtype = 'float32')
# Splitting the dats sets.
X_train = train[:,1:].astype('float32')
y_train = train[:,0].astype('float32')
X_test  = test[:,1:].astype('float32')
y_test  = test[:,0].astype('float32')

# Normalization.
X_train = X_train/255
X_test  = X_test/255

# One-Hot Encoding. 
n_classes = 10
y_train = keras.utils.to_categorical(y_train,n_classes)
y_test = keras.utils.to_categorical(y_test,n_classes) 


X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32')

# Checking the Shape after Pre porcessing
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
result = model.fit(X_train, y_train, batch_size=128, epochs=15, verbose=1, validation_data=(X_test, y_test))
# Train Set Accracy vs Test Accuracy.
plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('Model Accuracy')
plt.legend(['train','test'],loc = 'upper left')
plt.show()

# Train Set Loss vs Test Set Accuracy.
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Model Loss')
plt.legend(['train','test'],loc = 'upper left')
plt.show()
