import keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
import numpy as np
from keras import backend as k
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train.shape
x_test.shape
# import matplotlib.pyplot as plt
# fig = plt.figure()

# for i in range(9):
#   plt.subplot(3,3,i+1)
#   plt.tight_layout()
#   plt.imshow(X_train[i],
#              cmap='gray',interpolation='none')
#   plt.title('Digits {}'.format(y_train[i]))
#   plt.xticks([])
#   plt.yticks([])
print('X_train shape', x_train.shape)
print('Y_train shape', y_train.shape)
print('X_test shape', x_test.shape)
print('Y_test shape', y_test.shape)
num_classes = 10
input_shape= (28,28,1)
#Reshape
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
#images have shape of (28,28,1)
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
#convert class vectors to binary class metrics
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train[0]
y_train[:10]
#Build the model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape= input_shape))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
#Train the model
batch_siz = 128
num_epoch = 10
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model_log = model.fit(x_train,y_train,
                      batch_size=batch_siz,
                      epochs= num_epoch,
                      verbose=1,
                      validation_data=(x_test,y_test))
score= model.evaluate(x_test,y_test,verbose= 0)

print('Test Loss' , score[0])
print('Test accuracy' , score[1])
import os

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_log.history['accuracy'])
plt.plot(model_log.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['train','test'],loc='lower right')


plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='lower right')
plt.tight_layout()