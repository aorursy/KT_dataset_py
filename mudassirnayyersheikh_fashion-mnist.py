from keras import *
from keras.layers import *
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("../input/fashion-mnist_train.csv") 
data = df.values.astype(np.float32)
np.random.shuffle(data)

total_imgs = 60000

x_train = data[0:total_imgs, 1:].reshape(-1,28,28,1)

y_train = to_categorical(data[0:total_imgs, 0])
print(np.shape(x_train))
print(np.shape(y_train))
print(x_train.shape[1:])
model = Sequential()
model.add(Conv2D(32, (3, 3), kernel_initializer=initializers.glorot_normal(seed=None), input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3), kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3), kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.00000001, decay=0.1, amsgrad=False),
              metrics=['accuracy'])
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
trained_model = model.fit(x_train, y_train, validation_split=0.2, epochs=15, batch_size=32)
# list all data in history
# summarize history for accuracy
plt.figure(figsize=(15,10))
plt.plot(trained_model.history['acc'], color = 'r')
plt.plot(trained_model.history['val_acc'], color = 'b')

plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.figure(figsize=(15,10))
plt.plot(trained_model.history['loss'], color = 'r')
plt.plot(trained_model.history['val_loss'], color = 'b')
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
