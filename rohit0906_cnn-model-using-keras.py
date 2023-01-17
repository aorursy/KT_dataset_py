import numpy as np
import pandas as pd
train=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
test=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
y_train=train['label']
y_test=test['label']
del train['label']
del test['label']
x_train=train.values
x_test=test.values
x_train.shape
x_test.shape
x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)
import matplotlib.pyplot as plt
plt.imshow(x_train[0][:,:,0])
plt.title(y_train[0])
plt.show()
import seaborn as sns
g = sns.countplot(y_train)

y_train.value_counts()
y_train.shape
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train, num_classes=10)
y_test=to_categorical(y_test, num_classes=10)
print(y_train[0])
x_train=x_train/255
x_test=x_test/255
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
model=Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
#model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
import keras.utils
tf.keras.utils.plot_model(model, show_shapes=True)
epochs=50
batch_size=600
history = model.fit(x_train, y_train,
                              epochs = epochs, verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size, validation_split=0.2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
score = model.evaluate(x_test, y_test, verbose=0)
print("Loss on test data",score[0])
print("Accuracy on test data", score[1]*100)