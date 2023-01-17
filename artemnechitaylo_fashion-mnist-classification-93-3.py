import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
train_data = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
test_data = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
labels = ['T-shirt/top'
,'Trouser'
,'Pullover'
,'Dress'
,'Coat'
,'Sandal'
,'Shirt'
,'Sneaker'
,'Bag'
,'Ankle boot']
train_data.head()
y_train = train_data['label']
y_test = test_data['label']
train_data.drop(['label'], axis='columns', inplace=True)
test_data.drop(['label'], axis='columns', inplace=True)
train_data.head()
x_train = np.array(train_data).astype('float32')
x_test = np.array(test_data).astype('float32')
x_train = x_train.reshape(60000, 28, 28)
x_train = x_train / 255.
x_test = x_test.reshape(10000, 28, 28)
x_test = x_test / 255.
for i in range(9):
    plt.subplot(3, 3, (i + 1))
    plt.imshow(x_train[i][:,:], cmap=plt.get_cmap('gray'))
    plt.title(labels[y_train[i]])
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
best_model_path = 'fashion_mnist_model.h5'
checkpoint_callback = ModelCheckpoint(best_model_path,
                                     monitor='val_accuracy',
                                     save_best_only=True,
                                     verbose=1)
reduce_callback = ReduceLROnPlateau(monitor='val_accuracy',
                                   patience=3,
                                   factor=0.5,
                                   min_lr=0.00001,
                                   verbose=1)
callbacks_list = [checkpoint_callback, reduce_callback]
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
             metrics=['accuracy'],
             optimizer='adam')
history = model.fit(x_train,
          y_train,
          epochs=25,
          batch_size=100,
          validation_split=0.2,
          callbacks=callbacks_list,
          verbose=1)
model.load_weights(best_model_path)
results = model.evaluate(x_test,
               y_test,
               verbose=1)
plt.plot(history.history['accuracy'], 
         label='accuracy')
plt.plot(history.history['val_accuracy'],
         label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Percentage of correct responses')
plt.legend()
plt.show()
plt.plot(history.history['loss'], 
         label='Loss')
plt.plot(history.history['val_loss'],
         label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('Percentage of correct responses')
plt.legend()
plt.show()
counts = [results[0], results[1]]
groups = ['Testing model loss\n' + str(int(counts[0] * 10000) / 100) + '%',
          'Testing model accuracy\n' + str(int(counts[1] * 10000) / 100) + '%']

colors = ['r', 'b']
plt.title('Results of testing')

width = len(counts) * 0.3
plt.bar(groups, counts, width=width, color=colors, alpha=0.6, bottom=2, linewidth=2)