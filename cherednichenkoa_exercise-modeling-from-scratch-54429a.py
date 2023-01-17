import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
import tensorflow as tf

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw, train_size, val_size):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data, train_size=50000, val_size=5000)
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D

fashion_model = Sequential()
fashion_model.add(Conv2D(12, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model.add(Conv2D(12, kernel_size=(3, 3),
                 activation='relu'))
fashion_model.add(Conv2D(12, kernel_size=(3, 3),
                 activation='relu'))                  
fashion_model.add(Flatten())
fashion_model.add(Dense(100,activation='relu'))
fashion_model.add(Dense(num_classes,activation='softmax'))
fashion_model.compile(loss = keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
fashion_model.fit(x=x, y=y, batch_size=100, epochs=4, validation_split=0.2)
import matplotlib.pyplot as plt
fashion_file_test = "../input/fashionmnist/fashion-mnist_test.csv"
fashion_data_test = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
fashion_data_test[:1]
x_test,y_test = prep_data(fashion_data_test, train_size=50000, val_size=5000)

x_test[:1].reshape(28,28)
plt.imshow(x_test[:1].reshape(28,28))
plt.imshow(x[0].reshape(28,28), interpolation='nearest')
preds = fashion_model.predict_classes(x_test[:10])
print('Predictions', preds)
print('Labels ',fashion_data_test[:10,0])

