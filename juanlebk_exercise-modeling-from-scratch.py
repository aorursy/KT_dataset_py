# import library
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
# Some initialize
img_width, img_height = 28, 28
num_classes = 10
# define function to prepare data
def prep_data(raw_data):
    y = raw_data[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes) # one hot end coding
    x = raw_data[:, 1:]
    num_images = raw_data.shape[0]
    out_x = x.reshape(num_images, img_height, img_width, 1)
    return out_x, out_y

# load data
fashion_file = '../input/fashionmnist/fashion-mnist_train.csv'
fashion_data = np.loadtxt(fname=fashion_file, skiprows=1, delimiter=',')
X, Y = prep_data(fashion_data)
# import library for specify model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
# specify model
fashion_model = Sequential()
fashion_model.add(layer=Conv2D(filters=12, kernel_size=(3,3), activation='relu', input_shape=(img_height, img_width, 1)))
fashion_model.add(layer=Conv2D(filters=12, kernel_size=(3,3), activation='relu'))
fashion_model.add(layer=Conv2D(filters=12, kernel_size=(3,3), activation='relu'))

fashion_model.add(layer=Flatten())
fashion_model.add(layer=Dense(units=100,activation='relu'))
fashion_model.add(layer=Dense(units=num_classes, activation='softmax'))
# compile model
fashion_model.compile()
# Your code to fit the model here