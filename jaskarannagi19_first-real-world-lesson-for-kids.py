import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras

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
import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model

def build_network(num_actions):
    
    inputs = Input(shape=(img_rows, img_cols, 1,))
    model = Convolution2D(filters=12, kernel_size=(3,3),activation='relu')(inputs)
    
    model = Convolution2D(filters=12, kernel_size=(3,3), activation='relu',)(model)
    model = Flatten()(model)
    model = Dense(activation='relu', units=10)(model)
    q_values = Dense(units=num_actions, activation='linear')(model)
    m = Model(input=inputs, output=q_values)
    return  m

f_model = build_network(num_classes)

# Your code to compile the model in this cell
f_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
# Your code to fit the model here
f_model.fit(x,y, batch_size=100, epochs=4, validation_split=0.2)
fashion_test_file = "../input/fashionmnist/fashion-mnist_test.csv"

fashion_test_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')


def prep_test_data(raw):    

    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x


test = prep_test_data(fashion_test_data)        
prediction = f_model.predict(test)

print(prediction)
print(prediction.shape)