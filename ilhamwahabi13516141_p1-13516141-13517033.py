from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D
# from keras.layers.convolutional import Convolution2D, MaxPooling2D

#Parameter untuk model
wInput = 256
nfilter = 64
conv_size = 3
pool_size = 112
n_stride = 2

#inisialisasi model
# model = Sequential()
# Load image
train_ds = image_dataset_from_directory(
    directory='../input/weather/test',
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256)
)
# Filter 1 dan 2
model = Sequential()
# model.add(Conv2D(nfilter, conv_size, conv_size, border_mode ="same", input_shape=(wInput, wInput, 3))
# model.add(Conv2D(nfilter, conv_size, conv_size, border_mode ="same", input_shape=(wInput, wInput, 64))

# max pooling
model.add(MaxPooling2D(pool_size=(pool_size, pool_size),strides=(n_stride,n_stride)))

# Fully Connected Layer
model.add(Dense(512))
model.add(Dense(512))
model.add(Dense(4, activation="softmax"))