import keras 
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    path = '/kaggle/input/fashionmnist/fashion-mnist_train.csv'
    raw_data = np.loadtxt(path, skiprows = 1, delimiter = ',')
    raw_X = raw_data[:,1:]
    raw_Y = raw_data[:,0]
    return raw_X, raw_Y
raw_X, raw_Y = get_data()
raw_X = raw_X.reshape(-1, 28, 28, 1)
raw_Y = keras.utils.to_categorical(raw_Y, 10)
def get_model():
    X_input = keras.layers.Input((28, 28, 1))
    Yhat = keras.layers.Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = "same", data_format = "channels_last", activation = "relu")(X_input)
    Yhat = keras.layers.Conv2D(filters = 6, kernel_size = 3, strides = 1, padding = "same", data_format = "channels_last", activation = "relu")(Yhat)
    Yhat = keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid", data_format = "channels_last")(Yhat)
    Yhat = keras.layers.Conv2D(filters = 12, kernel_size = 3, strides = 1, padding = "same", data_format = "channels_last", activation = "relu")(Yhat)
    Yhat = keras.layers.Conv2D(filters = 24, kernel_size = 3, strides = 1, padding = "same", data_format = "channels_last", activation = "relu")(Yhat)
    Yhat = keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid", data_format = "channels_last")(Yhat)
    Yhat = keras.layers.Flatten(data_format = "channels_last")(Yhat)
    Yhat = keras.layers.Dense(512, activation = "relu")(Yhat)
    Yhat = keras.layers.Dense(64, activation = "relu")(Yhat)
    Yhat = keras.layers.Dense(10, activation = "softmax")(Yhat)
    
    model = keras.models.Model(inputs = X_input, outputs = Yhat)
    return model
learning_rate = 0.001
batch_size = 1024
epochs = 100
model = get_model()
model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = model.fit(x = raw_X, y = raw_Y, batch_size = batch_size, epochs = epochs, validation_split = 0.5)
