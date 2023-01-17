import tensorflow.keras as keras

import numpy as np

import matplotlib.pyplot as plt
def load_data():

    X_path = '/kaggle/input/cifar-10-training-set-csv/train_images.csv'

    Y_path = '/kaggle/input/cifar-10-training-set-csv/train_labels_numeric.csv'

    

    raw_X = np.loadtxt(X_path, delimiter = ',', skiprows = 0)

    raw_Y = np.loadtxt(Y_path, delimiter = ',', skiprows = 1)

    

    raw_Y = raw_Y[:, 1]

    raw_X = raw_X.reshape(-1, 32, 32, 3)

    raw_X = raw_X/255

    raw_Y = keras.utils.to_categorical(raw_Y, 10)

    return raw_X, raw_Y
X, Y = load_data()
def identity_block(X, filters, dropout_rate):

    F1 , F2 = filters

    F3 = X.shape[-1]

    

    X_shortcut = X

    

    X = keras.layers.Dropout(rate = dropout_rate)(X)

    X = keras.layers.Conv2D(F1, kernel_size = 1, strides = 1,padding = 'valid', data_format = "channels_last")(X)

    X = keras.layers.BatchNormalization(axis = 3)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.Dropout(rate = dropout_rate)(X)

    X = keras.layers.Conv2D(F2, kernel_size = 3, strides = 1,padding = 'same', data_format = "channels_last")(X)

    X = keras.layers.BatchNormalization(axis = 3)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.Dropout(rate = dropout_rate)(X)

    X = keras.layers.Conv2D(F3, kernel_size = 1, strides = 1,padding = 'valid', data_format = "channels_last")(X)

    X = keras.layers.BatchNormalization(axis = 3)(X)

    

    X = keras.layers.Add()([X, X_shortcut])

    X = keras.layers.Activation('relu')(X)

    

    return X
def dense_block(X, dropout_rate):

    #X = keras.layers.AveragePooling2D()(X)

    X = keras.layers.Flatten()(X)

    X = keras.layers.Dropout(rate = dropout_rate)(X)

    X = keras.layers.Dense(10, activation = 'softmax')(X)

    return X
def starting_block(input_shape, dropout_rate):

    X_input = keras.layers.Input(input_shape)

    X = keras.layers.ZeroPadding2D(padding = 3, data_format = "channels_last")(X_input)

    

    X = keras.layers.Dropout(rate = dropout_rate)(X)

    X = keras.layers.Conv2D(filters = 64, kernel_size = 7, strides = 2, data_format = "channels_last")(X)

    X = keras.layers.BatchNormalization(axis = 3)(X)

    X = keras.layers.MaxPooling2D(pool_size = 3, strides = 2, data_format = "channels_last")(X)

    return X_input, X

    

    
def convolutional_block(X, filters, stride, dropout_rate):

    

    X_shortcut = X

    F1, F2, F3 = filters

    

    X = keras.layers.Dropout(rate = dropout_rate)(X)

    X = keras.layers.Conv2D(filters = F1, kernel_size = 1, strides = stride, padding = "valid", data_format = "channels_last")(X)

    X = keras.layers.BatchNormalization(axis = 3)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.Dropout(rate = dropout_rate)(X)

    X = keras.layers.Conv2D(filters = F2, kernel_size = 3, strides = 1, padding = 'same', data_format = "channels_last")(X)

    X = keras.layers.BatchNormalization(axis = 3)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.Dropout(rate = dropout_rate)(X)

    X = keras.layers.Conv2D(filters = F3, kernel_size = 1, strides = 1, padding = "valid", data_format = "channels_last")(X)

    X = keras.layers.BatchNormalization(axis = 3)(X)

    

    

    X_shortcut = keras.layers.Dropout(rate = dropout_rate)(X_shortcut)

    X_shortcut = keras.layers.Conv2D(filters = F3, kernel_size = 1, strides = stride, padding = "valid", data_format = "channels_last")(X_shortcut)

    X_shortcut = keras.layers.BatchNormalization(axis = 3)(X_shortcut)

    

    X = keras.layers.Add()([X, X_shortcut])

    X = keras.layers.Activation('relu')(X)

    

    return X
def get_model(input_shape, dropout_rate_1, dropout_rate_2):

    X_input, X = starting_block(input_shape, dropout_rate_1)

    X = convolutional_block(X, [64, 64, 256], 1,dropout_rate_2)

    X = identity_block(X, [64, 64],dropout_rate_2)

    X = identity_block(X, [64, 64],dropout_rate_2)

    X = convolutional_block(X, [128, 128, 512], 2,dropout_rate_2)

    X = identity_block(X, [128, 128],dropout_rate_2)

    X = identity_block(X, [128, 128],dropout_rate_2)

    X = identity_block(X, [128, 128],dropout_rate_2)

    X = convolutional_block(X, [256, 256, 1024], 2,dropout_rate_2)

    X = identity_block(X, [256, 256],dropout_rate_2)

    X = identity_block(X, [256, 256],dropout_rate_2)

    X = identity_block(X, [256, 256],dropout_rate_2)

    X = identity_block(X, [256, 256],dropout_rate_2)

    X = identity_block(X, [256, 256],dropout_rate_2)

    X = convolutional_block(X, [512, 512, 2048], 2,dropout_rate_2)

    X = identity_block(X, [512, 512],dropout_rate_2)

    X = identity_block(X, [512, 512],dropout_rate_2)

    X = dense_block(X,dropout_rate_2)

    model = keras.models.Model(inputs = X_input, outputs = X)

    return model
learning_rate = 0.0001

batch_size = 2048

epochs = 3000
model = get_model(X.shape[1:], 0.1, 0.5)
model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = model.fit(x = X, y = Y, batch_size = batch_size, epochs = epochs, validation_split = 0.5)
hist.history.keys()
plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.show()

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.show()