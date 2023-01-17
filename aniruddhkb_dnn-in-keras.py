import numpy as np
import keras
import matplotlib.pyplot as plt
def load_train_mnist():
    path = '/kaggle/input/digit-recognizer/train.csv'
    raw_data = np.loadtxt(path, delimiter = ',', skiprows = 1)
    raw_X = raw_data[:, 1:]
    raw_Y = raw_data[:, 0]
    return (raw_X, raw_Y)
train_X, train_Y = load_train_mnist()
def get_model():
    X_input = keras.layers.Input((784,))
    #Yhat = keras.layers.Dense(512, activation = 'relu')(X_input)
    Yhat = keras.layers.Dense(256, activation = 'relu')(X_input)
    Yhat = keras.layers.Dense(64, activation = 'relu')(Yhat)
    Yhat = keras.layers.Dense(10, activation = 'softmax')(Yhat)
    
    model = keras.models.Model(inputs = X_input, outputs = Yhat)
    
    return model
learning_rate = 0.001
batch_size = 2048
num_epochs = 1000
model = get_model()

model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_Y_one_hotted = keras.utils.to_categorical(train_Y, 10)
hist = model.fit(x = train_X, y = train_Y_one_hotted, batch_size = batch_size, epochs = num_epochs, validation_split = 0.5)
plt.plot(hist.history['loss'])
plt.plot(hist.history['accuracy'])
plt.show()
