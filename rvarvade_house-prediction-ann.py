from keras.models import Sequential 

from keras.layers import Dense

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
import pandas as pd

df = pd.read_csv("../input/housepricedata/housepricedata.csv")

df.head(7)
dataset = df.values

dataset
X = dataset[:,0:10]

Y = dataset[:,10]
min_max_scaler = MinMaxScaler()

X_scale = min_max_scaler.fit_transform(X)

X_scale
X_train, X_test, Y_train, Y_test = train_test_split(

    X_scale, Y, test_size=0.2, random_state=1)

X_train, X_val, Y_train, Y_val = train_test_split(

    X_train, Y_train, test_size=0.1, random_state=1)

print(X_train.shape, X_test.shape, X_val.shape, Y_train.shape, Y_test.shape, Y_val.shape)
from keras.layers import Dense, Conv2D, Flatten



from keras.models import Sequential

from keras.layers import Dense



# The models architechture 4 layers, 3 with 32 neurons and activation function = relu function, 

# the last layer has 1 neuron with an activation function = sigmoid function which returns a value btwn 0 and 1

# The input shape/ input_dim = 10 the number of features in the data set

model = Sequential([

    Dense(32, activation='relu', input_shape=(10,)),

    Dense(32, activation='relu'),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid')

])



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, Y_train,

          batch_size=32, epochs=100,

          validation_data=(X_val, Y_val))
#visualize the training loss and the validation loss to see if the model is overfitting

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
