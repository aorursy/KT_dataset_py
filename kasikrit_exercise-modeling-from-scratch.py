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
fashion_data.shape
x, y = prep_data(fashion_data, train_size=50000, val_size=5000)
x.shape
y.shape

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D

# Your Code Here
fashion_model = Sequential()
fashion_model.add(Conv2D(12, kernel_size=(3,3),
                        activation='relu',
                        input_shape=(img_rows, img_cols,1)))

fashion_model.add(Conv2D(12, kernel_size=(3,3),
                        activation='relu'))

fashion_model.add(Conv2D(12, kernel_size=(3,3),
                        activation='relu'))

fashion_model.add(Flatten())
fashion_model.add(Dense(100, activation='relu'))
fashion_model.add(Dense(num_classes, activation='softmax'))


# Your code to compile the model in this cell
fashion_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam', #stochis gradient descent
              metrics=['accuracy'])

# Your code to fit the model here
fashion_model.fit(x, y,
                 batch_size=100,
                 epochs=4,
                 validation_split=0.2)
# evaluate the model
X, Y = x, y
scores = fashion_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (fashion_model.metrics_names[1], scores[1]*100))
# evaluate the model
X, Y = x, y
scores = fashion_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (fashion_model.metrics_names[1], scores[1]*100))
# serialize model to JSON
model_json = fashion_model.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
fashion_model.save_weights("fashion_model.h5")
print("Saved model to disk")
!ls -l