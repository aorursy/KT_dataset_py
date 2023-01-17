import numpy as np

from tensorflow.keras.utils import to_categorical



img_rows, img_cols = 28, 28

num_classes = 10



def prep_data(raw):

    y = raw[:, 0]

    out_y = to_categorical(y, num_classes)

    

    num_images = raw.shape[0]

    x = raw[:,1:]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    

    return out_x, out_y



fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"

fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')

x, y = prep_data(fashion_data)



# set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.exercise_7 import *

print("Setup Complete")
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Flatten, Dense



# your Code Here

fashion_model = Sequential()



# check your answer

q_1.check()
# q_1.solution()
# your code here

fashion_model.add(Conv2D(12, kernel_size=(3, 3), activation="relu", input_shape=(img_rows, img_cols, 1)))



# check your answer

q_2.check()
# q_2.hint()

# q_2.solution()
# add more layers

fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation="relu"))

fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation="relu"))

fashion_model.add(Flatten())

fashion_model.add(Dense(100, activation="relu"))



# add prediction layer

fashion_model.add(Dense(num_classes, activation="softmax"))



# check your answer

q_3.check()
# q_3.solution()
# your code to compile the model in this cell

from tensorflow.keras.losses import categorical_crossentropy

fashion_model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=["accuracy"])



# check your answer

q_4.check()
# q_4.solution()
# your code to fit the model here

from sklearn.model_selection import train_test_split

fashion_model.fit(x, y, batch_size=100, epochs=4, validation_split=0.2)



# check your answer

q_5.check()
# q_5.solution()
second_fashion_model = Sequential()



# add first layer

second_fashion_model.add(Conv2D(12, kernel_size=(3, 3), activation="relu", input_shape=(img_rows, img_cols, 1)))



# add more layers

second_fashion_model.add(Conv2D(10, kernel_size=(3, 3), activation="relu"))

second_fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation="relu"))

second_fashion_model.add(Flatten())

second_fashion_model.add(Dense(200, activation="relu"))



# add prediction layer

second_fashion_model.add(Dense(num_classes, activation="softmax"))



# compile model

second_fashion_model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=["accuracy"])



# fit model

second_fashion_model.fit(x, y, batch_size=100, epochs=6, validation_split=0.2)



# don't remove this line (ensures comptibility with tensorflow 2.0)

second_fashion_model.history.history['val_acc'] = second_fashion_model.history.history['val_accuracy']



# check your answer

q_6.check()
# q_6.solution()