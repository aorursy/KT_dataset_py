import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout



train_file = '../input/train.csv'

raw_data = pd.read_csv(train_file) # training data



img_rows, img_cols = 28, 28

num_classes = 10



def data_prep(raw):

    out_y = keras.utils.to_categorical(raw.label, num_classes) # one-hot encoding

    

    num_imgs = raw.shape[0] # no of rows in dataframe == no of images

    x_as_array = raw.values[:,1:] # taking all data except 1st column for training, returns a numpy array

    x_shaped_arr = x_as_array.reshape(num_imgs, img_rows, img_cols, 1) # making x into a 4-D array

    out_x = x_shaped_arr / 255 # dividing each value by 255 to reduce range to <1

    return out_x, out_y # returning pre-processed data



X, y = data_prep(raw_data)

X.shape

plt.imshow(X[100].reshape([28,28]))

plt.show()



model = Sequential() # creating a new Sequential model to add layers to it

# adding convolution layers, with 20 convolutions, of size 2x2, using ReLU activation function

model.add(Conv2D(30, kernel_size = 3, strides = 2, activation = 'relu', input_shape = (img_rows, img_cols, 1)))

# adding Dropout layer to prevent overfitting. Also, added strides = 2 for the same

model.add(Dropout(0.5))

model.add(Conv2D(30, kernel_size = 3, strides = 2, activation = 'relu')) # don't need input_shape after 1st time

model.add(Dropout(0.5))

model.add(Flatten()) # flattens the input for dense layer next

model.add(Dense(350, activation = 'relu')) # dense layer having 128 neurons

model.add(Dense(num_classes, activation = 'softmax')) # final output layer, having num_classes neurons, and using

# softmax function, which converts any value of input layer into a probability



model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # configuring model for

# training, using accuracy as a metric, adam optimizer and crossentropy loss function
model.fit(X,y, batch_size = 250, epochs = 6, validation_split = 0.2)
test_file = '../input/test.csv'

test_data = pd.read_csv(test_file)

x_test = (test_data.values.reshape(test_data.shape[0], img_rows, img_cols, 1)) / 255

preds_temp = model.predict(x_test)

preds = preds_temp.argmax(axis = -1)

preds



output = pd.DataFrame({'ImageId': test_data.index + 1, 'Label': preds})

output.to_csv('submission.csv', index = False)
test_data.head(10)