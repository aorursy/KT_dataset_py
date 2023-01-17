import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow import keras



img_rows, img_cols = 28, 28

num_classes = 10



def prep_data(raw):

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y



fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"

fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')

x, y = prep_data(fashion_data)



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.exercise_7 import *

print("Setup Complete")
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D



# Your Code Here



fashion_model = Sequential()

____

q_1.check()
#q_1.solution()
# Your code here

fashion_model.add(Conv2D(12,kernel_size =(3,3),

                         activation = 'relu',

                         input_shape = (img_rows, img_cols, 1)

                        )

                 )

____

q_2.check()
# q_2.hint()

#q_2.solution()
# Your code here



fashion_model.add(Conv2D(20,kernel_size =(3,3),activation = 'relu'))

fashion_model.add(Conv2D(20,kernel_size =(3,3),activation = 'relu'))



fashion_model.add(Flatten())



fashion_model.add(Dense(100, activation='relu'))



fashion_model.add(Dense(num_classes, activation='softmax'))





____



q_3.check()
# q_3.solution()

# Your code to compile the model in this cell



fashion_model.compile(loss="categorical_crossentropy",

                     optimizer='adam',

                     metrics =['accuracy']

                     )



____

q_4.check()
# q_4.solution()




fashion_model.fit(x,y,

                 batch_size = 100,

                 epochs = 4,

                 validation_split = 0.2)



# Your code to fit the model here

____

q_5.check()
#q_5.solution()
# Your code below

second_fashion_model = Sequential()



second_fashion_model.add(Conv2D(20,kernel_size =(3,3),

                         activation = 'relu',

                         input_shape = (img_rows, img_cols, 1)))



second_fashion_model.add(Conv2D(20,kernel_size =(3,3),activation = 'relu'))

second_fashion_model.add(Flatten())

second_fashion_model.add(Dense(200, activation='relu'))

second_fashion_model.add(Dense(num_classes, activation='softmax'))



second_fashion_model.compile(loss="categorical_crossentropy",

                     optimizer='adam',

                     metrics =['accuracy']

                     )



second_fashion_model.fit(x,y,

                 batch_size = 100,

                 epochs = 4,

                 validation_split = 0.2)



____



q_6.check()
#another test

third_fashion_model = Sequential()



third_fashion_model.add(Conv2D(12,kernel_size =(3,3),

                         activation = 'relu',

                         input_shape = (img_rows, img_cols, 1)))



third_fashion_model.add(Conv2D(20,kernel_size =(3,3),activation = 'relu'))



third_fashion_model.add(Conv2D(20,kernel_size =(3,3),activation = 'relu'))



third_fashion_model.add(Conv2D(20,kernel_size =(3,3),activation = 'relu'))



third_fashion_model.add(Conv2D(20,kernel_size =(3,3),activation = 'relu'))



third_fashion_model.add(Flatten())

#third_fashion_model.add(Dense(200, activation='relu'))

third_fashion_model.add(Dense(num_classes, activation='softmax'))



third_fashion_model.compile(loss="categorical_crossentropy",

                     optimizer='adam',

                     metrics =['accuracy']

                     )



third_fashion_model.fit(x,y,

                 batch_size = 100,

                 epochs = 4,

                 validation_split = 0.2)

#another test

third_fashion_model = Sequential()



third_fashion_model.add(Conv2D(12,kernel_size =(3,3),

                         activation = 'relu',

                         input_shape = (img_rows, img_cols, 1)))



third_fashion_model.add(Conv2D(20,kernel_size =(3,3),activation = 'relu'))



third_fashion_model.add(Conv2D(20,kernel_size =(3,3),activation = 'relu'))

third_fashion_model.add(Conv2D(20,kernel_size =(3,3),activation = 'relu'))



third_fashion_model.add(Flatten())



third_fashion_model.add(Dense(200, activation='relu'))



third_fashion_model.add(Dense(200, activation='relu'))

third_fashion_model.add(Dense(num_classes, activation='softmax'))



third_fashion_model.compile(loss="categorical_crossentropy",

                     optimizer='adam',

                     metrics =['accuracy']

                     )



third_fashion_model.fit(x,y,

                 batch_size = 100,

                 epochs = 4,

                 validation_split = 0.2)

#q_6.solution()