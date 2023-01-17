import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout



#define image size and num classes

img_rows, img_cols = 28, 28

num_classes = 10



def data_prep(raw):

    out_y = keras.utils.to_categorical(raw.label, num_classes)



    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y



train_file = "../input/digit-recognizer/train.csv"

raw_data = pd.read_csv(train_file)



x, y = data_prep(raw_data)



model = Sequential()

model.add(Conv2D(20, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])

model.fit(x, y,

          batch_size=128,

          epochs=2,

          validation_split = 0.2)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
plt.imshow(x[100][:,:,0])
test=pd.read_csv('../input/digit-recognizer/test.csv')
test = test / 255.0

test = test.values.reshape(-1,28,28,1)
# predict results

results= model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)