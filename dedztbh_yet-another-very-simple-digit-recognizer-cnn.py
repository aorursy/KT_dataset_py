# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout



img_rows, img_cols = 28, 28

num_classes = 10



def data_prep(raw):

    out_y = keras.utils.to_categorical(raw.label, num_classes)



    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)



    out_x = x_shaped_array / 255

    return out_x, out_y



train_file = "../input/train.csv"

raw_data = pd.read_csv(train_file)



x, y = data_prep(raw_data)



from PIL import Image

from matplotlib.pyplot import imshow



def generate_image(raw):

    showing = np.zeros((img_rows, img_cols))

    for r in range(0, len(raw)):

        for c in range(0, len(raw[r])):

            showing[r][c] = raw[r][c][0]

    return Image.fromarray(showing)



img = generate_image(x[0])

imshow(img)

digit_model = Sequential([

    Conv2D(filters=32, kernel_size=(5,5), activation="relu", input_shape=(img_rows, img_rows, 1)),

    Conv2D(filters=32, kernel_size=(5,5), activation="relu"),

    Conv2D(filters=32, kernel_size=(5,5), activation="relu", strides=2),

    Dropout(0.25),

    Conv2D(filters=64, kernel_size=(3,3), activation="relu"),

    Conv2D(filters=64, kernel_size=(3,3), activation="relu"),

    Conv2D(filters=64, kernel_size=(3,3), activation="relu", strides=2),

    Dropout(0.25),

    Flatten(),

    Dense(256, activation="relu"),

    Dropout(0.5),

    Dense(num_classes, activation="softmax")

])
digit_model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split



validation_split = 0.1

epochs = 50

batch_size = 512



#Prepare for some basic data augmentation

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=validation_split, stratify=y)



data_generator_with_aug = ImageDataGenerator(width_shift_range=0.1,

                                             height_shift_range=0.1,

                                             zoom_range = 0.1,

                                             rotation_range=10,

                                             validation_split=validation_split)



data_generator_no_aug = ImageDataGenerator(validation_split=validation_split)



train_generator = data_generator_with_aug.flow(

    train_x, train_y,

    batch_size=batch_size,

    subset='training')



validation_generator = data_generator_no_aug.flow(

    val_x, val_y,

    subset='validation') 



training_history = digit_model.fit_generator(train_generator,

                                             steps_per_epoch=len(train_x) / batch_size,

                                             validation_data=validation_generator,

                                             epochs=epochs)



# No Augmentation 

# training_history = digit_model.fit(x, y,

#           batch_size=batch_size,

#           epochs=epochs,

#           validation_split = validation_split)
import matplotlib.pyplot as plt



# summarize history for accuracy

plt.plot(training_history.history['acc'])

plt.plot(training_history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(training_history.history['loss'])

plt.plot(training_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
def data_prep_test(raw):

    num_images = raw.shape[0]

    x_shaped_array = raw.values.reshape(num_images, img_rows, img_cols, 1)

    out_x = x_shaped_array / 255

    return out_x



raw_data_test = pd.read_csv("../input/test.csv")



x_test = data_prep_test(raw_data_test)
showing_test = np.zeros((img_rows, img_cols))



img_test = generate_image(x_test[0])

imshow(img_test)
result = digit_model.predict(x_test)
np.argmax(result[0])
collapsed_results = np.transpose(np.asmatrix(list(map(lambda x: np.argmax(x), result))))

    

# add indicies

indicies = np.transpose(np.asmatrix(range(1,collapsed_results.shape[0]+1)))



# print(collapsed_results.shape)

# print(indicies.shape)



output = np.append(np.asmatrix(["ImageId", "Label"]), np.append(indicies, collapsed_results, axis=1), axis=0)
pd.DataFrame(output).to_csv("out.csv",index=False,header=False)