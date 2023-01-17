import tensorflow as tf

tf.__version__
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout



img_rows, img_cols = 28, 28

num_classes = 10



def data_prep(raw):

    out_y = to_categorical(raw.label, num_classes)



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



model.compile(loss=categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])
model.summary()
model.fit(x, y,

          batch_size=1024,

          epochs=100,

          validation_split = 0.2)