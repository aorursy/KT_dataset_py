import numpy as np

import pandas as pd
train = pd.read_csv("../input/train.csv")
X_train_org = np.array(train.drop(['label'], axis=1)).reshape((-1, 28, 28, 1))
from keras.utils.np_utils import to_categorical

y_train_org = train.label

y_train_org = to_categorical(y_train_org, num_classes=10)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train_org, y_train_org, test_size = 0.01)
import matplotlib.pyplot as plt

plt.imshow(X_train_org[0,:,:,0])
from keras import Sequential

from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten



model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', \

                 input_shape=(28, 28, 1), name='conv2d_1'))



model.add(MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool_1'))



model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', name='conv2d_2'))



model.add(MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool_2'))



model.add(Flatten(name='flatten_1'))



model.add(Dense(256, activation='relu', name='dense_1'))

model.add(Dropout(0.4, name='dropout_1'))

model.add(BatchNormalization())



model.add(Dense(10, activation='softmax', name='dense_2'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

        rescale=1/255, # dividing all pixel value by 255, to make sure our input value is in (0,1)

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

)
batch_size = 64

data_flow = datagen.flow(x=X_train, y=y_train, batch_size=batch_size)

history = model.fit_generator(data_flow, steps_per_epoch=len(X_train_org) // batch_size, \

                              epochs=10, validation_data=(X_val, y_val))
X_test = np.array(pd.read_csv("../input/test.csv")).reshape((-1, 28, 28, 1))

# When predicting, remember to divide test image pixel value by 255, too.

X_test = X_test / 255
y_test = model.predict(X_test, verbose=1)
y_test = np.argmax(y_test, axis=-1)
y_test = pd.DataFrame(y_test, index=range(1, len(y_test)+1), columns=['Label'])
y_test.to_csv("submission.csv", index_label='ImageId')