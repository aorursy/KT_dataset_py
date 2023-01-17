!pip install visualkeras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, InputLayer, ZeroPadding2D

import visualkeras  # <--- Import visualkeras


# create VGG16
image_size = 224
model = Sequential()
model.add(InputLayer(input_shape=(image_size, image_size, 3)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
model.add(visualkeras.SpacingDummyLayer())  # These are dummy layers to tell visualkeras when to insert additional padding between layers, you can remove them if you want.

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
model.add(visualkeras.SpacingDummyLayer())  # dummy

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
model.add(visualkeras.SpacingDummyLayer())  # dummy

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(visualkeras.SpacingDummyLayer())  # dummy

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D())
model.add(visualkeras.SpacingDummyLayer())  # dummy

model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))
visualkeras.layered_view(model, type_ignore=[ZeroPadding2D, Flatten])