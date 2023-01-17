import os

from os import listdir
listdir("../input/intel-image-classification/seg_train/seg_train")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train = '../input/intel-image-classification/seg_train/seg_train'

data = ImageDataGenerator(rescale=1./255)
Train = data.flow_from_directory(
    train,
    target_size = (150,150),
    class_mode = 'categorical'
)

test = '../input/intel-image-classification/seg_test/seg_test'

Val = data.flow_from_directory(
    test,
    target_size = (150,150),
    class_mode = 'categorical'
)

pred = '../input/intel-image-classification/seg_pred/seg_pred'

pred = data.flow_from_directory(
    test,
    target_size = (150,150),
    class_mode = 'categorical'
)
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.optimizers import SGD

model=Sequential([])
model.add(Conv2D(32,(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(.1))

model.add(Dense(64,activation='relu'))
model.add(Dropout(.1))

model.add(Dense(6,activation='softmax'))

# compile model
opt = SGD(lr=0.01, momentum=0.7)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit_generator(Train, epochs=10, validation_data = Val, verbose = 1)
model.predict(pred)
