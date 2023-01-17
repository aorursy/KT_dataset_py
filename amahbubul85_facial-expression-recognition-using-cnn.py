import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
%matplotlib inline

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img

from IPython.display import SVG, Image
#!pip install livelossplot
#from livelossplot import PlotLossesKerasTF
import tensorflow as tf
print("Tensorflow version:", tf.__version__)
!apt-get install p7zip-full
!p7zip -d train.7z

!apt-get install p7zip-full
!p7zip -d test.7z
# No of image types in train and test folders
for expression in os.listdir("../input/face-recognition/train/train/"):
    print(str(len(os.listdir("../input/face-recognition/train/train/" + expression))) + " " + expression + " images")
print("\n")
for expression in os.listdir("../input/face-recognition/test/test/"):
    print(str(len(os.listdir("../input/face-recognition/test/test/" + expression))) + " " + expression + " images")
img_size = 48
batch_size = 64

datagen_train = ImageDataGenerator(horizontal_flip=True)

train_generator = datagen_train.flow_from_directory("../input/face-recognition/train/train/",
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory("../input/face-recognition/test/test/",
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)
# This is for prediction at the end
datagen_test = ImageDataGenerator()
test_generator = datagen_test.flow_from_directory(
        "../input/face-recognition/test/test/",
        target_size=(48, 48),
        color_mode="grayscale",
        shuffle = False,
        class_mode='categorical',
        batch_size=1)
print(train_generator.class_indices)
# Initialising the CNN
model = Sequential()

# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

opt = Adam(lr=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
epochs =20
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.00001, mode='auto')
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)
#callbacks = [PlotLossesKerasTF(), checkpoint, reduce_lr]
callbacks = [checkpoint, reduce_lr]

history = model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    callbacks=callbacks
)

# Predicting the images in the test set
filenames = test_generator.filenames
nb_samples = len(filenames)
predict = np.argmax(model.predict_generator(test_generator,steps = nb_samples),axis=1)

# load all images into a list
folder_path = '/content/test/'
img_width=48
img_height=48
images = []
for img in filenames:
    img = os.path.join(folder_path, img)
    img = image.load_img(img, color_mode = "grayscale",target_size=(img_width, img_height))
    images.append(img)

# predicting a random image from test set 
n = random.randint(0,len(predict))
print(predict[n])
print(list(train_generator.class_indices.keys())[list(train_generator.class_indices.values()).index(predict[n])])
images[n]
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)