import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
%matplotlib inline
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
train_dir ="../input/fer2013/train/"
df = {}
for expression in os.listdir(train_dir):
    directory = train_dir + expression
    df[expression] = len(os.listdir(directory))
df = pd.DataFrame(df, index=[0])
df
img_size = 48
plt.figure(0, figsize=(12,20))
index = 0
for link in os.listdir("../input/fer2013/test"):
    for i in range(1,3):
        index += 1
        plt.subplot(7,5,index)
        img = load_img("../input/fer2013/test/" + link + "/" +os.listdir("../input/fer2013/test/" + link)[i], target_size=(img_size, img_size))
        plt.imshow(img, cmap="gray")

plt.tight_layout()

# Generates new train and test images
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                                '../input/fer2013/train',
                                target_size=(img_size,img_size),
                                batch_size=64,
                                color_mode="grayscale",
                                class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
                                '../input/fer2013/test',
                                target_size=(img_size,img_size),
                                batch_size=64,
                                color_mode="grayscale",
                                class_mode='categorical')
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation = 'softmax'))
model.summary()
#opt = SGD(lr=0.01, momentum=0.9)
#Compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_history = model.fit(
        train_generator,
        steps_per_epoch= 64, #train_generator.batch_size,
        epochs=50,
        validation_data=test_generator,
        validation_steps= 64 #test_generator.batch_size
)

