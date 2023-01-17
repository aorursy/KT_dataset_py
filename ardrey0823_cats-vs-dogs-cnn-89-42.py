import numpy as np 

import pandas as pd 

import os

from os.path import join

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Dense, MaxPool2D

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.applications.resnet50 import preprocess_input

import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping
# Loading training data



dog_folderpath = '../input/cat-and-dog/training_set/training_set/dogs'

cat_folderpath = '../input/cat-and-dog/training_set/training_set/cats'



dog_imgpaths = [join(dog_folderpath, file) for file in os.listdir(dog_folderpath) if 'jpg' in file]

cat_imgpaths = [join(cat_folderpath, file) for file in os.listdir(cat_folderpath) if 'jpg' in file]



# Loading Testing data



dog_test_path = '../input/cat-and-dog/test_set/test_set/dogs'

cat_test_path = '../input/cat-and-dog/test_set/test_set/cats'



dog_test_img_paths = [join(dog_test_path, file) for file in os.listdir(dog_test_path) if 'jpg' in file]

cat_test_img_paths = [join(cat_test_path, file) for file in os.listdir(cat_test_path) if 'jpg' in file]
def data_prep(img_paths, img_rows, img_cols):

    imgs = [load_img(path, target_size=(img_rows, img_cols)) for path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    output = img_array/255

    return output
# Declaring constant variables



img_rows, img_cols = 100, 100

num_classes = 2
# Training Data Preprocessing

dog_data = data_prep(dog_imgpaths, img_rows, img_cols)

cat_data = data_prep(cat_imgpaths, img_rows, img_cols)



# Testing Data Preprocessing

dog_test_img = data_prep(dog_test_img_paths, img_rows, img_cols)

cat_test_img = data_prep(cat_test_img_paths, img_rows, img_cols)
fig, ax = plt.subplots(1, 2, figsize=(10,10))

ax[0].set_title('Dog')

ax[0].imshow(dog_data[0])

ax[1].set_title('Cat')

ax[1].imshow(cat_data[0])
# Preparing labels and features

dog_label = np.ones(4005)

cat_label = np.zeros(4000)

y = np.concatenate((dog_label, cat_label))

X = np.concatenate((dog_data, cat_data))

y = to_categorical(y, num_classes)



dog_test_label = np.ones(1012)

cat_test_label = np.zeros(1011)

y_test = np.concatenate((dog_test_label, cat_test_label))

X_test = np.concatenate((dog_test_img, cat_test_img))

y_test = to_categorical(y_test, num_classes)



class_list = ['cat' , 'dog']
# Adding Layers to the model

model = Sequential()



model.add(Conv2D(50, kernel_size =(3,3), input_shape=(img_rows,img_cols,3), padding='same', activation='relu'))

model.add(MaxPool2D(2,2))

model.add(BatchNormalization())





model.add(Conv2D(50, kernel_size =(3,3),padding='same', activation='relu'))

model.add(MaxPool2D(2,2))

model.add(BatchNormalization())







model.add(Conv2D(60, kernel_size =(3,3),padding='same', activation='relu'))

model.add(MaxPool2D(2,2))

model.add(BatchNormalization())





model.add(Conv2D(80, kernel_size =(3,3),padding='same', activation='relu'))

model.add(MaxPool2D(2,2))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Conv2D(80, kernel_size =(3,3),padding='same', activation='relu'))

model.add(MaxPool2D(2,2))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))



model.add(Dropout(0.50))

model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss=categorical_crossentropy, optimizer=Adam(0.001), metrics=['accuracy'])
# Instantiate stopper for model training

stopper = EarlyStopping(monitor='val_loss', patience=2, mode='auto')
history = model.fit(X, y, epochs=30, callbacks=[stopper], validation_data=(X_test, y_test))
model.evaluate(X_test, y_test)
preds = model.predict(X_test)
predicted_class = [np.argmax(pred) for pred in preds]
plt.title('Predicted: {}'.format(class_list[predicted_class[1013]]))

plt.imshow(X_test[1013])