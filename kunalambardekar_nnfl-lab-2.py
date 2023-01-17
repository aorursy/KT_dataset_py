import numpy as np

import pandas as pd 

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os
os.listdir('../input/nnfl-cnn-lab2/upload')
FAST_RUN = False

IMAGE_WIDTH=128

IMAGE_HEIGHT=128

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3
train_data = pd.read_csv("../input/nnfl-cnn-lab2/upload/train_set.csv", sep=',', encoding = "utf-8")
train_data.head()
train_data['label'].value_counts()
file_nm = os.listdir("../input/nnfl-cnn-lab2/upload/train_images/train_images")

sample_img = random.choice(file_nm)

img = load_img("../input/nnfl-cnn-lab2/upload/train_images/train_images/"+sample_img)

plt.imshow(img)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax')) # 2 because we have cat and dog classes



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



model.summary()
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
callbacks = [early_stop, learning_rate_reduction]
train_data = train_data.astype({"label" : int})

train_df = train_data

train_df.head()
data=train_df

data['label'] = data['label'].replace({0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6'})
train_split, validate_split = train_test_split(data, test_size=.2, random_state=42)
train_split = train_split.reset_index(drop=True)

validate_split = validate_split.reset_index(drop=True)
train_values = train_split.shape[0]

validate_values = validate_split.shape[0]

batch_size=15
print(train_values)

print(validate_values)
train_data_generator = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_data_generator.flow_from_dataframe(

    train_split, 

    "../input/nnfl-cnn-lab2/upload/train_images/train_images", 

    x_col='image_name',

    y_col='label',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_split, 

    "../input/nnfl-cnn-lab2/upload/train_images/train_images", 

    x_col='image_name',

    y_col='label',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
temp_df = train_split.sample(n=1).reset_index(drop=True)

example_generator = train_data_generator.flow_from_dataframe(

    temp_df, 

    "../input/nnfl-cnn-lab2/upload/train_images/train_images", 

    x_col='image_name',

    y_col='label',

    target_size=IMAGE_SIZE,

    class_mode='categorical'

)
plt.figure(figsize=(12, 12))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
epochs=3 if FAST_RUN else 15

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=validate_values//batch_size,

    steps_per_epoch=train_values//batch_size,

    callbacks=callbacks

)
model.save_weights("model.h5")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, epochs, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, epochs, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()


import re



_nsre = re.compile('([0-9]+)')

def natural_sort_key(s):

    return [int(text) if text.isdigit() else text.lower()

            for text in re.split(_nsre, s)]
test_files = (os.listdir("../input/nnfl-cnn-lab2/upload/test_images/test_images"))

test_files.sort(key=natural_sort_key)

test_df = pd.DataFrame({'image_name': test_files})

nb_samples = test_df.shape[0]
test_df.head()
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../input/nnfl-cnn-lab2/upload/test_images/test_images/", 

    x_col='image_name',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['label'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['label'] = test_df['label'].replace(label_map)
test_df['label'].value_counts()
test_df.head()
submission_df = test_df.copy()

submission_df.to_csv('2016A7PS0098G_sub1.csv', index=False)