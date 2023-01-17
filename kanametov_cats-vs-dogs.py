import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import VGG16

from tensorflow.keras import models, layers, optimizers

from tensorflow.keras import callbacks

from tensorflow.keras.preprocessing.image import ImageDataGenerator
len(os.listdir('../input/dogs-vs-cats/train/train'))
train_dir = "../input/dogs-vs-cats/train/train"

test_dir = "../input/dogs-vs-cats/test1/test1"



train_images = os.listdir(train_dir)

train_labels = []

for image in train_images:

    label = image.split('.')[0]

    train_labels.append(label)

df = pd.DataFrame({

    'id': train_images,

    'label': train_labels

})
train_set, val_set = train_test_split(df, test_size=0.2)



train_gen = ImageDataGenerator(rescale=1./255,

                               horizontal_flip=True,

                               rotation_range=45,

                               zoom_range=0.2,

                               shear_range=0.2,

                               height_shift_range=0.2,

                               width_shift_range=0.2,

                               fill_mode='nearest')

val_gen = ImageDataGenerator(rescale=1./255)



train_data = train_gen.flow_from_dataframe(

    train_set, 

    train_dir, 

    x_col='id',

    y_col='label',

    target_size=(150, 150),

    class_mode='binary',

    batch_size=100)



val_data = val_gen.flow_from_dataframe(

    val_set, 

    train_dir, 

    x_col='id',

    y_col='label',

    target_size=(150, 150),

    class_mode='binary',

    batch_size=100)
model = models.Sequential([layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),

                           layers.Conv2D(16, (3, 3), activation='relu'),

                           layers.BatchNormalization(),

                           layers.MaxPooling2D(2, 2),

                           

                           layers.Conv2D(32, (3, 3), activation='relu'),

                           layers.Conv2D(32, (3, 3), activation='relu'),

                           layers.BatchNormalization(),

                           layers.MaxPooling2D(2, 2),

                           

                           layers.Dropout(0.2),

                           

                           layers.Conv2D(64, (3, 3), activation='relu'),

                           layers.Conv2D(64, (3, 3), activation='relu'),

                           layers.BatchNormalization(),

                           layers.MaxPooling2D(2, 2),

                           

                           layers.Flatten(),

                           layers.Dense(256, activation='relu'),

                           layers.Dense(1, activation='sigmoid')])
model.summary()
model.compile(loss='binary_crossentropy',

             optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),

             metrics=['accuracy'])
history = model.fit_generator(train_data,

                             steps_per_epoch=200,

                             epochs=10,

                             validation_data=val_data,

                             validation_steps=50)
model.save('model.h5')
def plot_history(history):

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)

    

    plt.plot(epochs, acc, 'b--', label='acc')

    plt.plot(epochs, val_acc, 'r--', label='val_acc')

    plt.xlabel('epochs')

    plt.ylabel('accuracy')

    plt.legend()

    plt.grid()

    

    plt.figure()

    plt.plot(epochs, loss, 'b--', label='loss')

    plt.plot(epochs, val_loss, 'r--', label='val_loss')

    plt.xlabel('epochs')

    plt.ylabel('accuracy')

    plt.legend()

    plt.grid()
test_images = os.listdir(test_dir)

submission = pd.DataFrame({

    'id': test_images

})





test_gen = ImageDataGenerator(rescale=1./255)



test_data = test_gen.flow_from_dataframe(

    submission, 

    test_dir, 

    x_col='id',

    y_col=None,

    class_mode=None,

    target_size=(150, 150),

    batch_size=100,

    shuffle=False

)



predictions = model.predict_generator(test_data, steps=125)
predictions = [1 if pred > 0.5 else 0 for pred in predictions]



submission['label'] = predictions



label_maps = dict((i, j) for j, i in train_data.class_indices.items())

submission['label'] = submission['label'].replace(label_maps)





submission['label'] = submission['label'].replace({ 'dog': 1, 'cat': 0 })



submission.to_csv('submission.csv', index=False)



submission.head()