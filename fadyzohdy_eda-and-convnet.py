# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import cv2
import os
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
# from keras.utils import to_categorical, Sequence
from keras.preprocessing.image import ImageDataGenerator
print(os.listdir("../input/fruits-360_dataset/fruits-360/Training"))
print(os.listdir("../input/fruits-360_dataset/fruits-360/Test"))
def construct_fruits_dict(path):
    _dict = {}
    for fruit in os.listdir(path):
        _dict[fruit] = []
        for img in os.listdir(f'{path}/{fruit}'):
            _dict[fruit].append(f'{path}/{fruit}/{img}')
    return _dict
training_dict = construct_fruits_dict("../input/fruits-360_dataset/fruits-360/Training")
test_dict = construct_fruits_dict("../input/fruits-360_dataset/fruits-360/Test")
fruits = list(training_dict.keys())
fig, axs = plt.subplots(9, 9, figsize=(20,20))
c = 0
for i in range(9):
    for j in range(9):
        f = fruits[c]
        img = plt.imread(training_dict[f][0])
        axs[i,j].set_title(f)
        axs[i,j].imshow(img)
        axs[i,j].axis('off')
        c += 1
train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.25)
train_generator = train_datagen.flow_from_directory(
    "../input/fruits-360_dataset/fruits-360/Training",
    target_size=(100, 100),
    batch_size=128,
    shuffle=True,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    "../input/fruits-360_dataset/fruits-360/Training",
    target_size=(100, 100),
    batch_size=128,
    shuffle=True,
    class_mode='categorical',
    subset='validation'
)
test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = train_datagen.flow_from_directory(
    "../input/fruits-360_dataset/fruits-360/Test",
    target_size=(100, 100),
    batch_size=128,
    class_mode='categorical'
)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(81, activation='sigmoid'))
model.summary()
model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(
    generator=train_generator, 
    epochs=15, 
    steps_per_epoch=int(np.ceil(31018 / 128)),
    validation_data=validation_generator,
    validation_steps=int(np.ceil(10304 / 128))
)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model.evaluate_generator(generator=test_generator, steps=int(np.ceil(13877 / 128)))
model.predict(np.array([plt.imread(training_dict['Lemon'][50]) / 255]))[0]
fruits[np.argmax(model.predict(np.array([plt.imread(training_dict['Lemon'][50]) / 255]))[0] )]
print(os.listdir("../input/fruits-360_dataset/fruits-360/test-multiple_fruits/"))
multi = plt.imread("../input/fruits-360_dataset/fruits-360/test-multiple_fruits/cocos_kiwi_orange_dates_salak_plum_tamarilo_maracuja.jpg") / 255
multi = cv2.resize(multi, (100, 100))
pred = model.predict(np.array([multi]))
fruits[np.argmax(pred[0])]
plt.imshow(multi)
