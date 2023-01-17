import os

import shutil

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import keras

from keras import layers, models, optimizers

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data = pd.read_csv('../input/mobile.csv')



image_paths = train_data['image_name'].values
train_data[train_data['Category']==57].head()
sns.countplot(x=train_data['Category'],data=train_data)

plt.title('Count Across Category Types')
cwd = os.getcwd()

print ('Current directory: {}'.format(cwd))
new_folder_paths = ['Train',

                    os.path.join('Train','Mobile')]

for folder_path in new_folder_paths:

    if (os.path.isdir(folder_path) is False):

        os.mkdir(folder_path)
os.listdir('/kaggle/working/Train')
folder_path_dict = {i:'Mobile' for i in range(31, 58, 1)}

for category in range(31,58,1):

        

    category_img_paths = train_data[train_data['Category']==category]['image_path'].values.tolist()

    folder_path = os.path.join('Train', folder_path_dict[category], str(category))



    if (os.path.isdir(folder_path) is False):

        os.mkdir(folder_path)



    for img_path in category_img_paths:

        img_name = img_path.split('/')[1]

        corrected_img_path = "../input/mobile_image_resized/mobile_image_resized/train/"

        

            

        # if there is no image found, just pass and we will have a look later on

#         try:

        shutil.move(os.path.join('../input/mobile_image_resized/mobile_image_resized/train/', img_name), os.path.join('..',folder_path, img_name))

#         print('Added {} to {}'.format(img_name, img_path))

#         except FileNotFoundError:

#             pass

category_img_paths = train_data[train_data['Category']==57]['image_path'].values.tolist()

folder_path = os.path.join('Train', folder_path_dict[57], str(57))

for img_path in category_img_paths:

    img_name = img_path.split('/')[1]

os.path.join('../input/mobile_image_resized/mobile_image_resized/train/', img_name)
os.listdir('../input/mobile_image_resized/mobile_image_resized/train')
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(17, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.adam(),

              metrics=['acc'])



history = model.fit_generator(

      train_generator,

      steps_per_epoch=100,

      epochs=30,

      validation_data=validation_generator,

      validation_steps=50)

model.save('cnn_baseline_beauty.h5')
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



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