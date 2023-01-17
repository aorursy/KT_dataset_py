# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import cv2
import random
import zipfile
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
mantled_dir = os.path.join("/kaggle/input/10-monkey-species/training/training/n0/")
patas_monkey_dir = os.path.join("/kaggle/input/10-monkey-species/training/training/n1/")
bald_uakari_dir = os.path.join("/kaggle/input/10-monkey-species/training/training/n2/")
japanese_macaque_dir = os.path.join("/kaggle/input/10-monkey-species/training/training/n3/")
pygmy_marmoset_dir = os.path.join("/kaggle/input/10-monkey-species/training/training/n4/")
white_headed_capuchin_dir = os.path.join("/kaggle/input/10-monkey-species/training/training/n5/")
silvery_marmoset_dir = os.path.join("/kaggle/input/10-monkey-species/training/training/n6/")
common_squirrel_monkey_dir = os.path.join("/kaggle/input/10-monkey-species/training/training/n7/")
black_headed_night_monkey_dir = os.path.join("/kaggle/input/10-monkey-species/training/training/n8/")
nilgiri_langur_dir = os.path.join("/kaggle/input/10-monkey-species/training/training/n9/")

mantled_names = os.listdir(mantled_dir)
print(mantled_names[:10])

patas_monkey_names = os.listdir(patas_monkey_dir)
print(patas_monkey_names[:10])

bald_uakari_names = os.listdir(bald_uakari_dir)
print(bald_uakari_names[:10])

japanese_macaque_names = os.listdir(japanese_macaque_dir)
print(japanese_macaque_names[:10])

pygmy_marmoset_names = os.listdir(pygmy_marmoset_dir)
print(pygmy_marmoset_names[:10])

white_headed_capuchin_names = os.listdir(white_headed_capuchin_dir)
print(white_headed_capuchin_names[:10])

silvery_marmoset_names = os.listdir(silvery_marmoset_dir)
print(silvery_marmoset_names[:10])

common_squirrel_monkey_names = os.listdir(common_squirrel_monkey_dir)
print(common_squirrel_monkey_names[:10])

black_headed_night_monkey_names = os.listdir(black_headed_night_monkey_dir)
print(black_headed_night_monkey_names[:10])

nilgiri_langur_names = os.listdir(nilgiri_langur_dir)
print(nilgiri_langur_names[:10])
pic_index = 2

mantled_names_next = [os.path.join(mantled_dir,fname)
                     for fname in mantled_names[pic_index-2:pic_index]]

patas_monkey_names_next = [os.path.join(patas_monkey_dir,fname)
                          for fname in patas_monkey_names[pic_index-2:pic_index]]

bald_uakari_names_next = [os.path.join(bald_uakari_dir,fname)
                                      for fname in bald_uakari_names[pic_index-2:pic_index]]

japanese_macaque_names_next = [os.path.join(japanese_macaque_dir,fname)
                              for fname in japanese_macaque_names[pic_index-2:pic_index]]

pygmy_marmoset_names_next = [os.path.join(pygmy_marmoset_dir,fname)
                                         for fname in pygmy_marmoset_names[pic_index-2:pic_index]]

white_headed_capuchin_names_next = [os.path.join(white_headed_capuchin_dir,fname)
                                   for fname in white_headed_capuchin_names[pic_index-2:pic_index]]

silvery_marmoset_names_next = [os.path.join(silvery_marmoset_dir,fname)
                              for fname in silvery_marmoset_names[pic_index-2:pic_index]]

common_squirrel_monkey_names_next = [os.path.join(common_squirrel_monkey_dir,fname)
                                    for fname in common_squirrel_monkey_names[pic_index-2:pic_index]]

black_headed_night_monkey_names_next = [os.path.join(black_headed_night_monkey_dir,fname)
                                       for fname in black_headed_night_monkey_names[pic_index-2:pic_index]]

nilgiri_langur_names_next = [os.path.join(nilgiri_langur_dir,fname)
                            for fname in nilgiri_langur_names[pic_index-2:pic_index]]

for i, img_path in enumerate(mantled_names_next + patas_monkey_names_next + bald_uakari_names_next + japanese_macaque_names_next + pygmy_marmoset_names_next + 
                            white_headed_capuchin_names_next + silvery_marmoset_names_next + common_squirrel_monkey_names_next + 
                            black_headed_night_monkey_names_next + nilgiri_langur_names_next):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis("Off")
    plt.show()
model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(32,(3,3),activation="relu"),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(64,(3,3),activation="relu"),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(64,(3,3),activation="relu"),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(512,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])
model.summary()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode="nearest")

valid_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory("/kaggle/input/10-monkey-species/training/training/",
                                                   batch_size=20,
                                                   target_size=(150,150),
                                                   class_mode="categorical")


valid_generator = valid_datagen.flow_from_directory("/kaggle/input/10-monkey-species/validation/validation/",
                                                   batch_size=20,
                                                   target_size=(150,150),
                                                   class_mode="categorical")
history = model.fit(train_generator,
                   validation_data=valid_generator,
                   steps_per_epoch=20,
                   epochs=200,
                   verbose=1)
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

plt.plot(epochs,acc,"r",label="Training Accuracy")
plt.plot(epochs,val_acc,"b",label="Validation Accuracy")
plt.title("Training And Validation Accuracy")

plt.legend()
plt.figure()

plt.plot(epochs,loss,"r",label="Training Loss")
plt.plot(epochs,val_loss,"b",label="Validation Loss")
plt.title("Training And Validation Loss")

plt.legend()
plt.show()
from keras.preprocessing import image

path = "/kaggle/input/monkey-class-pa/pa_1.jpg"
cv_img = cv2.imread(path)

img = image.load_img(path,target_size=(150,150))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)

images = np.vstack([x])
classes = model.predict(images,batch_size=10)
print(classes)

plt.imshow(cv_img)
