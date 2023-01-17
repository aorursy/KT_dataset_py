# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from glob import glob 
import seaborn as sns
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(os.listdir("../input/chest-xray-pneumonia/chest_xray"))
train = "../input/chest-xray-pneumonia/chest_xray/train"
val = "../input/chest-xray-pneumonia/chest_xray/val"
test = "../input/chest-xray-pneumonia/chest_xray/test"
plt.figure(1, figsize = (15 , 7))
plt.subplot(1 , 2 , 1)
img = glob(train+"/PNEUMONIA/*.jpeg") #Getting an image in the PNEUMONIA folder
img = np.asarray(plt.imread(img[0]))
plt.title('PNEUMONIA X-RAY')
plt.imshow(img)

plt.subplot(1 , 2 , 2)
img = glob(train+"/NORMAL/*.jpeg") #Getting an image in the NORMAL folder
img = np.asarray(plt.imread(img[0]))
plt.title('NORMAL CHEST X-RAY')
plt.imshow(img)

plt.show()
train_datagen = ImageDataGenerator(rescale            = 1/255,
                                   shear_range        = 0.2,
                                   zoom_range         = 0.2,
                                   horizontal_flip    = True,
                                   rotation_range     = 40,
                                   width_shift_range  = 0.2,
                                   height_shift_range = 0.2)

test_datagen = ImageDataGenerator(rescale = 1/255)
training_set = train_datagen.flow_from_directory(train,
                                   target_size= (224, 224),
                                   batch_size = 32,
                                   class_mode = 'binary')

val_set = test_datagen.flow_from_directory(val,
                                   target_size=(224, 224),
                                   batch_size = 32,
                                   class_mode ='binary')

test_set = test_datagen.flow_from_directory(test,
                                   target_size= (224, 224),
                                   batch_size = 32,
                                   class_mode = 'binary')
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))


model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nb_train_samples = 5217
nb_validation_samples = 17
epochs = 20
batch_size = 16
model_train = model.fit_generator(training_set,
                         steps_per_epoch=nb_train_samples // batch_size,
                         epochs=epochs,
                         validation_data=val_set,
                         validation_steps=nb_validation_samples // batch_size)
# model.evaluate_generator(generator=val_set,
# steps=100)
# evaluate the model
scores = model.evaluate_generator(test_set)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# Get the predictions on test set
preds = model.predict(test_set)
preds = np.squeeze((preds > 0.5).astype('int'))
path_val_image = "../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg" # copied path of the Pneumonia X-ray image

from keras.preprocessing import image

img = image.load_img(path_val_image, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

classes = model.predict(x)
print(classes)
if classes>0.5:
    print(" pneumonia")
else:
    print("normal")