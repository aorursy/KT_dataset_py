











# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.metrics import classification_report, confusion_matrix

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
%matplotlib inline
import keras
from keras import Model, layers
for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    for filename in filenames:

        print(filename)

        print(f'{dirname}//{filename}')

        img = mpimg.imread(f'{dirname}//{filename}')

        plt.imshow(img)

        plt.show()
from keras.preprocessing import image

from keras.applications.resnet import ResNet152

from keras.applications.resnet import preprocess_input

resnet = ResNet152(weights='imagenet')
import tensorflow as tf
import cv2

import numpy as np
input_path = "../input/twix-minis/input/"
import keras

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
train = []

classes_N = 0

for dirname, _, filenames in os.walk('../input/twix-minis/input/train'):

    for filename in filenames:

        list_of_objects = {}

        list_of_objects['path'] = dirname + '/' + filename

        list_of_objects['name'] = filename

        list_of_objects['class'] =classes_N

        train.append(list_of_objects)

    classes_N += 1

train = pd.DataFrame(train) 
train
test = []

classes_N = 0

for dirname, _, filenames in os.walk('../input/twix-minis/input/validation'):

    for filename in filenames:

        list_of_objects = {}

        list_of_objects['path'] = dirname+ '/' + filename

        list_of_objects['name'] = filename

        list_of_objects['class'] =classes_N

        test.append(list_of_objects)

    classes_N += 1

test = pd.DataFrame(test) 
test
num_classes =len(set(train['class']))

num_classes
conv_base = ResNet152(

    include_top=False,

    weights='imagenet')



for layer in conv_base.layers:

    layer.trainable = False
from keras.layers import LeakyReLU
def lrelu(x):

    return tf.keras.activations.relu(x, alpha=0.01)
x = conv_base.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation=lrelu)(x) 

predictions = layers.Dense(num_classes, activation='softmax')(x)

model = Model(conv_base.input, predictions)
optimizer = keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])
input_path = '../input/twix-minis/input/'
train_datagen = ImageDataGenerator(

    shear_range=10,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True,

    rotation_range=30,

    zca_epsilon=0.3,

    width_shift_range=0.2,

    height_shift_range=0.2,

    preprocessing_function=preprocess_input)



train_generator = train_datagen.flow_from_directory(

    input_path + 'train',

    batch_size=32,

    class_mode='categorical',

    target_size=(224,224))



validation_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input)



validation_generator = validation_datagen.flow_from_directory(

    input_path + 'validation',

    shuffle=False,

    class_mode='categorical',

    target_size=(224,224))
from keras.callbacks import EarlyStopping, ModelCheckpoint



early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5, mode='max', verbose=1)

checkpoint = ModelCheckpoint('model_best_weights.h5', monitor='val_accuracy',save_best_only=True)
history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=320 // 32,  # added in Kaggle

                              epochs=2,

                              validation_data=validation_generator,

                              validation_steps=1,  # added in Kaggle

                              callbacks = [early_stop,checkpoint]

                             )
history.history
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
models = keras.models.load_model('/kaggle/working/model_best_weights.h5', custom_objects = {"lrelu": lrelu})
Y_pred = model.predict_generator(validation_generator)

y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')

# print(confusion_matrix(validation_generator.classes, y_pred))
target_names = ['Хороший класс', 'Волосяные трещины', 'Сломанный и раздавленный', 'Тонкий слой шоколада', 'Хвостики', 'Затертая поверхность', 'Значительное искажение', 'Отверстия в шоколаде', 'Открытый центр', 'Плохая декорация', 'Плохое донышко',

               'Повреждение упаковочной машины', ]
validation_generator.class_indices
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
validation_img_paths = ["validation/alien/11.jpg",

                        "validation/alien/22.jpg",

                        "validation/predator/33.jpg"]

img_list = [Image.open(input_path + img_path) for img_path in validation_img_paths]
valid_binary = np.where(validation_generator.classes!=0, 1, validation_generator.classes)
valid_binary
y_pred_binary = np.where(y_pred!=0, 1, y_pred)
target_names_binary = ['Хороший класс', 'Дефект']
print(classification_report(valid_binary, y_pred_binary, target_names=target_names_binary))
keras.__version__
print(classification_report(valid_binary, y_pred_binary, target_names=target_names))
from PIL import Image
validation_img_paths = ["train/Class2//IMG_6230.JPG",

                        "train/Class12//IMG_6302.JPG"]

img_list = [Image.open(input_path + img_path) for img_path in validation_img_paths]
validation_batch = np.stack([preprocess_input(np.array(img.resize((224,224))))

                             for img in img_list])
pred_probs = model.predict(validation_batch)

print(pred_probs[0][5])

print(pred_probs[1][4])
pred_probs
model.save('keras_restnet152.h5')