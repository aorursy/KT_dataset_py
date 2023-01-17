!nvidia-smi
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import pickle

import zipfile

import csv

import sys

import os





import tensorflow as tf

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from tensorflow.keras.callbacks import Callback

from tensorflow.keras.regularizers import l2

from tensorflow.keras import optimizers

from tensorflow.keras.models import Model

from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.layers import *



from sklearn.model_selection import train_test_split, StratifiedKFold



import PIL

from PIL import ImageOps, ImageFilter

#увеличим дефолтный размер графиков

from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

#графики в svg выглядят более четкими

%config InlineBackend.figure_format = 'svg' 

%matplotlib inline



print(os.listdir("../input"))

print('Python       :', sys.version.split('\n')[0])

print('Numpy        :', np.__version__)

print('Tensorflow   :', tf.__version__)

print('Keras        :', tf.keras.__version__)
!pip freeze > requirements.txt
# В setup выносим основные настройки: так удобнее их перебирать в дальнейшем.



EPOCHS               = 5  # эпох на обучение

BATCH_SIZE           = 64 # уменьшаем batch если сеть большая, иначе не влезет в память на GPU

LR                   = 1e-4

VAL_SPLIT            = 0.15 # сколько данных выделяем на тест = 15%



CLASS_NUM            = 10  # количество классов в нашей задаче

IMG_SIZE             = 224 # какого размера подаем изображения в сеть

IMG_CHANNELS         = 3   # у RGB 3 канала

input_shape          = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)



DATA_PATH = '../input/'

PATH = "../working/car/" # рабочая директория
# Устаналиваем конкретное значение random seed для воспроизводимости

os.makedirs(PATH,exist_ok=False)



RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)  

PYTHONHASHSEED = 0
train_df = pd.read_csv(DATA_PATH+"train.csv")

sample_submission = pd.read_csv(DATA_PATH+"sample-submission.csv")

train_df.head()
train_df.info()
train_df.Category.value_counts()

# распределение классов достаточно равномерное - это хорошо
print('Распаковываем картинки')

# Will unzip the files so that you can see them..

for data_zip in ['train.zip', 'test.zip']:

    with zipfile.ZipFile("../input/"+data_zip,"r") as z:

        z.extractall(PATH)

        

print(os.listdir(PATH))
print('Пример картинок (random sample)')

plt.figure(figsize=(12,8))



random_image = train_df.sample(n=9)

random_image_paths = random_image['Id'].values

random_image_cat = random_image['Category'].values



for index, path in enumerate(random_image_paths):

    im = PIL.Image.open(PATH+f'train/{random_image_cat[index]}/{path}')

    plt.subplot(3,3, index+1)

    plt.imshow(im)

    plt.title('Class: '+str(random_image_cat[index]))

    plt.axis('off')

plt.show()
image = PIL.Image.open(PATH+'/train/0/100380.jpg')

imgplot = plt.imshow(image)

plt.show()

image.size
# Вы помните, что аугментация данных важна, когда мы работаем с небольшим датасетом. Это как раз наш случай.

# Чтобы лучше понять работу параметров, попробуйте их изменить. К какому результату это приведет?

# Официальная документация: https://keras.io/preprocessing/image/



train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    rotation_range = 5,

    width_shift_range=0.1,

    height_shift_range=0.1,

    validation_split=VAL_SPLIT, # set validation split

    horizontal_flip=False)



test_datagen = ImageDataGenerator(rescale=1. / 255)



#Рекомендация Подключите более продвинутые библиотеки аугментации изображений (например: albumentations или imgaug, для них есть специальные "обертки" под Keras, например: https://github.com/mjkvaak/ImageDataAugmentor)
# Завернем наши данные в генератор:



train_generator = train_datagen.flow_from_directory(

    PATH+'train/',      # директория где расположены папки с картинками 

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    shuffle=True, seed=RANDOM_SEED,

    subset='training') # set as training data



test_generator = train_datagen.flow_from_directory(

    PATH+'train/',

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    shuffle=True, seed=RANDOM_SEED,

    subset='validation') # set as validation data



test_sub_generator = test_datagen.flow_from_dataframe( 

    dataframe=sample_submission,

    directory=PATH+'test_upload/',

    x_col="Id",

    y_col=None,

    shuffle=False,

    class_mode=None,

    seed=RANDOM_SEED,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,)



# Обратите внимание, что для сабмита мы используем другой источник test_datagen.flow_from_dataframe. Как вы думаете, почему?
base_model = Xception(weights='imagenet', include_top=False, input_shape = input_shape)
base_model.summary()

# Рекомендация: Попробуйте и другие архитектуры сетей
# Устанавливаем новую "голову" (head)



x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(256, activation='relu')(x)

x = Dropout(0.25)(x)

# and a logistic layer -- let's say we have 10 classes

predictions = Dense(CLASS_NUM, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR), metrics=["accuracy"])
model.summary()

# Рекомендация: Попробуйте добавить Batch Normalization
checkpoint = ModelCheckpoint('best_model.hdf5' , monitor = ['val_accuracy'] , verbose = 1  , mode = 'max')

callbacks_list = [checkpoint]



# Рекомендация 1. Добавьте другие функции из https://keras.io/callbacks/

# Рекомендация 2. Используйте разные техники управления Learning Rate

# https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6 (eng)

# http://teleported.in/posts/cyclic-learning-rate/ (eng)
history = model.fit_generator(

        train_generator,

        steps_per_epoch = len(train_generator),

        validation_data = test_generator, 

        validation_steps = len(test_generator),

        epochs = EPOCHS,

        callbacks = callbacks_list

)



# Рекомендация: попробуйте применить transfer learning с fine-tuning
# сохраним итоговую сеть и подгрузим лучшую итерацию в обучении (best_model)

model.save('../working/model_last.hdf5')

model.load_weights('best_model.hdf5')
scores = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=1)

print("Accuracy: %.2f%%" % (scores[1]*100))
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

 

epochs = range(len(acc))

 

plt.plot(epochs, acc, 'b', label='Training acc')

plt.plot(epochs, val_acc, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

 

plt.figure()

 

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

 

plt.show()
test_sub_generator.samples
test_sub_generator.reset()

predictions = model.predict_generator(test_sub_generator, steps=len(test_sub_generator), verbose=1) 

predictions = np.argmax(predictions, axis=-1) #multiple categories

label_map = (train_generator.class_indices)

label_map = dict((v,k) for k,v in label_map.items()) #flip k,v

predictions = [label_map[k] for k in predictions]
filenames_with_dir=test_sub_generator.filenames

submission = pd.DataFrame({'Id':filenames_with_dir, 'Category':predictions}, columns=['Id', 'Category'])

submission['Id'] = submission['Id'].replace('test_upload/','')

submission.to_csv('submission.csv', index=False)

print('Save submit')



# Рекомендация: попробуйте добавить Test Time Augmentation (TTA)

# https://towardsdatascience.com/test-time-augmentation-tta-and-how-to-perform-it-with-keras-4ac19b67fb4d
submission.head()
# Clean PATH

import shutil

shutil.rmtree(PATH)