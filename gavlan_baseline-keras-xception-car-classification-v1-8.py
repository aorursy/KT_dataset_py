!pip show keras
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn

import pickle

import csv

import os



from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from keras.callbacks import Callback

from keras.regularizers import l2

from keras import optimizers

from keras.models import Model

from keras.utils import np_utils

from keras.applications.xception import Xception

from keras.layers import *



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
# В сетап выношу основные настройки, так удобней их перебирать в дальнейшем



EPOCHS               = 10

BATCH_SIZE           = 32

LR                   = 1e-4



CLASS_NUM            = 10

IMG_SIZE             = 299 # Стандартый размер для некоторых сетей

IMG_CHANNELS         = 3

input_shape          = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)



DATA_PATH = '../input/'

PATH = "../working/car/"
os.makedirs(PATH,exist_ok=False)



RANDOM_SEED = 42



np.random.seed(RANDOM_SEED)



from tensorflow import set_random_seed

set_random_seed(RANDOM_SEED)
train_df = pd.read_csv(DATA_PATH+"train.csv")

sample_submission = pd.read_csv(DATA_PATH+"sample-submission.csv")

train_df.head()
train_df.info()
train_df.Category.value_counts()
print('Пример картинок (random sample)')

plt.figure(figsize=(12,8))



random_image = train_df.sample(n=9)

random_image_paths = random_image['Id'].values

random_image_cat = random_image['Category'].values



for index, path in enumerate(random_image_paths):

    im = PIL.Image.open(DATA_PATH+f'train/train/{random_image_cat[index]}/{path}')

    plt.subplot(3,3, index+1)

    plt.imshow(im)

    plt.title('Class: '+str(random_image_cat[index]))

    plt.axis('off')

plt.show()
image = PIL.Image.open(DATA_PATH+'/train/train/0/100380.jpg')

imgplot = plt.imshow(image)

plt.show()

image.size
# Аугментация данных очень важна когда у нас не большой датасет (как в нашем случае)

# Поиграйся тут параметрами чтоб понять что к чему. 

# Официальная дока https://keras.io/preprocessing/image/



train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    rotation_range = 5,

    width_shift_range=0.1,

    height_shift_range=0.1,

    validation_split=0.1, # set validation split

    horizontal_flip=False)



test_datagen = ImageDataGenerator(rescale=1. / 255)



# Задание для Про - попробуй подключить сторонние более продвинутые библиотеки аугминтации изображений
# "Заворачиваем" наши данные в generator



train_generator = train_datagen.flow_from_directory(

    DATA_PATH+'train/train/',

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    shuffle=True, seed=RANDOM_SEED,

    subset='training') # set as training data



test_generator = train_datagen.flow_from_directory(

    DATA_PATH+'train/train/',

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    shuffle=True, seed=RANDOM_SEED,

    subset='validation') # set as validation data



test_sub_generator = test_datagen.flow_from_dataframe(

    dataframe=sample_submission,

    directory=DATA_PATH+'test/test_upload',

    x_col="Id",

    y_col=None,

    shuffle=False,

    class_mode=None,

    seed=RANDOM_SEED,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,)



# кстати, ты заметил, что для сабмишена мы используем другой источник для генератора flow_from_dataframe? 

# Как ты думаешь, почему?
# Кстати Попробуй еще другие архитектуры сетей...

base_model = Xception(weights='imagenet', include_top=False, input_shape = input_shape)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape = input_shape)
base_model.summary()
# Устанавливаем новую "голову"

# Тут тоже можно поиграться, попробуй добавить Batch Normalization например.



#x = base_model.output

#x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

##x = Dense(256, activation='relu')(x)

#x = Dropout(0.25)(x)

# and a logistic layer -- let's say we have 10 classes

#predictions = Dense(CLASS_NUM, activation='softmax')(x)



# this is the model we will train

#model = Model(inputs=base_model.input, outputs=predictions)

#model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR), metrics=["accuracy"])
# Test 1



#x = base_model.output

#x = GlobalAveragePooling2D()(x)

#x = BatchNormalization()(x)  # new

# let's add a fully-connected layer

#x = Dense(256, activation='relu')(x)

#x = Dropout(0.25)(x)

#x = BatchNormalization()(x)  # new

# and a logistic layer -- let's say we have 10 classes

#predictions = Dense(CLASS_NUM, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR), metrics=["accuracy"])
# Test 2



#x = base_model.output

#x = GlobalAveragePooling2D()(x)

#x = BatchNormalization()(x)



#x = Dense(256, activation='relu')(x)

#x = Dropout(0.2)(x)

#x = BatchNormalization()(x) 



#x = Dense(64, activation='relu')(x) 

#x = Dropout(0.3)(x)  

#x = BatchNormalization()(x) 



#x = Dense(16, activation='relu')(x) 

#x = Dropout(0.4)(x)         

#x = BatchNormalization()(x)  

# and a logistic layer -- let's say we have 10 classes

#predictions = Dense(CLASS_NUM, activation='softmax')(x)



# this is the model we will train

#model = Model(inputs=base_model.input, outputs=predictions)

#model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR), metrics=["accuracy"])
# Test 3



x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)



x = Dense(256, activation='elu')(x)

x = Dropout(0.2)(x)

x = BatchNormalization()(x) 



x = Dense(64, activation='elu')(x) 

x = Dropout(0.3)(x)  

x = BatchNormalization()(x) 



x = Dense(16, activation='elu')(x) 

x = Dropout(0.4)(x)         

x = BatchNormalization()(x)  

# and a logistic layer -- let's say we have 10 classes

predictions = Dense(CLASS_NUM, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR), metrics=["accuracy"])
model.summary()
# Рекомендую добавть еще функции из https://keras.io/callbacks/

checkpoint = ModelCheckpoint('best_model.hdf5' , monitor = ['val_acc'] , verbose = 1  , mode = 'max')

callbacks_list = [checkpoint]



# Для про - попробуй добавить разные техники управления Learning Rate

# Например:

# https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6

# http://teleported.in/posts/cyclic-learning-rate/
# Обучаем

history = model.fit_generator(

        train_generator,

        steps_per_epoch = len(train_generator),

        validation_data = test_generator, 

        validation_steps = len(test_generator),

        epochs = EPOCHS,

        callbacks = callbacks_list

)



# попробуй применить transfer learning с fine-tuning

# Сначала замораживаем все слои кроме новой "головы"

# Потом, когда мы научили последние слои (голову) под новую задачу, можно разморозить все слои и пройтись маленьким лернинг рейтом
model.save('../working/model_last.hdf5')

model.load_weights('best_model.hdf5')
scores = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=1)

print("Accuracy: %.2f%%" % (scores[1]*100))
acc = history.history['acc']

val_acc = history.history['val_acc']

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



# Для Про - попробуй TTA
submission.head()