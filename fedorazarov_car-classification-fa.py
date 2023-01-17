# Обновление tensorflow

!pip install tensorflow --upgrade

# Загрузка модели efficientnet

!pip install -q efficientnet

# Загружаем обвязку под keras для использования продвинутых библиотек аугментации, например, albuminations

!pip install git+https://github.com/mjkvaak/ImageDataAugmentor
!nvidia-smi
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import pickle

import scipy.io

import tarfile

import csv

import sys

import os



import tensorflow as tf

import tensorflow.keras as keras

import tensorflow.keras.models as M

import tensorflow.keras.layers as L

import tensorflow.keras.backend as K

import tensorflow.keras.callbacks as C

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping

from tensorflow.keras.callbacks import Callback

from tensorflow.keras import optimizers

import efficientnet.tfkeras as efn

import zipfile



from ImageDataAugmentor.image_data_augmentor import *

import albumentations



from sklearn.model_selection import train_test_split



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

print('Keras        :', keras.__version__)
!pip freeze > requirements.txt
# Batch size

BATCH_SIZE           = 8 # уменьшаем batch если сеть большая, иначе не влезет в память на GPU

BATCH_SIZE_STEP4     = 4



# Epochs

EPOCHS_STEP1         = 10  # эпох на обучение

EPOCHS_STEP2         = 10  # эпох на обучение  

EPOCHS_STEP3         = 8  # эпох на обучение  

EPOCHS_STEP4         = 8  # эпох на обучение  



# Learning Rates

LR_STEP1             = 1e-3

LR_STEP2             = 1e-4

LR_STEP3             = 1e-5

LR_STEP4             = 1e-5



# Learning Rate One Cycle Policy

MAX_MOMENTUM = 0.98

BASE_MOMENTUM = 0.85

CYCLICAL_MOMENTUM = True

AUGMENT = True

CYCLES = 2.35



# Test-validation split

VAL_SPLIT            = 0.2 # сколько данных выделяем на тест = 20%



CLASS_NUM            = 10  # количество классов в нашей задаче

IMG_SIZE             = 250 # какого размера подаем изображения в сеть

IMG_SIZE_STEP4       = 512



IMG_CHANNELS         = 3   # у RGB 3 канала

input_shape          = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)



DATA_PATH = '../input/'

PATH = "../working/car/" # рабочая директория



# Создание рабочей директории

os.makedirs(PATH,exist_ok=False)



# Устанавливаем конкретное значение random seed для воспроизводимости



RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)  

PYTHONHASHSEED = 0
df = pd.read_csv(DATA_PATH+"train.csv")

sample_submission = pd.read_csv(DATA_PATH+"sample-submission.csv")

df.head()
df.info()
df.Category.value_counts()

# распределение классов достаточно равномерное - это хорошо
print('Распаковываем картинки')

# Will unzip the files so that you can see them..

for data_zip in ['train.zip', 'test.zip']:

    with zipfile.ZipFile("../input/"+data_zip,"r") as z:

        z.extractall(PATH)

        

print(os.listdir(PATH))
print('Пример картинок (random sample)')

plt.figure(figsize=(12,8))



random_image = df.sample(n=9)

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
AUGMENTATIONS = albumentations.Compose([

    albumentations.HorizontalFlip(p=0.5),

    albumentations.Rotate(limit=30, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),

    albumentations.OneOf([

        albumentations.CenterCrop(height=250, width=200),

        albumentations.CenterCrop(height=200, width=250),

    ],p=0.5),

    albumentations.OneOf([

        albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),

        albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)

    ],p=0.5),

    albumentations.GaussianBlur(p=0.05),

    albumentations.HueSaturationValue(p=0.5),

    albumentations.RGBShift(p=0.5),

    albumentations.FancyPCA(alpha=0.1, always_apply=False, p=0.5),

    albumentations.Resize(IMG_SIZE, IMG_SIZE)

])



train_datagen = ImageDataAugmentor(

        rescale=1./255,

        augment = AUGMENTATIONS,

        validation_split=VAL_SPLIT,

        )

        

test_datagen = ImageDataAugmentor(rescale=1./255)
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
from skimage import io



def imshow(image_RGB):

    io.imshow(image_RGB)

    io.show()



x,y = train_generator.next()

print('Пример картинок из train_generator')

plt.figure(figsize=(12,8))



for i in range(0,6):

    image = x[i]

    plt.subplot(3,3, i+1)

    plt.imshow(image)

    #plt.title('Class: '+str(y[i]))

    #plt.axis('off')

plt.show()
x,y = test_generator.next()

print('Пример картинок из test_generator')

plt.figure(figsize=(12,8))



for i in range(0,6):

    image = x[i]

    plt.subplot(3,3, i+1)

    plt.imshow(image)

    #plt.title('Class: '+str(y[i]))

    #plt.axis('off')

plt.show()
base_model = efn.EfficientNetB6(

    weights='imagenet', # Подгружаем веса imagenet

    include_top=False,  # Выходной слой (голову) будем менять т.к. у нас други классы

    input_shape=input_shape)
# Посмотрим на загруженную модель EfficientNetB6

base_model.summary()
# Для начала заморозим веса EfficientNetB6 и обучим только "голову". 

# Делаем это для того, чтобы хорошо обученные признаки на Imagenet не затирались в самом начале нашего обучения

base_model.trainable = False
model=M.Sequential()

model.add(base_model)

model.add(L.GlobalAveragePooling2D(),) # объединяем все признаки в единый вектор 



# Экспериментируем с архитектурой - добавляем ещё один полносвязный слой, dropout и batch-нормализацию

model.add(L.Dense(256, activation='relu'))

model.add(L.BatchNormalization())

model.add(L.Dropout(0.25))



model.add(L.Dense(CLASS_NUM, activation='softmax'))
# Смотрим на получившуюся модель

model.summary()
# Количество слоев

print(len(model.layers))
# Количество параметров обучения

len(model.trainable_variables)
# Статус слоев - будем обучать или нет

for layer in model.layers:

    print(layer, layer.trainable)
# Implement One Cycle Policy Algorithm in the Keras Callback Class



from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

from keras.losses import binary_crossentropy

from keras.metrics import binary_accuracy

from keras import backend as K

from keras.callbacks import *



class CyclicLR(keras.callbacks.Callback):

    

    def __init__(self,base_lr, max_lr, step_size, base_m, max_m, cyclical_momentum):

 

        self.base_lr = base_lr

        self.max_lr = max_lr

        self.base_m = base_m

        self.max_m = max_m

        self.cyclical_momentum = cyclical_momentum

        self.step_size = step_size

        

        self.clr_iterations = 0.

        self.cm_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}

        

    def clr(self):

        

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        

        if cycle == 2:

            x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)          

            return self.base_lr-(self.base_lr-self.base_lr/100)*np.maximum(0,(1-x))

        

        else:

            x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0,(1-x))

    

    def cm(self):

        

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        

        if cycle == 2:

            

            x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1) 

            return self.max_m

        

        else:

            x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

            return self.max_m - (self.max_m-self.base_m)*np.maximum(0,(1-x))

        

        

    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())

            

        if self.cyclical_momentum == True:

            if self.clr_iterations == 0:

                K.set_value(self.model.optimizer.momentum, self.cm())

            else:

                K.set_value(self.model.optimizer.momentum, self.cm())

            

            

    def on_batch_begin(self, batch, logs=None):

        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)

        

        if self.cyclical_momentum == True:

            self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.momentum))



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())

        

        if self.cyclical_momentum == True:

            K.set_value(self.model.optimizer.momentum, self.cm())
# Настройки

batch_size = BATCH_SIZE

epochs = EPOCHS_STEP1

base_lr = LR_STEP1

max_lr = base_lr*10

max_m = MAX_MOMENTUM

base_m = BASE_MOMENTUM



cyclical_momentum = CYCLICAL_MOMENTUM

augment = AUGMENT

cycles = CYCLES



# Расчет количества итерация и шага изменения learning rate

iterations = round(train_generator.samples//train_generator.batch_size*epochs)

iterations = list(range(0,iterations+1))

step_size = len(iterations)/(cycles)



model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=base_lr, momentum=BASE_MOMENTUM), metrics=["accuracy"])



clr =  CyclicLR(

    base_lr=base_lr,

    max_lr=max_lr,

    step_size=step_size,

    max_m=max_m,

    base_m=base_m,

    cyclical_momentum=cyclical_momentum

)

    

# Добавим ModelCheckpoint чтоб сохранять прогресс обучения модели и можно было потом подгрузить и дообучить модель.    

checkpoint = ModelCheckpoint('best_model.hdf5' , monitor = ['accuracy'] , verbose = 1  , mode = 'max')

earlystop = EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

callbacks_list = [checkpoint, earlystop, clr]



# Обучаем

history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_generator.samples//train_generator.batch_size,

    validation_data = test_generator, 

    validation_steps = test_generator.samples//test_generator.batch_size,

    epochs = epochs,

    callbacks = callbacks_list

    )
# Plot Learning Rate

import matplotlib.pyplot as plt

plt.plot(clr.history['iterations'], clr.history['lr'])

plt.xlabel('Training Iterations')

plt.ylabel('Learning Rate')

plt.title("One Cycle Policy")

plt.show()
# Plot momentum

import matplotlib.pyplot as plt

plt.plot(clr.history['iterations'], clr.history['momentum'])

plt.xlabel('Training Iterations')

plt.ylabel('Momentum')

plt.title("One Cycle Policy")

plt.show()
scores = model.evaluate_generator(test_generator, verbose=1)

print("Accuracy: %.2f%%" % (scores[1]*100))
# Сохраним итоговую сеть и подгрузим лучшую итерацию в обучении (best_model)

model.save('../working/model_step1.hdf5')

model.load_weights('best_model.hdf5')
# Посмотрим на количество слоев в базовой модели

print("Number of layers in the base model: ", len(base_model.layers))
# Разморозим базовую модель

base_model.trainable = True



# Установим количество слоев, которые будем переобучать

fine_tune_at = len(base_model.layers)//2



# Заморозим первую половину слоев

for layer in base_model.layers[:fine_tune_at]:

    layer.trainable = False
# Количество параметров

len(base_model.trainable_variables)
# Статус слоев - будем обучать или нет

for layer in model.layers:

    print(layer, layer.trainable)
# Настройки

#batch_size = BATCH_SIZE

epochs = EPOCHS_STEP2

base_lr = LR_STEP2

max_lr = base_lr*10

#max_m = MAX_MOMENTUM

#base_m = BASE_MOMENTUM



#cyclical_momentum = CYCLICAL_MOMENTUM

#augment = AUGMENT

#cycles = CYCLES



# Расчет количества итерация и шага изменения learning rate

iterations = round(train_generator.samples//train_generator.batch_size*epochs)

iterations = list(range(0,iterations+1))

step_size = len(iterations)/(cycles)



model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=base_lr, momentum=BASE_MOMENTUM), metrics=["accuracy"])



clr =  CyclicLR(

    base_lr=base_lr,

    max_lr=max_lr,

    step_size=step_size,

    max_m=max_m,

    base_m=base_m,

    cyclical_momentum=cyclical_momentum

)

    

# Добавим ModelCheckpoint чтоб сохранять прогресс обучения модели и можно было потом подгрузить и дообучить модель.    

#checkpoint = ModelCheckpoint('best_model.hdf5' , monitor = ['val_accuracy'] , verbose = 1  , mode = 'max')

#earlystop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

callbacks_list = [checkpoint, earlystop, clr]



# Обучаем

history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_generator.samples//train_generator.batch_size,

    validation_data = test_generator, 

    validation_steps = test_generator.samples//test_generator.batch_size,

    epochs = epochs,

    callbacks = callbacks_list

)
# Сохраним модель

model.save('../working/model_step2.hdf5')

model.load_weights('best_model.hdf5')
scores = model.evaluate_generator(test_generator, verbose=1)

print("Accuracy: %.2f%%" % (scores[1]*100))
base_model.trainable = True
# Настройки

#batch_size = BATCH_SIZE

epochs = EPOCHS_STEP3

base_lr = LR_STEP3

max_lr = base_lr*10

#max_m = MAX_MOMENTUM

#base_m = BASE_MOMENTUM



#cyclical_momentum = CYCLICAL_MOMENTUM

#augment = AUGMENT

#cycles = CYCLES



# Расчет количества итерация и шага изменения learning rate

iterations = round(train_generator.samples//train_generator.batch_size*epochs)

iterations = list(range(0,iterations+1))

step_size = len(iterations)/(cycles)



model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=base_lr, momentum=BASE_MOMENTUM), metrics=["accuracy"])



clr =  CyclicLR(

    base_lr=base_lr,

    max_lr=max_lr,

    step_size=step_size,

    max_m=max_m,

    base_m=base_m,

    cyclical_momentum=cyclical_momentum

)

    

# Добавим ModelCheckpoint чтоб сохранять прогресс обучения модели и можно было потом подгрузить и дообучить модель.    

#checkpoint = ModelCheckpoint('best_model.hdf5' , monitor = ['val_accuracy'] , verbose = 1  , mode = 'max')

#earlystop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

callbacks_list = [checkpoint, earlystop, clr]



# Обучаем

history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_generator.samples//train_generator.batch_size,

    validation_data = test_generator, 

    validation_steps = test_generator.samples//test_generator.batch_size,

    epochs = epochs,

    callbacks = callbacks_list

)
model.save('../working/model_step3.hdf5')

model.load_weights('best_model.hdf5')
scores = model.evaluate_generator(test_generator, verbose=1)

print("Accuracy: %.2f%%" % (scores[1]*100))
IMG_SIZE             = IMG_SIZE_STEP4

BATCH_SIZE           = BATCH_SIZE_STEP4

input_shape          = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
AUGMENTATIONS = albumentations.Compose([

    albumentations.HorizontalFlip(p=0.5),

    albumentations.Rotate(limit=30, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),

    # albumentations.OneOf([

    #     albumentations.CenterCrop(height=250, width=200),

    #     albumentations.CenterCrop(height=200, width=250),

    # ],p=0.5),

    # albumentations.OneOf([

    #     albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),

    #     albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)

    # ],p=0.5),

    # albumentations.GaussianBlur(p=0.05),

    # albumentations.HueSaturationValue(p=0.5),

    # albumentations.RGBShift(p=0.5),

    # albumentations.FancyPCA(alpha=0.1, always_apply=False, p=0.5),

    # albumentations.Resize(IMG_SIZE, IMG_SIZE)

])



train_datagen = ImageDataAugmentor(

        rescale=1./255,

        augment = AUGMENTATIONS,

        validation_split=VAL_SPLIT,

        )

        

test_datagen = ImageDataAugmentor(rescale=1./255)
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
base_model = efn.EfficientNetB6(weights='imagenet', include_top=False, input_shape=input_shape)
# Настройки

batch_size = BATCH_SIZE_STEP4

epochs = EPOCHS_STEP4

base_lr = LR_STEP4

max_lr = base_lr*10

#max_m = MAX_MOMENTUM

#base_m = BASE_MOMENTUM



#cyclical_momentum = CYCLICAL_MOMENTUM

#augment = AUGMENT

#cycles = CYCLES



# Расчет количества итерация и шага изменения learning rate

iterations = round(train_generator.samples//train_generator.batch_size*epochs)

iterations = list(range(0,iterations+1))

step_size = len(iterations)/(cycles)



model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=base_lr, momentum=BASE_MOMENTUM), metrics=["accuracy"])



model.load_weights('best_model.hdf5') # Подгружаем ранее обученные веса



clr =  CyclicLR(

    base_lr=base_lr,

    max_lr=max_lr,

    step_size=step_size,

    max_m=max_m,

    base_m=base_m,

    cyclical_momentum=cyclical_momentum

)

    

# Добавим ModelCheckpoint чтоб сохранять прогресс обучения модели и можно было потом подгрузить и дообучить модель.    

#checkpoint = ModelCheckpoint('best_model.hdf5' , monitor = ['val_accuracy'] , verbose = 1  , mode = 'max')

#earlystop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

callbacks_list = [checkpoint, earlystop, clr]



# Обучаем

history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_generator.samples//train_generator.batch_size,

    validation_data = test_generator, 

    validation_steps = test_generator.samples//test_generator.batch_size,

    epochs = epochs,

    callbacks = callbacks_list

)
model.save('../working/model_step4.hdf5')

model.load_weights('best_model.hdf5')
scores = model.evaluate_generator(test_generator, verbose=1)

print("Accuracy: %.2f%%" % (scores[1]*100))
from sklearn.metrics import accuracy_score
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
submission.head()
model.load_weights('best_model.hdf5')
AUGMENTATIONS = albumentations.Compose([

    albumentations.HorizontalFlip(p=0.5),

    albumentations.Rotate(limit=30, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),

    albumentations.OneOf([

        albumentations.CenterCrop(height=250, width=200),

        albumentations.CenterCrop(height=200, width=250),

    ],p=0.5),

    albumentations.OneOf([

        albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),

        albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)

    ],p=0.5),

    albumentations.GaussianBlur(p=0.05),

    albumentations.HueSaturationValue(p=0.5),

    albumentations.RGBShift(p=0.5),

    albumentations.FancyPCA(alpha=0.1, always_apply=False, p=0.5),

    albumentations.Resize(IMG_SIZE, IMG_SIZE)

])

      

test_datagen = ImageDataAugmentor( 

    rescale=1./255,

    augment = AUGMENTATIONS,

    validation_split=VAL_SPLIT,

)
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
tta_steps = 10 # берем среднее из 10 предсказаний

predictions = []



for i in range(tta_steps):

    preds = model.predict_generator(test_sub_generator, steps=len(test_sub_generator), verbose=1) 

    predictions.append(preds)



pred = np.mean(predictions, axis=0)
predictions = np.argmax(pred, axis=-1) #multiple categories

label_map = (train_generator.class_indices)

label_map = dict((v,k) for k,v in label_map.items()) #flip k,v

predictions = [label_map[k] for k in predictions]
filenames_with_dir=test_sub_generator.filenames

submission = pd.DataFrame({'Id':filenames_with_dir, 'Category':predictions}, columns=['Id', 'Category'])

submission['Id'] = submission['Id'].replace('test_upload/','')

submission.to_csv('submission_TTA.csv', index=False)

print('Save submit')
# Clean PATH

import shutil

shutil.rmtree(PATH)