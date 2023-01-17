from tqdm import tqdm

import cv2

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

from datetime import datetime as dt
# считываем файлы

submission=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')

train=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
train.head()
# смотрим баланс классов

train.loc[:,'healthy':'scab'].sum(axis=0)/(train.shape[0]/100)
test.head()
submission.head()
# загружаем изображения и меняем их размер

# если поменяете размер изображения тут, то не забудьте сделать это на входе нейронной сети

train_img=[]

train_label=[]

path='/kaggle/input/plant-pathology-2020-fgvc7/images'

for im in tqdm(train['image_id']):

    im=im+".jpg"

    final_path=os.path.join(path,im)

    img=cv2.imread(final_path)

    img=cv2.resize(img,(224,224))

    img=img.astype('float32')

    train_img.append(img)
test_img=[]

path='/kaggle/input/plant-pathology-2020-fgvc7/images'

for im in tqdm(test['image_id']):

    im=im+".jpg"

    final_path=os.path.join(path,im)

    img=cv2.imread(final_path)

    img=cv2.resize(img,(224,224))

    img=img.astype(('float32'))

    test_img.append(img)
# выделяем тренировочные метки

train_label=train.loc[:,'healthy':'scab']
# преобразовываем все массивы в numpy

train_img=np.array(train_img)/255.0

test_img=np.array(test_img)/255.0

train_label=np.array(train_label)
print(train_img.shape)

print(test_img.shape)

print(train_label.shape)
# создаем дата генератор для увеличения обучающей выборки

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.3, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True)  # randomly flip images





datagen.fit(train_img)
!pip install efficientnet
## ______________ БЛОК С ИМПОРТАМИ АРХИТЕКТУР ____________________

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201 

from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2

from tensorflow.keras.applications.nasnet import NASNetLarge

from efficientnet.tfkeras import EfficientNetB7, EfficientNetL2

## ______________ КОНЕЦ БЛОКА С ИМПОРТАМИ АРХИТЕКТУР ____________________



# импорт других полезных инструментов: слоев, оптимизаторов, функций обратной связи

from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
# тут можете вместо ResNet50 подставлять по очереди вышеперечисленные модели VGG19, InceptionV3, InceptionResNetV2, DenseNet201 и др

base_model=EfficientNetB7(include_top=False, weights='imagenet',input_shape=(224,224,3), pooling='avg')



model=Sequential()

model.add(base_model)



#  в этом блоке можете поварьировать количество блоков Dense, их параметры (128, 256, 1024, др) или функции активации,

# параметры блока Dropout, убрать/добваить BatchNormalization

model.add(Dense(1024,activation='relu'))

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.8))

model.add(Dense(4,activation='softmax'))



# если хотите полностью заморозить сеть, то приравняйте к True, иначе напишите какое количество слоев вы хотите разморозить

froze = True # 13

if froze is True:

    base_model.trainable=False

else:

    for layer in base_model.layers[:-int(froze)]:

        layer.trainable = False



# тут можно регулировать через сколько эпох (patience) шаг обучения (lr) будет меняться и на сколько (factor).

# при достижении min_lr обучение прервется даже если не закончились эпохи, которые вы задали для обучения

reduce_learning_rate = ReduceLROnPlateau(monitor='categorical_accuracy',

                                         factor=0.5,

                                         patience=5,

                                         cooldown=2,

                                         min_lr=0.0000001,

                                         verbose=1)

# тут можно мониторить оптимизируемую метрику и если она не улучшалась "patience" эпох, то остановить обучение

early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=10)



# тут мы мониторим оптимизируемую метрику и делаем сохранение модели только когда она становится лучше, худшие модели отбрасываем

check_point = ModelCheckpoint(filepath='resnet_50.h5', monitor='categorical_accuracy', save_best_only=True)



# тут просто добавляем список наших обратных связей, если посчитаете какие-либо лишними, то просто не добавляйте в список

callbacks = [reduce_learning_rate, early_stopping, check_point]

    

# тут можете попробовать менять "optimizer" и "metrics". Но если поменяете "metrics", то не забудьте поменять ее и выше во всех "monitor"

model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])

# описание модели

model.summary()
# запускаем обучение

start = dt.now()

history = model.fit_generator(datagen.flow(train_img, train_label, batch_size=32),

                    epochs=200,callbacks=callbacks)

print("Время работы модели: {}. Количество эпох: {}.".format(dt.now() - start, len(history.epoch)))
# чистим память

import gc

del train_img, train_label
# функции построение графиков потерь и точности

%matplotlib inline

def plot_loss(his, title):

    epoch = len(his.epoch)

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')

#     plt.plot(np.arange(0, epoch), his.history['val_loss'], label='valid_loss')



    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Loss')

    plt.legend(loc='upper right')

    plt.show()



def plot_acc(his, title):

    epoch = len(his.epoch)

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['categorical_accuracy'], label='categorical_accuracy')

#     plt.plot(np.arange(0, epoch), his.history['val_accuracy'], label='valid_accuracy')



    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Accuracy')

    plt.legend(loc='upper right')

    plt.show()
# строим графики

plot_loss(history,'Training Dataset')

plot_acc(history, 'Training Dataset')
# делаем предсказания

y_pred=model.predict(test_img)

print(y_pred)
# меняем файл submission

submission.loc[:,'healthy':'scab'] = y_pred
# сохраняем файл submission

submission.to_csv('submission.csv',index=False)