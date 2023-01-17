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



# Any results you write to the current directory are saved as output.
import os

import shutil

import re

import math



import pandas as pd

import numpy as np



import PIL.Image

import cv2



from random import shuffle

from glob import glob



from sklearn.model_selection import train_test_split



#from tensorflow.python.keras.applications import VGG16



from keras.applications import VGG16



from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



from tensorflow import keras

from tensorflow.keras.models import Model

#from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing.image import load_img, img_to_array



from keras import models

from keras import layers

from keras.callbacks import EarlyStopping, ModelCheckpoint



from keras.optimizers import Adam
base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')

train_dir = os.path.join(base_image_dir,'train_images/')

df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))

df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))

df = df.drop(columns=['id_code'])

df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe

df.head(10)
df.diagnosis.value_counts()
image_train, image_test, y_train, y_test = train_test_split(np.array(df.path), 

                                                            np.array(df.diagnosis), 

                                                            test_size=0.3,

                                                            random_state=123, 

                                                            stratify=df.diagnosis)
image_train.shape
image_and_class_train = dict(zip(image_train, y_train))

image_and_class_test = dict(zip(image_test, y_test))
IMG_SIZE = (224, 224)  # размер входного изображения сети

NUM_CLASSES = 5        # число классов
def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img
def circle_crop(path, img_size=(224,224), sigmaX=10):   

    """

    Create circular crop around image centre    

    """    

    

    img = cv2.imread(path)

    img = crop_image_from_gray(img)    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    height, width, depth = img.shape    

    

    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))

    

    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)

    img = cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)

    img = cv2.resize(img, img_size)

    return preprocess_input(img) 
# загружаем входное изображение и предобрабатываем

# ИСХОДНАЯ ПРЕДОБРАБОТКА ДАННЫХ



def load_image(path, target_size=IMG_SIZE):

    img = load_img(path, target_size=target_size)  # загрузка и масштабирование изображения

    array = img_to_array(img)

    return preprocess_input(array)  # предобработка для VGG16
# генератор для последовательного чтения обучающих данных с диска

def fit_generator(files, batch_size=32):

    while True:

        shuffle(files)

        for k in range(math.ceil(len(files) / batch_size)):   # округляем до ближайшего целого вверх

            i = k * batch_size                                # k -- номер батча в проходе                      

            j = i + batch_size

            if j > len(files):

                j = len(files)

                

            # если оставить функцию load_image, то будут исходные изображения

            #x = np.array([load_image(path)/255 for path in files[i:j]])         # картинки в виде матрицы

            

            # а это с предобработкой 

            x = np.array([circle_crop(path) for path in files[i:j]]) 

            

            label = np.array([image_and_class_train[path] for path in files[i:j]])   # метки классов

            y = keras.utils.to_categorical(label, num_classes=NUM_CLASSES)      # one hot кодирование

            yield (x, y)
# генератор последовательного чтения тестовых данных с диска

def predict_generator(files):

    while True:

        for path in files:

            

            # с предобработкой

            yield np.array([circle_crop(path)])

            

            # исходные

            #yield np.array([load_image(path)])

            
%matplotlib inline

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(20, 10))

for i, path in enumerate(image_train[:10], 1):

    subplot = fig.add_subplot(2, 5, i)

    

    # исходная картинка

    plt.imshow(plt.imread(path));

    



    subplot.set_title('{} \n label: {}'.format(os.path.basename(path), image_and_class_train[path]))
%matplotlib inline

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(20, 10))

for i, path in enumerate(image_train[:10], 1):

    subplot = fig.add_subplot(2, 5, i)

    

    # исходная картинка

    #plt.imshow(plt.imread(path));

    

    # новая предобработка

    image = circle_crop(path,sigmaX=10)

    plt.imshow(image)



    subplot.set_title('{} \n label: {}'.format(os.path.basename(path), image_and_class_train[path]))
conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
# фиксируем все веса предобученной сети кроме последнего блока 

set_trainable = False

for layer in conv_base.layers:

    if layer.name == 'block5_conv3':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False
conv_base.summary()
model_2 = models.Sequential()

model_2.add(conv_base)  # кусок VGG-16 добавлен в модель

model_2.add(layers.BatchNormalization())

model_2.add(layers.Flatten())

#model_1.add(layers.Dense(512, activation='relu'))

model_2.add(layers.Dense(NUM_CLASSES, activation='softmax'))

model_2.summary()
from keras import metrics
model_2.compile(optimizer=Adam(lr=0.001), 

              loss='categorical_crossentropy',  # функция потерь 'categorical_crossentropy' (log loss

              metrics=['accuracy', 'categorical_accuracy', metrics.Precision(), metrics.Recall(), metrics.AUC()])
shuffle(image_train)  # перемешиваем обучающую выборку



train_val_split = 100  # число изображений в валидационной выборке



validation_data = next(fit_generator(image_train[:train_val_split], train_val_split))



# запускаем процесс обучения

history = model_2.fit_generator(fit_generator(image_train[train_val_split:]),  # данные читаем функцией-генератором

        steps_per_epoch=10,  # число вызовов генератора за эпоху

        epochs=100,  # число эпох обучения

        validation_data=validation_data,

        callbacks=[ #EarlyStopping(patience = 5),

                   ModelCheckpoint(filepath='the_least_loss_new_preproc_gpu.h5',

                                  verbose=1,

                                  save_best_only=True)]

                               )
start = 0

plt.plot(history.history['loss'][start:])

plt.plot(history.history['val_loss'][start:])

plt.legend(['Train loss', 'Validation loss'])

plt.savefig('loss_preproc.png')
f1 = open('train_loss_preproc.txt', 'w')

f1.writelines('%s\n' % i for i in history.history['loss'][start:])

f1.close()
f2 = open('vall_loss_preproc.txt', 'w')

f2.writelines('%s\n' % i for i in history.history['val_loss'][start:])

f2.close()
plt.plot(history.history['accuracy'][start:])

plt.plot(history.history['val_accuracy'][start:])

plt.legend(['Train acc', 'Validation acc'])

plt.savefig('accuracy_preproc.png')



f3 = open('train_acc_preproc.txt', 'w')

f3.writelines('%s\n' % i for i in history.history['accuracy'][start:])

f3.close()



f4 = open('vall_acc_preproc.txt', 'w')

f4.writelines('%s\n' % i for i in history.history['val_accuracy'][start:])

f4.close()
plt.plot(history.history['precision_2'][start:])

plt.plot(history.history['val_precision_2'][start:])

plt.legend(['Train precision', 'Validation precision'])

plt.savefig('precision_preproc.png')



f5 = open('train_precision_preproc.txt', 'w')

f5.writelines('%s\n' % i for i in history.history['precision_2'][start:])

f5.close()



f6 = open('vall_precision_preproc.txt', 'w')

f6.writelines('%s\n' % i for i in history.history['val_precision_2'][start:])

f6.close()
plt.plot(history.history['recall_2'][start:])

plt.plot(history.history['val_recall_2'][start:])

plt.legend(['Train recall', 'Validation recall'])

plt.savefig('recall_preproc.png')



f7 = open('train_recall_preproc.txt', 'w')

f7.writelines('%s\n' % i for i in history.history['recall_2'][start:])

f7.close()



f8 = open('vall_recall_preproc.txt', 'w')

f8.writelines('%s\n' % i for i in history.history['val_recall_2'][start:])

f8.close()
plt.plot(history.history['auc_2'][start:])

plt.plot(history.history['val_auc_2'][start:])

plt.legend(['Train auc', 'Validation auc'])

plt.savefig('auc_preproc.png')



f9 = open('train_auc_preproc.txt', 'w')

f9.writelines('%s\n' % i for i in history.history['auc_2'][start:])

f9.close()



f10 = open('vall_auc_preproc.txt', 'w')

f10.writelines('%s\n' % i for i in history.history['val_auc_2'][start:])

f10.close()
model_2.save('preproc_100_epoch.h5')
# загружаем веса модели для наименьшего loss 

model_2.load_weights('the_least_loss_new_preproc_gpu.h5')
pred = model_2.predict_generator(predict_generator(image_test), len(image_test), max_queue_size=500)
%matplotlib inline

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(20, 20))

for i, (path, score) in enumerate(zip(image_test[80:][:10], pred[80:][:10]), 1):

    subplot = fig.add_subplot(math.ceil(i / 5), 5, i)

    plt.imshow(plt.imread(path))

    subplot.set_title('label: {} \n prediction: {} \n model confidence: {:.3f}'\

                      .format(image_and_class_test[path],

                              int(np.argmax(score)),

                             np.max(score)))
from IPython.display import HTML
create_download_link('/kaggle/output/project_3_model_2-vgg16.hdf5')
model_2.save('model_2.h5')
model_2.save_weights('my_model_weights.h5')
from IPython.display import FileLink, FileLinks

FileLinks('.')