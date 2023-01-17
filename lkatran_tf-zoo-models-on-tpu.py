## Уберите комменты и впишите свой USERNAME и Key from Kaggle

# !mkdir /root/.kaggle

# import json

# kaggle = {"username":"USERNAME","key":"Key from Kaggle"}

# with open('/root/.kaggle/kaggle.json', 'w') as f:

#     json.dump(kaggle, f)

# !chmod 600 /root/.kaggle/kaggle.json

# kaggle competitions download -c plant-pathology-2020-fgvc7
!pip install -q efficientnet
import math, re, os



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

import tensorflow as tf

import tensorflow.keras.layers as L

from sklearn import metrics

from sklearn.model_selection import train_test_split



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
AUTO = tf.data.experimental.AUTOTUNE

# Проверяем существующее оборудование и выбираем соответствующую стратегию использования вычислений

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    # онаружение TPU. Параметры не требуются если установленна переменная окружения TPU_NAME. На Kaggle это всегда True.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # если TPU отсутствует, то испльзуем стратегию по умолчанию для TF (CPU or GPU)



print("REPLICAS: ", strategy.num_replicas_in_sync)



# Путь к данным. Если работаете на Google Colaboratory, то замените KaggleDatasets().get_gcs_path() на путь к данным, который будет у вас

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# Конфигурация

EPOCHS = 40

BATCH_SIZE = 8 * strategy.num_replicas_in_sync
# функция, которая превращает айди картинки в полный путь к ней

def format_path(st):

    return GCS_DS_PATH + '/images/' + st + '.jpg'
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')



train_paths = train.image_id.apply(format_path).values

test_paths = test.image_id.apply(format_path).values



train_labels = train.loc[:, 'healthy':].values



## если планируете обучать модель с валидирующим набором данных

# train_paths, valid_paths, train_labels, valid_labels = train_test_split(

#     train_paths, train_labels, test_size=0.15, random_state=2020)
from matplotlib import pyplot as plt



# создаем сетку 2 на 5, для более компактного отображения символов и задаем размер их отображения

f, ax = plt.subplots(3, 6, figsize=(18, 7))

ax = ax.flatten()

# отрисовываем в цикле найденные топ N изображений частей графем

for i in range(18):

    img = plt.imread(f'../input/plant-pathology-2020-fgvc7/images/Train_{i}.jpg')

    ax[i].set_title(train[train['image_id']==f'Train_{i}'].melt()[train[train['image_id']==f'Train_{i}'].melt().value == 1]['variable'].values[0])

    ax[i].imshow(img)

print(img.shape)
# устанавливаем глобальные переменные

img_size = 768



# функция, которая читает изображение из файла и преобразовывает его к нужному размеру, а так же нормализует

def decode_image(filename, label=None, image_size=(img_size, img_size)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    

    if label is None:

        return image

    else:

        return image, label

# функция расширения данных

def data_augment(image, label=None, seed=2020):

    image = tf.image.random_flip_left_right(image, seed=seed)

    image = tf.image.random_flip_up_down(image, seed=seed)

    

    if label is None:

        return image

    else:

        return image, label
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .map(data_augment, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(512)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

## если планируете обучать модель с валидирующим набором данных

# valid_dataset = (

#     tf.data.Dataset

#     .from_tensor_slices((valid_paths, valid_labels))

#     .map(decode_image, num_parallel_calls=AUTO)

#     .batch(BATCH_SIZE)

#     .cache()

#     .prefetch(AUTO)

# )



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_paths)

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

)
# функция управляющая изменениями шага обучения в процессе тренировки нейронной сети

LR_START = 0.00001

LR_MAX = 0.0001 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 15

LR_SUSTAIN_EPOCHS = 3

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



# построим график изменения шага обучение в зависимости от эпох

rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
# тут можете заменять архитектуру EfficientNetB7 на любую другую из загруженных и наблюдать результат 

# VGG16, VGG19, InceptionV3, InceptionResNetV2, DenseNet121, DenseNet169, DenseNet201, Xception,

#ResNet50, ResNet50V2, ResNet101V2, ResNet152V2, NASNetLarge, EfficientNetL2



def get_model():

    base_model =  InceptionResNetV2(weights='imagenet',

                                 include_top=False, pooling='avg',

                                 input_shape=(img_size, img_size, 3))

    x = base_model.output

    predictions = Dense(train_labels.shape[1], activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=predictions)



with strategy.scope():

    model = get_model()

    

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])

model.summary()
model.fit(

            train_dataset, 

            steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,

            callbacks=[lr_callback],

            epochs=EPOCHS

         )
def display_training_curves(training, title, subplot):

    """

    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

    """

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

#     ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    #ax.set_ylim(0.28,1.05)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])
display_training_curves(

    model.history.history['loss'], 

    'loss', 211)

display_training_curves(

    model.history.history['categorical_accuracy'], 

    'accuracy', 212)
name_model = 'InceptionResNetV2.h5'



## если на Google Colab, то

# from google.colab import drive

# drive.mount('/content/drive')

# name_model = 'drive/My Drive/Colab Notebooks/efficientnet.h5'



model.save(name_model)



## загрузить модель

# model = tf.keras.models.load_model(name_model) # загрузить готовую модель для дальнейшего использования
probs = model.predict(test_dataset, verbose=1)

sub.loc[:, 'healthy':] = probs

sub.to_csv('submission.csv', index=False)

sub.head()
# создаем сетку 2 на 5, для более компактного отображения символов и задаем размер их отображения

f, ax = plt.subplots(3, 5, figsize=(18, 8))

ax = ax.flatten()

# отрисовываем в цикле найденные топ N изображений частей графем

for i in range(15):

    img = plt.imread(f'../input/plant-pathology-2020-fgvc7/images/Test_{i}.jpg')

    ax[i].set_title(sub[sub['image_id']==f'Test_{i}'].melt().iloc[1:][sub[sub['image_id']==f'Test_{i}'].melt().iloc[1:].value >= 0.8]['variable'].values[0])

    ax[i].imshow(img)

print(img.shape)
## если работаете с Google Colaboratory

# !kaggle competitions submit -c plant-pathology-2020-fgvc7 -f submission.csv -m "xception_1_efficient_1"