import tensorflow as tf

# последовательная модель (стек слоев)

from tensorflow.keras.models import Sequential, Model

# полносвязный слой и слой выпрямляющий матрицу в вектор

from tensorflow.keras.layers import Dense, Flatten, Input

# слой выключения нейронов и слой нормализации выходных данных (нормализует данные в пределах текущей выборки)

from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D, GaussianDropout

# слои свертки и подвыборки

from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

# работа с обратной связью от обучающейся нейронной сети

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# вспомогательные инструменты

from tensorflow.keras import utils

from tensorflow.keras.regularizers import *

from keras.applications.resnet50 import ResNet50

import numpy as np

import os

from tensorflow.random import set_seed

def seed_everything(seed):

    np.random.seed(seed)

    set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'



seed = 42

seed_everything(seed)



# работа с изображениями

from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

%matplotlib inline 



#  библиотека для работы с наборами данных на Kaggle

from kaggle_datasets import KaggleDatasets

import matplotlib.pyplot as plt

%matplotlib inline 

print("Tensorflow version " + tf.__version__)
# Обнаружение оборудования, возврат соответствующей стратегии распространения: TPU, GPU, CPU

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # Обнаружение TPU. Параметры среды не требуются, если задана переменная среды TPU_NAME. На Kaggle это всегда так.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # стратегия распространения по умолчанию в Tensorflow. Работает на CPU и одном GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
GCS_DS_PATH = KaggleDatasets().get_gcs_path() #получаем путь к наборам данных
IMAGE_SIZE = [192, 192] # при таком размере графическому процессору не хватит памяти. Используйте TPU

EPOCHS = 60

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



NUM_TRAINING_IMAGES = 12753

NUM_TEST_IMAGES = 7382

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE # находим количество шагов за эпоху
def decode_image(image_data):

    """Декодирует изображение в vyjujvthye. vfnhbwe (тензор)

    Нормализует данные и преобразовывает изображения к указанному размеру"""

    image = tf.image.decode_jpeg(image_data, channels=3) # Декодирование изображения в формате JPEG в тензор uint8.

    image = tf.cast(image, tf.float32) / 255.0  # преобразовать изображение в плавающее в диапазоне [0, 1]

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # явный размер, необходимый для TPU

#     image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string означает байтовую строку

        "class": tf.io.FixedLenFeature([], tf.int64),  # [] означает отдельный элемент

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT) # парсим отдельный пример в указанном формате

    image = decode_image(example['image']) # преобразуем изображение к нужному нам формату

    label = tf.cast(example['class'], tf.int32)

    return image, label # возвращает набор данных пар (изображение, метка)



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string означает байтовую строку

        "id": tf.io.FixedLenFeature([], tf.string),  # [] означает отдельный элемент

        # класс отсутствует, задача этого конкурса - предсказать классы цветов для тестового набора данных

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image']) # преобразуем изображение к нужному нам формату

    idnum = example['id']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    """Читает из TFRecords. Для оптимальной производительности одновременное чтение из нескольких

    файлов без учета порядка данных. Порядок не имеет значения, поскольку мы все равно будем перетасовывать данные"""



    ignore_order = tf.data.Options() # Представляет параметры для tf.data.Dataset.

    if not ordered:

        ignore_order.experimental_deterministic = False # отключить порядок, увеличить скорость



    dataset = tf.data.TFRecordDataset(filenames) # автоматически чередует чтение из нескольких файлов

    dataset = dataset.with_options(ignore_order) # использует данные сразу после их поступления, а не в исходном порядке

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord)

    # возвращает набор данных пар (изображение, метка), если метка = Истина, или пар (изображение, идентификатор), если метка = Ложь

    return dataset



def get_training_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/train/*.tfrec'), labeled=True)

    dataset = dataset.repeat() # набор обучающих данных должен повторяться в течение нескольких эпох

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    return dataset



def get_validation_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/val/*.tfrec'), labeled=True, ordered=False)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache() # кешируем набор

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/test/*.tfrec'), labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    return dataset



training_dataset = get_training_dataset()

validation_dataset = get_validation_dataset()
# В данной клетке размещаются эксперименты с использованием API и готовой сети ResNet.

# Над экспериментом написан результат

# Лучший результат без готовой нейронки: 0.45819

# Результат с использованием ResNet50: 0.86448

from tensorflow.keras.layers import Input, LeakyReLU



# RESULT: 0.45819

def get_model_api():

    nb_classes = 104

    def conv2D_block(x, num_conv=2, num_filters=64, drop_rate=0.25):

        for _ in range(num_conv):

            x = Conv2D(num_filters, (5, 5), activation='relu', padding='same',

                            kernel_regularizer=L1L2(l1=0.0001, l2=0.0001),

                            kernel_initializer='glorot_uniform')(x)

            x = BatchNormalization()(x)



        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = GaussianDropout(drop_rate)(x)

        return x



    def conv2D_net(x, nb_classes, num_block=4, num_conv_strategy=[2,2,3,3],

               num_filters_st=[64, 128, 256, 512], drop_rates=[0.25, 0.35, 0.45, 0.5]):

        for i in range(num_block):

            x = conv2D_block(x, num_conv_strategy[i], num_filters_st[i], drop_rates[i])

        x = Flatten()(x)

        x = Dense(1024, activation='relu', kernel_regularizer=L1L2(l1=0.0001, l2=0.0001),

                        kernel_initializer='glorot_uniform')(x)

        x = BatchNormalization()(x)

        x = GaussianDropout(0.8)(x)

        x = Dense(nb_classes, activation='softmax')(x)

        return x



    input_layer = Input(shape=(*IMAGE_SIZE, 3), name='input_layer')

    output_layer = conv2D_net(input_layer, nb_classes)

    model = Model(input_layer, output_layer)

    return model



# RESULT: 0.00902 :(

def get_model_api_1():

    nb_classes = 104

    def conv2D_block(x, num_conv=4, num_filters=64, drop_rate=0.25):

        for _ in range(num_conv):

            x = Conv2D(num_filters, (3, 3), activation='elu', padding='same',

                            kernel_regularizer=L1L2(l1=0.0001, l2=0.0001),

                            kernel_initializer='glorot_uniform')(x)

            x = BatchNormalization()(x)



        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = GaussianDropout(drop_rate)(x)

        return x



    def conv2D_net(x, nb_classes, num_block=4, num_conv_strategy=[2,2,3,3],

               num_filters_st=[32, 64, 128, 256], drop_rates=[0.25, 0.35, 0.45, 0.5]):

        for i in range(num_block):

            x = conv2D_block(x, num_conv_strategy[i], num_filters_st[i], drop_rates[i])

        x = Flatten()(x)

        x = Dense(nb_classes, activation='elu', kernel_regularizer=L1L2(l1=0.0001, l2=0.0001),

                        kernel_initializer='glorot_uniform')(x)

        x = BatchNormalization()(x)

        x = GaussianDropout(0.8)(x)

        x = LeakyReLU(104)(x)

        return x



    input_layer = Input(shape=(*IMAGE_SIZE, 3), name='input_layer')

    output_layer = conv2D_net(input_layer, nb_classes)

    model = Model(input_layer, output_layer)

    return model



#RESULT: 0.21091

def get_model_api_2():

    nb_classes = 104

    def conv2D_block(x, num_conv=2, num_filters=64, drop_rate=0.25):

        for _ in range(num_conv):

            x = Conv2D(num_filters, (5, 5), activation='relu', padding='same',

                            kernel_regularizer=L1L2(l1=0.0001, l2=0.0001),

                            kernel_initializer='glorot_uniform')(x)

            x = BatchNormalization()(x)



        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = GaussianDropout(drop_rate)(x)

        return x



    def conv2D_net(x, nb_classes, num_block=6, num_conv_strategy=[2,2,3,3,3,4],

               num_filters_st=[32, 64, 128, 256, 512, 1024], drop_rates=[0.25, 0.35, 0.45, 0.5, 0.55, 0.6]):

        for i in range(num_block):

            x = conv2D_block(x, num_conv_strategy[i], num_filters_st[i], drop_rates[i])

        x = Flatten()(x)

        x = Dense(1024, activation='relu', kernel_regularizer=L1L2(l1=0.0001, l2=0.0001),

                        kernel_initializer='glorot_uniform')(x)

        x = BatchNormalization()(x)

        x = GaussianDropout(0.7)(x)

        x = Dense(nb_classes, activation='softmax')(x)

        return x



    input_layer = Input(shape=(*IMAGE_SIZE, 3), name='input_layer')

    output_layer = conv2D_net(input_layer, nb_classes)

    model = Model(input_layer, output_layer)

    return model



# RESULT: 0.31675

def get_model_api_3():

    nb_classes = 104

    def conv2D_block(x, num_conv=3, num_filters=64, drop_rate=0.25):

        for _ in range(num_conv):

            x = Conv2D(num_filters, (5, 5), activation='relu', padding='same',

                            kernel_regularizer=L1L2(l1=0.0001, l2=0.0001),

                            kernel_initializer='glorot_uniform')(x)

            x = BatchNormalization()(x)



        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = GaussianDropout(drop_rate)(x)

        return x



    def conv2D_net(x, nb_classes, num_block=5, num_conv_strategy=[2,2,2,3,3],

               num_filters_st=[64, 128, 256, 512, 1024], drop_rates=[0.25, 0.35, 0.45, 0.5, 0.55]):

        for i in range(num_block):

            x = conv2D_block(x, num_conv_strategy[i], num_filters_st[i], drop_rates[i])

        x = Flatten()(x)

        x = Dense(1024, activation='relu', kernel_regularizer=L1L2(l1=0.0001, l2=0.0001),

                        kernel_initializer='glorot_uniform')(x)

        x = BatchNormalization()(x)

        x = GaussianDropout(0.8)(x)

        x = Dense(nb_classes, activation='softmax')(x)

        return x



    input_layer = Input(shape=(*IMAGE_SIZE, 3), name='input_layer')

    output_layer = conv2D_net(input_layer, nb_classes)

    model = Model(input_layer, output_layer)

    return model



# 0.86448

def get_model_resnet():

    input_layer = Input(shape=(*IMAGE_SIZE, 3), name='input_layer')

    nb_classes = 104



    model = tf.keras.applications.ResNet50(

    include_top=True,

    weights="imagenet",

    input_tensor=input_layer,

    input_shape=None,

    pooling=None,

    classes=1000)

        

    return model

# RESULT: 0.11677

def get_model():

 

    # Создаем последовательную модель

    model = Sequential()



    ## Первый сверточный блок

    # Первый сверточный слой

    model.add(Conv2D(64, (5, 5), input_shape=(*IMAGE_SIZE, 3), activation='relu'))

    model.add(BatchNormalization())

    # Второй сверточный слой

    model.add(Conv2D(64, (5, 5), activation='relu'))

    model.add(BatchNormalization())

    # # Первый слой подвыборки

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Первый Слой регуляризации Dropout

    model.add(GaussianDropout(0.25))



    ## Второй сверточный блок

    # Четвертый сверточный слой

    model.add(Conv2D(128, (5, 5), activation='relu'))

    model.add(BatchNormalization())

    # Пятый сверточный слой

    model.add(Conv2D(128, (5, 5), activation='relu'))

    model.add(BatchNormalization())

    # Второй слой подвыборки

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Второй Слой регуляризации Dropout

    model.add(GaussianDropout(0.35))



    ## Третий сверточный блок

    # Шестой сверточный слой

    model.add(Conv2D(256, (5, 5), activation='relu'))

    model.add(BatchNormalization())

    # Седьмой сверточный слой

    model.add(Conv2D(256, (5, 5), activation='relu'))

    model.add(BatchNormalization())

    # Восьмой сверточный слой

    model.add(Conv2D(256, (5, 5), activation='relu'))

    model.add(BatchNormalization())

    # Третий слой подвыборки

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Третий Слой регуляризации Dropout

    model.add(GaussianDropout(0.45))



    ## Четвертый сверточный блок

    # Девятый сверточный слой

    model.add(Conv2D(512, (5, 5), activation='relu'))

    model.add(BatchNormalization())

    # Десятый сверточный слой

    model.add(Conv2D(512, (5, 5), activation='relu'))

    model.add(BatchNormalization())

    # Одиннадцатый сверточный слой

    model.add(Conv2D(512, (5, 5), activation='relu'))

    model.add(BatchNormalization())

    # Четвертый слой подвыборки

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Четвертый Слой регуляризации Dropout

    model.add(GaussianDropout(0.5))

    

    ## Полносвязный блок

    # Слой преобразования данных из 2D представления в плоское

    model.add(Flatten())

    # Полносвязный слой для классификации

    model.add(Dense(1024, activation='relu'))

    # Четвертый слой нормализации данных

    model.add(BatchNormalization())

    # Четвертый Слой регуляризации Dropout

    model.add(GaussianDropout(0.8))

    # Выходной полносвязный слой

    model.add(Dense(104, activation='softmax'))

    return model





with strategy.scope():    

    model = get_model_resnet()

model.summary()
callbacks_list = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),

                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3),

                  ]



model.compile(

    optimizer='nadam',

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)



historical = model.fit(training_dataset, 

          steps_per_epoch=STEPS_PER_EPOCH, 

          epochs=EPOCHS, 

          callbacks=callbacks_list,

          validation_data=validation_dataset)
# Поскольку мы разделяем набор данных и выполняем итерацию отдельно для изображений и идентификаторов, порядок имеет значение.

test_ds = get_test_dataset(ordered=True) 



print('Вычисляем предсказания...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Создание файла submission.csv...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # все в одной партии

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')