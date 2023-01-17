!pip install -q tensorflow==2.3
#аугментации изображений

!pip install albumentations -q
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import sys

import PIL

import cv2

import re



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



# # keras

import tensorflow as tf

import tensorflow.keras.layers as L

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import albumentations



# plt

import matplotlib.pyplot as plt

#увеличим дефолтный размер графиков

from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

#графики в svg выглядят более четкими

%config InlineBackend.figure_format = 'svg' 

%matplotlib inline



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('Python       :', sys.version.split('\n')[0])

print('Numpy        :', np.__version__)

print('Tensorflow   :', tf.__version__)
def mape(y_true, y_pred):

    return np.mean(np.abs((y_pred-y_true)/y_true))
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
!pip freeze > requirements.txt
DATA_DIR = '../input/sf-dst-car-price-prediction-part2/'

train = pd.read_csv(DATA_DIR + 'train.csv')

test = pd.read_csv(DATA_DIR + 'test.csv')

sample_submission = pd.read_csv(DATA_DIR + 'sample_submission.csv')
train.info()
train.nunique()
# split данных

data_train, data_test = train_test_split(train, test_size=0.15, shuffle=True, random_state=RANDOM_SEED)
# Наивная модель

predicts = []

for index, row in pd.DataFrame(data_test[['model_info', 'productionDate']]).iterrows():

    query = f"model_info == '{row[0]}' and productionDate == '{row[1]}'"

    predicts.append(data_train.query(query)['price'].median())



# заполним не найденные совпадения

predicts = pd.DataFrame(predicts)

predicts = predicts.fillna(predicts.median())



# округлим

predicts = (predicts // 1000) * 1000



#оцениваем точность

print(f"Точность наивной модели по метрике MAPE: {(mape(data_test['price'], predicts.values[:, 0]))*100:0.2f}%")
#посмотрим, как выглядят распределения числовых признаков

def visualize_distributions(titles_values_dict):

  columns = min(3, len(titles_values_dict))

  rows = (len(titles_values_dict) - 1) // columns + 1

  fig = plt.figure(figsize = (columns * 6, rows * 4))

  for i, (title, values) in enumerate(titles_values_dict.items()):

    hist, bins = np.histogram(values, bins = 20)

    ax = fig.add_subplot(rows, columns, i + 1)

    ax.bar(bins[:-1], hist, width = (bins[1] - bins[0]) * 0.7)

    ax.set_title(title)

  plt.show()



visualize_distributions({

    'mileage': train['mileage'].dropna(),

    'modelDate': train['modelDate'].dropna(),

    'productionDate': train['productionDate'].dropna()

})
#используем все текстовые признаки как категориальные без предобработки

categorical_features = ['bodyType', 'brand', 'color', 'engineDisplacement', 'enginePower', 'fuelType', 'model_info', 'name',

  'numberOfDoors', 'vehicleTransmission', 'Владельцы', 'Владение', 'ПТС', 'Привод', 'Руль']



#используем все числовые признаки

numerical_features = ['mileage', 'modelDate', 'productionDate']
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

train['sample'] = 1 # помечаем где у нас трейн

test['sample'] = 0 # помечаем где у нас тест

test['price'] = 0 # в тесте у нас нет значения price, мы его должны предсказать, поэтому пока просто заполняем нулями



data = test.append(train, sort=False).reset_index(drop=True) # объединяем

print(train.shape, test.shape, data.shape)
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['description','sell_id',], axis = 1, inplace=True)

    

    

    # ################### Numerical Features ############################################################## 

    # Далее заполняем пропуски

    for column in numerical_features:

        df_output[column].fillna(df_output[column].median(), inplace=True)

    # тут ваш код по обработке NAN

    # ....

    

    # Нормализация данных

    scaler = MinMaxScaler()

    for column in numerical_features:

        df_output[column] = scaler.fit_transform(df_output[[column]])[:,0]

    

    

    

    # ################### Categorical Features ############################################################## 

    # Label Encoding

    for column in categorical_features:

        df_output[column] = df_output[column].astype('category').cat.codes

        

    # One-Hot Encoding: в pandas есть готовая функция - get_dummies.

    df_output = pd.get_dummies(df_output, columns=categorical_features, dummy_na=False)

    # тут ваш код не Encoding фитчей

    # ....

    

    

    # ################### Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    # ....

    

    

    # ################### Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    df_output.drop(['vehicleConfiguration'], axis = 1, inplace=True)

    

    return df_output
# Запускаем и проверяем, что получилось

df_preproc = preproc_data(data)

df_preproc.sample(10)
df_preproc.info()
# Теперь выделим тестовую часть

train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)

test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)



y = train_data.price.values     # наш таргет

X = train_data.drop(['price'], axis=1)

X_sub = test_data.drop(['price'], axis=1)
test_data.info()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=RANDOM_SEED)
model = CatBoostRegressor(iterations = 5000,

                          #depth=10,

                          #learning_rate = 0.5,

                          random_seed = RANDOM_SEED,

                          eval_metric='MAPE',

                          custom_metric=['RMSE', 'MAE'],

                          od_wait=500,

                          #task_type='GPU',

                         )

model.fit(X_train, y_train,

         eval_set=(X_test, y_test),

         verbose_eval=100,

         use_best_model=True,

         #plot=True

         )
test_predict_catboost = model.predict(X_test)

print(f"TEST mape: {(mape(y_test, test_predict_catboost))*100:0.2f}%")
sub_predict_catboost = model.predict(X_sub)

sample_submission['price'] = sub_predict_catboost

sample_submission.to_csv('catboost_submission.csv', index=False)
X_train.head(5)
model = Sequential()

model.add(L.Dense(512, input_dim=X_train.shape[1], activation="relu"))

model.add(L.Dropout(0.5))

model.add(L.Dense(256, activation="relu"))

model.add(L.Dropout(0.5))

model.add(L.Dense(1, activation="linear"))
model.summary()
# Compile model

optimizer = tf.keras.optimizers.Adam(0.01)

model.compile(loss='MAPE',optimizer=optimizer, metrics=['MAPE'])
checkpoint = ModelCheckpoint('../working/best_model.hdf5' , monitor=['val_MAPE'], verbose=0  , mode='min')

earlystop = EarlyStopping(monitor='val_MAPE', patience=50, restore_best_weights=True,)

callbacks_list = [checkpoint, earlystop]
history = model.fit(X_train, y_train,

                    batch_size=512,

                    epochs=500, # фактически мы обучаем пока EarlyStopping не остановит обучение

                    validation_data=(X_test, y_test),

                    callbacks=callbacks_list,

                    verbose=0,

                   )
plt.title('Loss')

plt.plot(history.history['MAPE'], label='train')

plt.plot(history.history['val_MAPE'], label='test')

plt.show();
model.load_weights('../working/best_model.hdf5')

model.save('../working/nn_1.hdf5')
test_predict_nn1 = model.predict(X_test)

print(f"TEST mape: {(mape(y_test, test_predict_nn1[:,0]))*100:0.2f}%")
sub_predict_nn1 = model.predict(X_sub)

sample_submission['price'] = sub_predict_nn1[:,0]

sample_submission.to_csv('nn1_submission.csv', index=False)
data.description
# TOKENIZER

# The maximum number of words to be used. (most frequent)

MAX_WORDS = 100000

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 256
# split данных

text_train = data.description.iloc[X_train.index]

text_test = data.description.iloc[X_test.index]

text_sub = data.description.iloc[X_sub.index]
%%time

tokenize = Tokenizer(num_words=MAX_WORDS)

tokenize.fit_on_texts(data.description)
tokenize.word_index
%%time

text_train_sequences = sequence.pad_sequences(tokenize.texts_to_sequences(text_train), maxlen=MAX_SEQUENCE_LENGTH)

text_test_sequences = sequence.pad_sequences(tokenize.texts_to_sequences(text_test), maxlen=MAX_SEQUENCE_LENGTH)

text_sub_sequences = sequence.pad_sequences(tokenize.texts_to_sequences(text_sub), maxlen=MAX_SEQUENCE_LENGTH)



print(text_train_sequences.shape, text_test_sequences.shape, text_sub_sequences.shape, )
# вот так теперь выглядит наш текст

print(text_train.iloc[6])

print(text_train_sequences[6])
model_nlp = Sequential()

model_nlp.add(L.Input(shape=MAX_SEQUENCE_LENGTH, name="seq_description"))

model_nlp.add(L.Embedding(len(tokenize.word_index)+1, MAX_SEQUENCE_LENGTH,))

model_nlp.add(L.LSTM(256, return_sequences=True))

model_nlp.add(L.Dropout(0.5))

model_nlp.add(L.LSTM(128,))

model_nlp.add(L.Dropout(0.25))

model_nlp.add(L.Dense(64, activation="relu"))

model_nlp.add(L.Dropout(0.25))
model_mlp = Sequential()

model_mlp.add(L.Dense(512, input_dim=X_train.shape[1], activation="relu"))

model_mlp.add(L.Dropout(0.5))

model_mlp.add(L.Dense(256, activation="relu"))

model_mlp.add(L.Dropout(0.5))
combinedInput = L.concatenate([model_nlp.output, model_mlp.output])

# being our regression head

head = L.Dense(64, activation="relu")(combinedInput)

head = L.Dense(1, activation="linear")(head)



model = Model(inputs=[model_nlp.input, model_mlp.input], outputs=head)
model.summary()
optimizer = tf.keras.optimizers.Adam(0.01)

model.compile(loss='MAPE',optimizer=optimizer, metrics=['MAPE'])
checkpoint = ModelCheckpoint('../working/best_model.hdf5', monitor=['val_MAPE'], verbose=0, mode='min')

earlystop = EarlyStopping(monitor='val_MAPE', patience=10, restore_best_weights=True,)

callbacks_list = [checkpoint, earlystop]
history = model.fit([text_train_sequences, X_train], y_train,

                    batch_size=512,

                    epochs=500, # фактически мы обучаем пока EarlyStopping не остановит обучение

                    validation_data=([text_test_sequences, X_test], y_test),

                    callbacks=callbacks_list

                   )
plt.title('Loss')

plt.plot(history.history['MAPE'], label='train')

plt.plot(history.history['val_MAPE'], label='test')

plt.show();
model.load_weights('../working/best_model.hdf5')

model.save('../working/nn_mlp_nlp.hdf5')
test_predict_nn2 = model.predict([text_test_sequences, X_test])

print(f"TEST mape: {(mape(y_test, test_predict_nn2[:,0]))*100:0.2f}%")
sub_predict_nn2 = model.predict([text_sub_sequences, X_sub])

sample_submission['price'] = sub_predict_nn2[:,0]

sample_submission.to_csv('nn2_submission.csv', index=False)
# убедимся, что цены и фото подгрузились верно

plt.figure(figsize = (12,8))



random_image = train.sample(n = 9)

random_image_paths = random_image['sell_id'].values

random_image_cat = random_image['price'].values



for index, path in enumerate(random_image_paths):

    im = PIL.Image.open(DATA_DIR+'img/img/' + str(path) + '.jpg')

    plt.subplot(3, 3, index + 1)

    plt.imshow(im)

    plt.title('price: ' + str(random_image_cat[index]))

    plt.axis('off')

plt.show()
size = (320, 240)



def get_image_array(index):

    images_train = []

    for index, sell_id in enumerate(data['sell_id'].iloc[index].values):

        image = cv2.imread(DATA_DIR + 'img/img/' + str(sell_id) + '.jpg')

        assert(image is not None)

        image = cv2.resize(image, size)

        images_train.append(image)

    images_train = np.array(images_train)

    print('images shape', images_train.shape, 'dtype', images_train.dtype)

    return(images_train)



images_train = get_image_array(X_train.index)

images_test = get_image_array(X_test.index)

images_sub = get_image_array(X_sub.index)
from albumentations import (

    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,

    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,

    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,

    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose

)





#пример взят из официальной документации: https://albumentations.readthedocs.io/en/latest/examples.html

augmentation = Compose([

    HorizontalFlip(),

    OneOf([

        IAAAdditiveGaussianNoise(),

        GaussNoise(),

    ], p=0.2),

    OneOf([

        MotionBlur(p=0.2),

        MedianBlur(blur_limit=3, p=0.1),

        Blur(blur_limit=3, p=0.1),

    ], p=0.2),

    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=1),

    OneOf([

        OpticalDistortion(p=0.3),

        GridDistortion(p=0.1),

        IAAPiecewiseAffine(p=0.3),

    ], p=0.2),

    OneOf([

        CLAHE(clip_limit=2),

        IAASharpen(),

        IAAEmboss(),

        RandomBrightnessContrast(),

    ], p=0.3),

    HueSaturationValue(p=0.3),

], p=1)



#пример

plt.figure(figsize = (12,8))

for i in range(9):

    img = augmentation(image = images_train[0])['image']

    plt.subplot(3, 3, i + 1)

    plt.imshow(img)

    plt.axis('off')

plt.show()
def make_augmentations(images):

  print('применение аугментаций', end = '')

  augmented_images = np.empty(images.shape)

  for i in range(images.shape[0]):

    if i % 200 == 0:

      print('.', end = '')

    augment_dict = augmentation(image = images[i])

    augmented_image = augment_dict['image']

    augmented_images[i] = augmented_image

  print('')

  return augmented_images
# NLP part

tokenize = Tokenizer(num_words=MAX_WORDS)

tokenize.fit_on_texts(data.description)
def process_image(image):

    return augmentation(image = image.numpy())['image']



def tokenize_(descriptions):

  return sequence.pad_sequences(tokenize.texts_to_sequences(descriptions), maxlen = MAX_SEQUENCE_LENGTH)



def tokenize_text(text):

    return tokenize_([text.numpy().decode('utf-8')])[0]



def tf_process_train_dataset_element(image, table_data, text, price):

    im_shape = image.shape

    [image,] = tf.py_function(process_image, [image], [tf.uint8])

    image.set_shape(im_shape)

    [text,] = tf.py_function(tokenize_text, [text], [tf.int32])

    return (image, table_data, text), price



def tf_process_val_dataset_element(image, table_data, text, price):

    [text,] = tf.py_function(tokenize_text, [text], [tf.int32])

    return (image, table_data, text), price



train_dataset = tf.data.Dataset.from_tensor_slices((

    images_train, X_train, data.description.iloc[X_train.index], y_train

    )).map(tf_process_train_dataset_element)



test_dataset = tf.data.Dataset.from_tensor_slices((

    images_test, X_test, data.description.iloc[X_test.index], y_test

    )).map(tf_process_val_dataset_element)



y_sub = np.zeros(len(X_sub))

sub_dataset = tf.data.Dataset.from_tensor_slices((

    images_sub, X_sub, data.description.iloc[X_sub.index], y_sub

    )).map(tf_process_val_dataset_element)



#проверяем, что нет ошибок (не будет выброшено исключение):

train_dataset.__iter__().__next__();

test_dataset.__iter__().__next__();

sub_dataset.__iter__().__next__();
#нормализация включена в состав модели EfficientNetB3, поэтому на вход она принимает данные типа uint8

efficientnet_model = tf.keras.applications.efficientnet.EfficientNetB3(weights = 'imagenet', include_top = False, input_shape = (size[1], size[0], 3))

efficientnet_output = L.GlobalAveragePooling2D()(efficientnet_model.output)
#строим нейросеть для анализа табличных данных

tabular_model = Sequential([

    L.Input(shape = X.shape[1]),

    L.Dense(512, activation = 'relu'),

    L.Dropout(0.5),

    L.Dense(256, activation = 'relu'),

    L.Dropout(0.5),

    ])
# NLP

nlp_model = Sequential([

    L.Input(shape=MAX_SEQUENCE_LENGTH, name="seq_description"),

    L.Embedding(len(tokenize.word_index)+1, MAX_SEQUENCE_LENGTH,),

    L.LSTM(256, return_sequences=True),

    L.Dropout(0.5),

    L.LSTM(128),

    L.Dropout(0.25),

    L.Dense(64),

    ])
#объединяем выходы трех нейросетей

combinedInput = L.concatenate([efficientnet_output, tabular_model.output, nlp_model.output])



# being our regression head

head = L.Dense(256, activation="relu")(combinedInput)

head = L.Dense(1,)(head)



model = Model(inputs=[efficientnet_model.input, tabular_model.input, nlp_model.input], outputs=head)

model.summary()
optimizer = tf.keras.optimizers.Adam(0.005)

model.compile(loss='MAPE',optimizer=optimizer, metrics=['MAPE'])
checkpoint = ModelCheckpoint('../working/best_model.hdf5', monitor=['val_MAPE'], verbose=0, mode='min')

earlystop = EarlyStopping(monitor='val_MAPE', patience=10, restore_best_weights=True,)

callbacks_list = [checkpoint, earlystop]
history = model.fit(train_dataset.batch(30),

                    epochs=100,

                    validation_data = test_dataset.batch(30),

                    callbacks=callbacks_list

                   )
plt.title('Loss')

plt.plot(history.history['MAPE'], label='train')

plt.plot(history.history['val_MAPE'], label='test')

plt.show();
model.load_weights('../working/best_model.hdf5')

model.save('../working/nn_final.hdf5')
test_predict_nn3 = model.predict(test_dataset.batch(30))

print(f"TEST mape: {(mape(y_test, test_predict_nn3[:,0]))*100:0.2f}%")
sub_predict_nn3 = model.predict(sub_dataset.batch(30))

sample_submission['price'] = sub_predict_nn3[:,0]

sample_submission.to_csv('nn3_submission.csv', index=False)
blend_predict = (test_predict_catboost + test_predict_nn3[:,0]) / 2

print(f"TEST mape: {(mape(y_test, blend_predict))*100:0.2f}%")
blend_sub_predict = (sub_predict_catboost + sub_predict_nn3[:,0]) / 2

sample_submission['price'] = blend_sub_predict

sample_submission.to_csv('blend_submission.csv', index=False)
# MLP

model_mlp = Sequential()

model_mlp.add(L.Dense(512, input_dim=X_train.shape[1], activation="relu"))

model_mlp.add(L.Dropout(0.5))

model_mlp.add(L.Dense(256, activation="relu"))

model_mlp.add(L.Dropout(0.5))
# FEATURE Input

# Iput

productiondate = L.Input(shape=[1], name="productiondate")

# Embeddings layers

emb_productiondate = L.Embedding(len(X.productionDate.unique().tolist())+1, 20)(productiondate)

f_productiondate = L.Flatten()(emb_productiondate)
combinedInput = L.concatenate([model_mlp.output, f_productiondate,])

# being our regression head

head = L.Dense(64, activation="relu")(combinedInput)

head = L.Dense(1, activation="linear")(head)



model = Model(inputs=[model_mlp.input, productiondate], outputs=head)
model.summary()
optimizer = tf.keras.optimizers.Adam(0.01)

model.compile(loss='MAPE',optimizer=optimizer, metrics=['MAPE'])
history = model.fit([X_train, X_train.productionDate.values], y_train,

                    batch_size=512,

                    epochs=500, # фактически мы обучаем пока EarlyStopping не остановит обучение

                    validation_data=([X_test, X_test.productionDate.values], y_test),

                    callbacks=callbacks_list

                   )
model.load_weights('../working/best_model.hdf5')

test_predict_nn_bonus = model.predict([X_test, X_test.productionDate.values])

print(f"TEST mape: {(mape(y_test, test_predict_nn_bonus[:,0]))*100:0.2f}%")
# 