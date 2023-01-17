# Author: Timur Abdualimov, SOVIET team

# Competition: Recommended system, SkillFctory

# First date code: 17.05.2020

# Used: Kaggle notebook, GPU!





import numpy as np

import pandas as pd 

import seaborn as sns



from wordcloud import WordCloud, STOPWORDS



import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import matplotlib

from matplotlib.pyplot import figure



%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (12,8)



import tensorflow as tf

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model

import tensorflow_hub as hub



import sys

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

    

RANDOM_SEED = 13



print('Python       :', sys.version.split('\n')[0])

print('Pandas       :', pd.__version__)

print('Numpy        :', np.__version__)

print('Tensorflow   :', tf.__version__)
def open_data():

    """ open datasets"""

    global train, test, sample_submission # объявляем переменные глобальными

    train = pd.read_csv('/kaggle/input/recommendations/train.csv', low_memory = False)

    train = train.drop_duplicates().reset_index(drop = True) # удалим дубликаты, если есть

    test = pd.read_csv('/kaggle/input/recommendations/test.csv', low_memory = False)

    sample_submission = pd.read_csv('/kaggle/input/recommendations/sample_submission.csv')

    

open_data() # открываем все и записываем датасет в переменные



def param_data(data): # посмотрим на данные

    """dataset required parameters """

    param = pd.DataFrame({

              'dtypes': data.dtypes.values,

              'nunique': data.nunique().values,

              'isna': data.isna().sum().values,

              'loc[0]': data.loc[0].values,

              }, 

             index = data.loc[0].index)

    return param



pd.concat([param_data(train), param_data(test)], 

          axis=1, 

          keys = [f'↓ ОБУЧАЮЩАЯ ВЫБОРКА ↓ {train.shape}', f'↓ ТЕСТОВАЯ ВЫБОРКА ↓ {test.shape}'],  

          sort=False)
def viz_na(data):

    """NA visualisation"""

    global cols

    cols = data.columns # запишем названия строки сделаем переменную глобальной

    # определяем цвета 

    # желтый - пропущенные данные, синий - не пропущенные

    colours = ['#000099', '#ffff00'] 

    sns.heatmap(data[cols].isnull(), cmap=sns.color_palette(colours))

    plt.show()





viz_na(train)

viz_na(test)
def stat_na_per_percent(data):

    print(f'{data.shape}')

    for col in data.columns:

        pct_missing = np.mean(data[col].isnull())

        print('{} - {}%'.format(col, round(pct_missing*100)))

    print("END", end = '\n\n')

stat_na_per_percent(train)

stat_na_per_percent(test)
def concat_train_test(train , test):

    """

    prepare final data, concat train and test

    """

    global orta

    train['sample'] = 1 # объединяем трейн и тест для совместных правок

    test['sample'] = 0

    test['rating'] = -2

    orta = train.append(test, sort = False).reset_index(drop = True) # закончили объединение и присвоили имя основной перемнной

    orta.drop(['Id'], axis = 1, inplace = True)



concat_train_test(train, test)
orta.select_dtypes(include = ['float64', 'int64'])
sns.distplot(orta['rating'], label = 'rating')

plt.show()
sns.countplot(x='verified', data=orta, label = 'verified');
sns.distplot(orta['rating'], label = 'rating')

plt.show()
orta.isna().sum()
# удалим столбцы с пропусками где процент больше 80%

orta = orta.drop(['image', 'vote'], axis = 1)
# создадим отдельные столбцы, которые показывают были ли пропущенные значения в столбцах и заполним пропуски

for i in orta.columns:

    if orta[i].isna().sum() != 0:

        orta[str(i) + '_isNAN'] = pd.isna(orta[i]).astype('uint8')

        orta[i] = orta[i].fillna('[]')



# удалим столбцы style, reviewerName, reviewTime за ненадобностью

orta = orta.drop(['style', 'reviewerName', 'reviewTime'], axis = 1)



# перевдем столбец verified в числовой

orta['verified'] = orta['verified'].astype(int)



# создадим столбцы где указывается длина текста в столбцах reviewText, summary

text_len_col = ['reviewText', 'summary']

for i in text_len_col:

    orta[str(i) + '_len'] = orta[i].apply(lambda x: len(x))



# переведем столбец с unixtime в нормальные часики    

orta['unixReviewTime'] = pd.to_datetime(orta['unixReviewTime'], unit='s')

# сколько прошло дней

orta['DaysPassed'] = (pd.datetime.now() - orta['unixReviewTime']).dt.days

# день недели в который оставили отзыв

orta['reviewWeekday'] = orta['unixReviewTime'].dt.weekday

# удалим столбец со временем за ненадобностью

orta = orta.drop(['unixReviewTime'], axis = 1)
sns.heatmap(orta.corr())

plt.show()
full_text = ' '.join([i for i in orta['summary']])



cloud = WordCloud(background_color='white', width=1200, height=1000).generate_from_text(full_text)

plt.figure(figsize=(25,16))

plt.axis('off')

plt.imshow(cloud);
train_data = orta.query('sample == 1').drop('sample', axis = 1)

test_data = orta.query('sample == 0').drop(['sample', 'rating'], axis = 1)



train_data = pd.get_dummies(train_data, prefix='', prefix_sep='', columns=['rating'])
input_1 = Input(shape=(), dtype=tf.string,  name = "Input_1")

input_2 = Input(shape=(), dtype=tf.string,  name = "Input_2")

input_3 = Input(shape=(11,), name = "Input_3")

embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"

hub_layer_1 = hub.KerasLayer(embedding, input_shape=[],  dtype=tf.string, trainable=True)

hub_layer_2 = hub.KerasLayer(embedding, input_shape=[],  dtype=tf.string, trainable=True)

    



x = hub_layer_1(input_1)

x = Dense(128, activation='elu')(x)

x = Model(inputs=input_1, outputs=x)





y = hub_layer_2(input_2)

y = Dense(128, activation='elu')(y)

y = Model(inputs=input_2, outputs=y)



z = Dense(128, activation="elu")(input_3)

z = Dense(128, activation="elu")(z)

z = Model(inputs=input_3, outputs=z)



u8 = concatenate([x.output, y.output, z.output])



q = Dense(128, activation='elu')(u8)

predictions = Dense(2, activation='softmax')(q)





tf.keras.backend.clear_session()

model = Model(inputs=[x.input, y.input, z.input], outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()    
history = model.fit([train_data['reviewText'], train_data['summary'], train_data.iloc[:,[0,3,4,5,6,7,8,9,10,11,12]]],

                    train_data.iloc[:,[-2,-1]],

                    epochs=3,

                    batch_size=200,

                    )
pred = model.predict([test_data['reviewText'], test_data['summary'], test_data.iloc[:,[0,3,4,5,6,7,8,9,10,11,12]]])
pred = np.argmax(pred, axis=-1)

sample_submission['rating'] = pred

sample_submission.to_csv('submission_2.csv', index=False)

sample_submission.head(3)
"""input_3 = Input(shape=(11,), name = "Input_3")

x = Dense(128, activation="elu")(input_3)

x = Dense(128, activation="elu")(x)

predictions = Dense(2, activation='softmax')(x)



tf.keras.backend.clear_session()

mod = Model(inputs=input_3, outputs=predictions)

mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



mod.summary()    

"""
"""his = mod.fit(train_data.iloc[:,[0,3,4,5,6,7,8,9,10,11,12]],

                    train_data.iloc[:,[-2,-1]],

                    epochs=3,

                    batch_size=100,

                    )

"""
"""

embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"

hub_layer = hub.KerasLayer(embedding, input_shape=[], 

                           dtype=tf.string, trainable=True)







x = hub_layer.output

x = Dense(128, activation='elu')(x)

x = Model(inputs=hub_layer.input, outputs=x)





y = hub_layer.output

y = Dense(128, activation='elu')(y)

y = Model(inputs=hub_layer.input, outputs=y)





u8 = concatenate([x.output, y.output])



q = Dense(128, activation='elu')(u8)

predictions = Dense(2, activation='softmax')(q)





tf.keras.backend.clear_session()

model = Model(inputs=[x.input, y.input], outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()

"""