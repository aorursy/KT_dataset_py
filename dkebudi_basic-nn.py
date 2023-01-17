from matplotlib import pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

import random 

import os

import cv2

import gc

from tqdm.auto import tqdm

import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.models import clone_model

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import plot_model

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import datetime as dt



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
HEIGHT = 236

WIDTH = 236



def get_n(df, field, n, top=True):

    top_graphemes = df.groupby([field]).size().reset_index(name='counts')['counts'].sort_values(ascending=not top)[:n]

    top_grapheme_roots = top_graphemes.index

    top_grapheme_counts = top_graphemes.values

    top_graphemes = class_map_df[class_map_df['component_type'] == field].reset_index().iloc[top_grapheme_roots]

    top_graphemes.drop(['component_type', 'label'], axis=1, inplace=True)

    top_graphemes.loc[:, 'count'] = top_grapheme_counts

    return top_graphemes



def image_from_char(char):

    image = Image.new('RGB', (WIDTH, HEIGHT))

    draw = ImageDraw.Draw(image)

    myfont = ImageFont.truetype('/kaggle/input/kalpurush/kalpurush-2.ttf', 120)

    w, h = draw.textsize(char, font=myfont)

    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 3), char, font=myfont)



    return image



def resize(df, size=64, need_progress_bar=True):

    resized = {}

    for i in range(df.shape[0]):

        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized
train_data = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

top_10_vowels = get_n(train_data, 'vowel_diacritic', 5)

top_10_roots = get_n(train_data, 'grapheme_root', 5)

top_10_consonants = get_n(train_data, 'consonant_diacritic', 5)

tops = [(top_10_vowels,'vowel_diacritic'), (top_10_roots, 'grapheme_root'), (top_10_consonants, 'consonant_diacritic')]



for top in tops:

    f, ax = plt.subplots(1, 5, figsize=(16, 8))

    ax = ax.flatten()

    for i in range(5):

        ax[i].imshow(image_from_char(top[0]['component'].iloc[i]), cmap='Greys')

        ax[i].set_title(f"{top[1]}:{i}")
import seaborn as sns

def plot_count(feature, title, df, size=1):

    '''

    Plot count of classes of selected feature; feature is a categorical value

    param: feature - the feature for which we present the distribution of classes

    param: title - title to show in the plot

    param: df - dataframe 

    param: size - size (from 1 to n), multiplied with 4 - size of plot

    '''

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show() 

    

plot_count('grapheme_root', 'grapheme_root (first most frequent 20 values - train)', train_data, size=4)

plot_count('vowel_diacritic', 'vowel_diacritic (train)', train_data, size=3)

plot_count('consonant_diacritic', 'consonant_diacritic (train)', train_data, size=3)
import seaborn as sns 

def plot_count_heatmap(feature1, feature2, df, size=1):   

    '''

    Heatmap showing the distribution of couple of features

    param: feature1 - ex: vowel_diacritic

    param: feature2 - ex: consonant_diacritic

    '''

    tmp = df.groupby([feature1, feature2])['grapheme'].count()

    df = tmp.reset_index()

    df

    df_m = df.pivot(feature1, feature2, "grapheme")

    f, ax = plt.subplots(figsize=(9, size * 4))

    sns.heatmap(df_m, annot=True, fmt='3.0f', linewidths=.5, ax=ax)

    

plot_count_heatmap('grapheme_root','consonant_diacritic', train_data, size=8)
train_data =  pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_0.parquet'), train_data, on='image_id').drop(['image_id'], axis=1)

train_labels = train_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme']]

train_data = train_data.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme'], axis=1)

train_labels.head()
train_data = resize(train_data)/255

train_data = train_data.values.reshape(-1, 64, 64, 1)
model_dict = {

    'grapheme_root': Sequential(),

    'vowel_diacritic': Sequential(),

    'consonant_diacritic': Sequential()

}
for model_type, model in model_dict.items():

    model.add(Conv2D(64, 7, activation="relu", padding="same", input_shape=[64, 64, 1]))

    model.add(layers.BatchNormalization(momentum=0.15))

    model.add(MaxPooling2D(2))

    model.add(Conv2D(128, 3, activation="relu", padding="same"))

    model.add(Conv2D(128, 3, activation="relu", padding="same"))

    model.add(MaxPooling2D(2))

    model.add(Conv2D(256, 3, activation="relu", padding="same"))

    model.add(Conv2D(256, 3, activation="relu", padding="same"))

    model.add(MaxPooling2D(2))

    model.add(Flatten())

    model.add(Dense(1024, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(512, activation="relu"))

    model.add(Dropout(0.5))

    if model_type == 'grapheme_root':

        model.add(layers.Dense(168, activation='softmax', name='root_out'))

    elif model_type == 'vowel_diacritic':

        model.add(layers.Dense(11, activation='softmax', name='vowel_out'))

    elif model_type == 'consonant_diacritic':

        model.add(layers.Dense(7, activation='softmax', name='consonant_out'))

    model.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])
from tensorflow.keras.utils import plot_model

print('grapheme_root')

plot_model(model_dict['grapheme_root'])
batch_size = 32

epochs = 5

history_list = []
model_types = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']

for target in model_types:

    Y_train = train_labels[target]

    Y_train = pd.get_dummies(Y_train).values

    x_train, x_test, y_train, y_test = train_test_split(train_data, Y_train, test_size=0.1, random_state=123)

    datagen = ImageDataGenerator()

    datagen.fit(x_train)

    history = model_dict[target].fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 

                                               epochs = epochs, validation_data = (x_test, y_test))

    history_list.append(history)
plt.figure()

for i in range(3):

    plt.plot(np.arange(0, epochs), history_list[i].history['accuracy'], label='train_accuracy')

    plt.plot(np.arange(0, epochs), history_list[i].history['val_accuracy'], label='val_accuracy')

    plt.title(model_types[i])

    plt.xlabel('Epoch #')

    plt.ylabel('Accuracy')

    plt.legend(loc='lower right')

    plt.show()
preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}
target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    print("Parquet: {}".format(i))

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img, need_progress_bar=False)/255

    X_test = X_test.values.reshape(-1, 64, 64, 1)



    for i, p in preds_dict.items():

        preds = model_dict[i].predict(X_test)

        preds_dict[i] = np.argmax(preds, axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(model_types):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(preds_dict[comp][k])

    del df_test_img

    del X_test

    gc.collect()



df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)

df_sample.to_csv('submission.csv',index=False)

df_sample.head()