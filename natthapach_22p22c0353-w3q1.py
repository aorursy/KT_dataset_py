# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import tensorflow as tf

from tensorflow.keras.applications import vgg16, vgg19, resnet_v2

from tensorflow.keras import layers, Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import optimizers

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import cv2

import os

import seaborn as sns
np.random.seed(1)
df = pd.read_csv('/kaggle/input/super-ai-image-classification/train/train/train.csv')

df.category = df.category.astype('str')

df['path'] = '/kaggle/input/super-ai-image-classification/train/train/images/' + df.id

df['rand'] = np.random.rand(len(df))



train_df = df[df.rand <= 0.8]

val_df = df[df.rand > 0.8]



len(train_df), len(val_df)
candidates = [

    {

        'base': vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3)),

        'preprocessor': vgg16.preprocess_input,

        'dense': 64,

    },

    {

        'base': vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3)),

        'preprocessor': vgg19.preprocess_input,

        'dense': 64,

    },

    {

        'base': resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_shape=(256, 256, 3)),

        'preprocessor': resnet_v2.preprocess_input,

        'dense': 64,

    },

]
def plotAccHistory(history):

    history_df = pd.DataFrame(history.history)

#     history_df = history_df.iloc[::5]

    sns.pointplot(data=history_df, y='accuracy', x=history_df.index, color='red', label='train')

    sns.pointplot(data=history_df, y='val_accuracy', x=history_df.index, color='blue', label='val')

    plt.legend()

    plt.show()



def createModel(candidate):

    base = candidate['base']

    

    for l in base.layers:

        l.trainable = False

    x = base.output

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(candidate['dense'], activation='relu')(x)

    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(base.input, x)

    

    datagen = ImageDataGenerator(preprocessing_function=candidate['preprocessor'])

    train_gen = datagen.flow_from_dataframe(train_df, x_col='path', y_col='category', class_mode='binary', batch_size=256)

    val_gen = datagen.flow_from_dataframe(val_df, x_col='path', y_col='category', class_mode='binary', batch_size=256)

    

    return model, train_gen, val_gen
histories = []

for candidate in candidates:

    model, train_gen, val_gen = createModel(candidate)

    candidate['model'] = model

    candidate['train_gen'] = train_gen

    candidate['val_gen'] = val_gen

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(train_gen,

        validation_data=val_gen,

        steps_per_epoch=train_gen.n // train_gen.batch_size,

        epochs=30,

        verbose=True)

    histories.append(history)
for history in histories:

    plotAccHistory(history)
!ls /kaggle/input/super-ai-image-classification/val/val/images
selected_candidate = candidates[0]
filenames = []

paths = []

for file in os.listdir('/kaggle/input/super-ai-image-classification/val/val/images'):

    filenames.append(file)

    paths.append('/kaggle/input/super-ai-image-classification/val/val/images/' + file)

test_df = pd.DataFrame({ 'filename': filenames, 'path': paths })



# mock class for data generator

test_df['category'] = np.round(np.random.rand(len(test_df))).astype('int')

test_df['category'] = test_df['category'].astype('str')

test_df
datagen = ImageDataGenerator(preprocessing_function=selected_candidate['preprocessor'])

test_gen = datagen.flow_from_dataframe(test_df, x_col='path', y_col='category', class_mode='binary', batch_size=256, shuffle=False)
pred = selected_candidate['model'].predict_generator(test_gen, verbose=True)

pred = pred.reshape(pred.shape[0])

pred = np.round(pred)
test_df['prediction'] = pred.astype('int')

test_df[['filename', 'path', 'prediction']].to_csv('/kaggle/working/prediction.csv')
from IPython.display import FileLink

os.chdir(r'/kaggle/working')

FileLink(r'prediction.csv')