# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from skimage.io import imread,imshow

from glob import glob



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

base_dir = '../input/skin-cancer-mnist-ham10000'





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


id_path_dict = {os.path.splitext(os.path.basename(x))[0]:x for x in glob(os.path.join(base_dir,'*','*.jpg'))}

df = pd.read_csv('../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')

df['path'] = df['image_id'].map(id_path_dict.get)

df
_, ax1 = plt.subplots(2,2,figsize=(15,15))

df['dx'].value_counts().plot(kind='pie', ax=ax1[0,0])

df['sex'].value_counts().plot(kind= 'pie', ax=ax1[0,1])

df['dx_type'].value_counts().plot(kind= 'pie',ax=ax1[1,0])

df['localization'].value_counts().plot(kind= 'pie', ax=ax1[1,1])
fig , axm = plt.subplots(7,5,figsize=(25,30))

for axn , (type_name,type_row) in zip(axm, df.groupby(['dx'])):

    axn[0].set_title(type_name)

    for axc,(_,crows) in zip(axn,type_row.sample(5).iterrows()):

        axc.imshow(imread(crows['path']))

        

    
train_df, test_df = train_test_split(df,test_size=0.2)
fig , ax1 = plt.subplots(1,2,figsize=(10,5))

train_df['dx'].value_counts().plot(kind = 'bar', ax = ax1[0])

test_df['dx'].value_counts().plot(kind = 'bar', ax = ax1[1])
import tensorflow as tf

IDG = tf.keras.preprocessing.image.ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    zca_epsilon=1e-06,

    rotation_range=0.1,

    width_shift_range=0.3,

    height_shift_range=0.3,

    brightness_range=None,

    shear_range=0.1,

    zoom_range=0.4,

    channel_shift_range=0.0,

    fill_mode="nearest",

    cval=0.0,

    horizontal_flip=True,

    vertical_flip=True,

    rescale=None,

    preprocessing_function=None,

    data_format=None,

    validation_split=0.0,

    dtype=None,

)

train_DG = IDG.flow_from_dataframe(

    train_df,

    directory=None,

    x_col="path",

    y_col="dx",

    weight_col=None,

    target_size=(256, 256),

    color_mode="rgb",

    classes=None,

    class_mode="categorical",

    batch_size=32,

    shuffle=True,

    seed=None,

    save_to_dir=None,

    save_prefix="",

    save_format="png",

    subset=None,

    interpolation="nearest",

    validate_filenames=True

)

test_DG = IDG.flow_from_dataframe(

    test_df,

    directory=None,

    x_col="path",

    y_col="dx",

    weight_col=None,

    target_size=(256, 256),

    color_mode="rgb",

    classes=None,

    class_mode="categorical",

    batch_size=32,

    shuffle=True,

    seed=None,

    save_to_dir=None,

    save_prefix="",

    save_format="png",

    subset=None,

    interpolation="nearest",

    validate_filenames=True

)
import requests

model_URL = 'https://github.com/Fsunroo/RSNA-BoneAge/releases/download/v1.0/output.hdf5'

r = requests.get(model_URL)

with open('output.hdf5','wb') as f:

    f.write(r.content)

    f.close()
Source_model = tf.keras.models.load_model('output.hdf5')

Source_model.summary()
model = tf.keras.models.Sequential()

for layer in Source_model.layers[:-1]:

    model.add(layer)

model.add(tf.keras.layers.Dense(7,activation='softmax',name='dense_2'))

model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit_generator(train_DG,

                    validation_data = test_DG,

                    epochs=20

                   )
model.save('model_2.hdf5')

from IPython.display import FileLink

FileLink(r'model_2.hdf5')