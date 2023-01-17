import tensorflow as tf

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, _ in os.walk('/kaggle/input'):

    print(dirname)



for files in os.listdir('/kaggle/input/aptos2019-blindness-detection'):

    print(files)

# Any results you write to the current directory are saved as output.
names = []

for filenames in os.listdir('/kaggle/input/aptos2019-blindness-detection/train_images'):

    names.append(filenames)

    

len(names)
traindf = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')

traindf.head()
traindf['id_code'] = [(i + '.png') for i in traindf['id_code']]

traindf['diagnosis'] = traindf['diagnosis'].astype(str)

for i in traindf.head(5)['id_code']:

    print(i)
traindf.iloc[[1,3,4],:]
#This is your generator for the full dataset

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



train_generator = train_datagen.flow_from_dataframe(

        dataframe=traindf,

        directory='/kaggle/input/aptos2019-blindness-detection/train_images',

        x_col="id_code",

        y_col="diagnosis",

        target_size=(1000, 1000),

        batch_size=32)
#Splitting the names into 5 folds

from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

kf.get_n_splits(traindf['id_code'])
for train_index, test_index in kf.split(traindf['id_code']):

    #Uncomment if you want train and test

    #X_train, X_test = traindf.iloc[train_index], traindf.iloc[test_index]

    X_train = traindf.iloc[train_index]

    xtrain_generator = train_datagen.flow_from_dataframe(

        dataframe=X_train,

        directory='/kaggle/input/aptos2019-blindness-detection/train_images',

        x_col="id_code",

        y_col="diagnosis",

        target_size=(1000, 1000),

        batch_size=32)

    

    models_list = []

    #Add your models here.

    base_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(1024, activation='relu')(x)

    predictions = tf.keras.layers.Dense(5, activation='softmax')(x)

    for layer in base_model.layers:

        layer.trainable = False



    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    ###

    ###

    ### Uncomment the following lines to train your models

    #model.compile(optimizer='adam', loss='categorical_crossentropy')

    #model.fit_generator(xtrain_generator, steps_per_epoch=100, epochs=50)

    models_list.append(model)
#Just iterate through all models in models_list to and take their mean for final prediction

models_list[0].predict(np.zeros((1,1000,1000,3)))