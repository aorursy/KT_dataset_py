# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
DATA_PATH = '../input/aptos2019-blindness-detection'



TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train_images')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test_images')

TRAIN_LABEL_PATH = os.path.join(DATA_PATH, 'train.csv')

TEST_LABEL_PATH = os.path.join(DATA_PATH, 'test.csv')



df_train = pd.read_csv(TRAIN_LABEL_PATH)

df_test = pd.read_csv(TEST_LABEL_PATH)



print('num of train images ', len(os.listdir(TRAIN_IMG_PATH)))

print('num of test images  ', len(os.listdir(TEST_IMG_PATH)))
from sklearn.model_selection import train_test_split

df_train['diagnosis'] = df_train['diagnosis'].astype('str')

df_train = df_train[['id_code', 'diagnosis']]

if df_train['id_code'][0].split('.')[-1] != 'png':

    for index in range(len(df_train['id_code'])):

        df_train['id_code'][index] = df_train['id_code'][index] + '.png'

        

df_test = df_test[['id_code']]

if df_test['id_code'][0].split('.')[-1] != 'png':

    for index in range(len(df_test['id_code'])):

        df_test['id_code'][index] = df_test['id_code'][index] + '.png'



train_data = np.arange(df_train.shape[0])

train_idx, val_idx = train_test_split(train_data, train_size=0.8, random_state=2019)



X_train = df_train.iloc[train_idx, :]

X_val = df_train.iloc[val_idx, :]

X_test = df_test



print(X_train.shape)

print(X_val.shape)

print(X_test.shape)
from keras.preprocessing.image import ImageDataGenerator

import keras

num_classes = 5

img_size = (224, 224, 3)

#nb_train_samples = len(X_train)

#nb_validation_samples = len(X_val)

nb_test_samples = len(X_test)



batch_size = 32





datagen = ImageDataGenerator(

    preprocessing_function= \

    keras.applications.mobilenet.preprocess_input)

test_batches =datagen.flow_from_dataframe(

    dataframe=X_test,

    directory=TEST_IMG_PATH,

    x_col='id_code',

    y_col=None,

    target_size= img_size[:2],

    color_mode='rgb',

    class_mode=None,

    batch_size=batch_size,

    shuffle=False,

    seed=2019

)
# -*- coding: utf-8 -*-

"""

Created on Tue Jul 30 21:59:33 2019

https://www.kaggle.com/vbookshelf/skin-lesion-analyzer-tensorflow-js-web-app

@author: chopinforest

"""





import numpy as np

#import keras

#from keras import backend as K



import keras

from keras.layers import Dense, Dropout

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



#image_size = 224

num_classes=5



# create a copy of a mobilenet model

#keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)



mobile = keras.applications.mobilenet.MobileNet(weights='../input/mobilenet/mobilenet_1_0_224_tf.h5')

# CREATE THE MODEL ARCHITECTURE



# Exclude the last 5 layers of the above model.

# This will include all layers up to and including global_average_pooling2d_1

x = mobile.layers[-6].output



# Create a new dense layer for predictions

# 7 corresponds to the number of classes

x = Dropout(0.25)(x)

predictions = Dense(num_classes, activation='softmax')(x)



# inputs=mobile.input selects the input layer, outputs=predictions refers to the

# dense layer we created above.



model = Model(inputs=mobile.input, outputs=predictions)



# We need to choose how many layers we actually want to be trained.



# Here we are freezing the weights of all layers except the

# last 23 layers in the new model.

# The last 23 layers of the model will be trained.



for layer in model.layers[:-23]:

    layer.trainable = False

    

    

# Define Top2 and Top3 Accuracy



from keras.metrics import categorical_accuracy, top_k_categorical_accuracy



def top_3_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)



def top_2_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=2)

num_train_samples = len(X_train)

num_val_samples = len(X_val)

train_batch_size = 32

val_batch_size = 32





train_steps = np.ceil(num_train_samples / train_batch_size)

val_steps = np.ceil(num_val_samples / val_batch_size)









model.compile(Adam(lr=0.01), loss='categorical_crossentropy', 

              metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])





# Add weights to try to make the model more sensitive to melanoma



class_weights={

    0: 1.0, # akiec

    1: 1.0, # bcc

    2: 1.0, # bkl

    3: 2.0, # df

    4: 2.0, # mel # Try to make the model more sensitive to Melanoma.

}



filepath = "mobilenet_224.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1, 

                             save_best_only=True, mode='max')



reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2, 

                                   verbose=1, mode='max', min_lr=0.00001)

                              

                              

callbacks_list = [checkpoint, reduce_lr]

#from keras.models import load_model

#from keras.models import load_weights



model.load_weights('../input/my-new-weights/mobilenet_224.h5')

#model=load_model('../input/my-new-weights/mobilenet_224.h5')
from tqdm import tqdm

from math import ceil

# Apply TTA

preds_tta = []

tta_steps = 10

for i in tqdm(range(tta_steps)):

    test_batches.reset()

    preds = model.predict_generator(

        generator=test_batches ,

        steps =ceil(nb_test_samples/batch_size)

    )

    preds_tta.append(preds)
preds_mean = np.mean(preds_tta, axis=0)

predicted_class_indices = np.argmax(preds_mean, axis=1)
train_batches = datagen.flow_from_dataframe(

    dataframe=X_train, 

    directory=TRAIN_IMG_PATH,

    x_col='id_code',

    y_col='diagnosis',

    target_size=img_size[:2],

    color_mode='rgb',

    class_mode='categorical',

    batch_size=batch_size,

    seed=2019

)

labels = (train_batches.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

submission['diagnosis'] = predictions

submission.to_csv("submission.csv", index=False)

submission.head()
submission.pivot_table(index='diagnosis', aggfunc=len)
from matplotlib import pyplot as plt

import seaborn as sns

plt.figure(figsize=(12, 6))

sns.countplot(submission["diagnosis"])

plt.title("Number of data per each diagnosis")

plt.show()