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
import pandas as pd

import json

import numpy as np

import os

import keras

import matplotlib.pyplot as plt

from keras.layers import Dense,GlobalAveragePooling2D,Dropout

from keras.applications import DenseNet169

from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.optimizers import Adam

from keras.callbacks import Callback,ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing import image

from keras.applications.inception_v3 import InceptionV3

import numpy as np



base_model = InceptionV3(weights='imagenet', include_top=False)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(' Training data : ', train.shape[0])

print('Testing data : ', test.shape[0])
display(train.head())
from keras import regularizers

x=base_model.output

x=GlobalAveragePooling2D()(x)

x=Dropout(0.5)(x)

preds=Dense(5, activation='softmax',kernel_regularizer=regularizers.l2(0.0001))(x)
model = Model(inputs=base_model.input,outputs=preds)
model.compile(

    loss='categorical_crossentropy',

    optimizer=Adam(lr=0.0001),

    metrics=['accuracy']

)

train_df = pd.read_csv('../input/train.csv')

train_df["id_code"]=train_df["id_code"].apply(lambda x:x+".png")

train_df['diagnosis'] = train_df['diagnosis'].astype(str)

train_df.head()
train_df.count()
nb_classes = 5

lbls = list(map(str, range(nb_classes)))

batch_size = 32

img_size = 224

nb_epochs = 30
train_datagen=ImageDataGenerator(

    rescale=1./255,

    featurewise_center=True,

    featurewise_std_normalization=True,

    zca_whitening=True,

    rotation_range=45,

    width_shift_range=0.2, 

    height_shift_range=0.2,

    horizontal_flip=True,

    vertical_flip=True,

    validation_split=0.2,   

    zoom_range = 0.3,

    )
train_generator=train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    classes=lbls,

    target_size=(img_size,img_size),

    subset='training')



print('break')



valid_generator=train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical", 

    classes=lbls,

    target_size=(img_size,img_size),

    subset='validation')
from keras.callbacks import EarlyStopping, ModelCheckpoint



es= EarlyStopping(monitor='val_loss', mode ='min', verbose = 1, patience = 10)

mc = ModelCheckpoint('modeldense.h5', monitor='val_loss', save_best_only = True, mode ='min', verbose = 1)



history = model.fit_generator(

    generator=train_generator,

    steps_per_epoch=30,

    epochs=nb_epochs,

    validation_data=valid_generator,

    validation_steps = 30,

    callbacks=[es,mc]

)
history.history
history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()


complete_datagen = ImageDataGenerator(rescale=1./255)

complete_generator = complete_datagen.flow_from_dataframe(  

        dataframe=train_df,

        directory = "../input/train_images/",

        x_col="id_code",

        target_size=(512, 512),

        batch_size=1,

        shuffle=False,

        class_mode=None)



STEP_SIZE_COMPLETE = complete_generator.n//complete_generator.batch_size

train_preds = model.predict_generator(complete_generator, steps=STEP_SIZE_COMPLETE)

train_preds = [np.argmax(pred) for pred in train_preds]
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix



labels=['0 - No DR','1 - Mild','2 - Moderate','3 - Severe','4 - Proliferative DR']

cnf_matrix=confusion_matrix(train_df['diagnosis'].astype('int'),train_preds)

cnf_matrix_norm=cnf_matrix.astype('float')/cnf_matrix.sum(axis=1)[:,np.newaxis]

df_cm=pd.DataFrame(cnf_matrix_norm,index=labels,columns=labels)

plt.figure(figsize=(16,7))

sns.heatmap(df_cm,annot=True,fmt='.2f',cmap='Blues')

plt.show()
from sklearn.metrics import cohen_kappa_score



print("Train Cohen Kappa score: %.3f" % cohen_kappa_score(train_preds, train_df['diagnosis'].astype('int'), weights='quadratic'))
test = pd.read_csv('../input/test.csv')

test["id_code"] = test["id_code"].apply(lambda x: x + ".png")





test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_dataframe(  

        dataframe=test,

        directory = "../input/test_images/",

        x_col="id_code",

        target_size=(512, 512),

        batch_size=1,

        shuffle=False,

        class_mode=None)



test_generator.reset()

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

preds = model.predict_generator(test_generator, steps=STEP_SIZE_TEST)

predictions = [np.argmax(pred) for pred in preds]
filenames = test_generator.filenames

results = pd.DataFrame({'id_code':filenames, 'diagnosis':predictions})

results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])

results.to_csv('submission.csv',index=False)

results.head(10)

f, ax = plt.subplots(figsize=(14, 8.7))

ax = sns.countplot(x="diagnosis", data=results, palette="GnBu_d")

sns.despine()

plt.show()