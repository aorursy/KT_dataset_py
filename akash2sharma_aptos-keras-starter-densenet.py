# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/densenet-169"))



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

from sklearn.metrics import cohen_kappa_score
base_model=DenseNet169(weights = "../input/densenet-169/DenseNet-BC-169-32-no-top.h5",

                       include_top=False

                      )
x=base_model.output

x=GlobalAveragePooling2D()(x)

x=Dropout(0.5)(x)

preds=Dense(5, activation='sigmoid')(x)
model = Model(inputs=base_model.input,outputs=preds)
model.compile(

    loss='binary_crossentropy',

    optimizer=Adam(lr=0.00001),

    metrics=['accuracy']

)
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train_df["id_code"]=train_df["id_code"].apply(lambda x:x+".png")

train_df['diagnosis'] = train_df['diagnosis'].astype(str)

train_df.head()
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

    validation_split=0.1,   

    zoom_range = 0.3,

    )
train_generator=train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/aptos2019-blindness-detection/train_images",

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

    directory="../input/aptos2019-blindness-detection/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical", 

    classes=lbls,

    target_size=(img_size,img_size),

    subset='validation')
checkpoint = ModelCheckpoint(

    'dense_net.h5', 

    monitor='val_loss', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



history = model.fit_generator(

    generator=train_generator,

    steps_per_epoch=30,

    epochs=nb_epochs,

    validation_data=valid_generator,

    validation_steps = 30,

    callbacks=[checkpoint]

)
with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
sam_sub_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sam_sub_df["id_code"]=sam_sub_df["id_code"].apply(lambda x:x+".png")

print(sam_sub_df.shape)

sam_sub_df.head()
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(  

        dataframe=sam_sub_df,

        directory = "../input/aptos2019-blindness-detection/test_images",    

        x_col="id_code",

        target_size = (img_size,img_size),

        batch_size = 1,

        shuffle = False,

        class_mode = None

        )
predict=model.predict_generator(test_generator, steps = len(test_generator.filenames))
predict.shape
filenames=test_generator.filenames

results=pd.DataFrame({"id_code":filenames,

                      "diagnosis":np.argmax(predict,axis=1)})

results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])

results.to_csv("submission.csv",index=False)
results.head()