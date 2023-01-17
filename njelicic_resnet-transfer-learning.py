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



from keras.applications import ResNet50

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Dropout, BatchNormalization

from keras.models import Model 

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras import regularizers

from shutil import copy2
train_dir = "../input/aerial-cactus-identification/train/train/"

test_dir = "../input/aerial-cactus-identification/test/test/"

train_df = pd.read_csv('../input/aerial-cactus-identification/train.csv')

train_df.head()

label = train_df['has_cactus'].tolist()

fn = train_df['id'].tolist()
os.mkdir('../img_train/')

os.mkdir('../img_train/yes/')

os.mkdir('../img_train/no/')



os.mkdir('../img_val/')

os.mkdir('../img_val/yes/')

os.mkdir('../img_val/no/')




train_fn = fn[:14000]

val_fn = fn[14000:]



train_labels = label[:14000]

val_labels = label[14000:]



for i in zip(train_labels,train_fn):

    if i[0] == 1:

        copy2(str(train_dir+i[1]),str('../img_train/yes/'+i[1]))

    elif i[0] == 0:

        copy2(str(train_dir+i[1]),str('../img_train/no/'+i[1]))



for i in zip(val_labels,val_fn):

    if i[0] == 1:

        copy2(str(train_dir+i[1]),str('../img_val/yes/'+i[1]))

    elif i[0] == 0:

        copy2(str(train_dir+i[1]),str('../img_val/no/'+i[1]))

        
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True

)



val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(

        '../img_val/',

        target_size=(32,32),

        color_mode="rgb",

        batch_size=64,

        class_mode='binary')



train_generator = train_datagen.flow_from_directory(

        '../img_train/',

        target_size=(32,32),

        color_mode="rgb",

        batch_size=64,

        class_mode='binary')



step_train = train_generator.n//train_generator.batch_size

step_val= val_generator.n//val_generator.batch_size






base_model = ResNet50(weights="../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False,input_shape=(32,32,3))



for layer in base_model.layers:

    layer.trainable = True

    

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(rate=0.6)(x)

predictions = Dense(1, activation = 'sigmoid',activity_regularizer=regularizers.l1(0.01)) (x)







model = Model(inputs=base_model.input, outputs=predictions)



model.compile(optimizer='Adam', loss='binary_crossentropy',metrics=['accuracy'])

cp = ModelCheckpoint('ResNet50.hdf5' , monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)





reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=3, min_lr=0.0001)



model.fit_generator(

        train_generator,

        steps_per_epoch=step_train,

        validation_data= val_generator,

        validation_steps = step_val,

        epochs=20,

        callbacks=[cp,reduce_lr])
model.load_weights("ResNet50.hdf5")

print(os.listdir("../input/aerial-cactus-identification/"))
from PIL import Image

import tqdm

test_set = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

pred = np.empty((test_set.shape[0],))





for n in range(4000):

    data = np.array(Image.open('../input/aerial-cactus-identification/test/test/'+test_set.id[n]))

    data = data.astype(np.float32) / 255.

    pred[n] = model.predict(data.reshape((1, 32, 32, 3)))



test_set['has_cactus'] = pred

test_set.to_csv('sample_submission.csv', index=False)
test_set