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

train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train_df['diagnosis'] = train_df['diagnosis'].astype('str')

train_df['id_code'] = train_df['id_code'].astype(str)+'.png'
from keras.preprocessing.image import ImageDataGenerator



datagen=ImageDataGenerator(

    rescale=1./255, 

    validation_split=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    zca_whitening = True)



batch_size = 16

image_size = 224







train_gen=datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/aptos2019-blindness-detection/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    target_size=(image_size,image_size),

    subset='training')



test_gen=datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/aptos2019-blindness-detection/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical", 

    target_size=(image_size,image_size),

    subset='validation')
y_train = train_df['diagnosis']

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)

num_classes = y_train.shape[1]
from keras.applications.densenet import DenseNet121

dense_121 = DenseNet121(weights='../input/densenet121/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

model.summary()
from keras.layers import GlobalAveragePooling2D, Flatten, Dense, GaussianDropout, BatchNormalization, Dropout, Conv2D, MaxPooling2D

from keras.models import Model

from keras import regularizers, optimizers



x = dense_121.get_layer('conv5_block16_2_conv').output

x = Conv2D(32, (3, 3), input_shape=[96,96,3], activation='relu')(x)

x = GlobalAveragePooling2D()(x)

#x = BatchNormalization()(x)

#x = GaussianDropout(0.3)(x)

#x = Conv2D(30, (5, 5), activation='relu', kernel_constraint=maxnorm(3))(x)

#x = MaxPooling2D(pool_size=(2, 2))(x)

#x = Dropout(0.2)(x)

#x = Flatten()(x)

x = Dense(32, activation='relu')(x)

# and a logistic layer -- let's say we have 5 classes

predictions = Dense(5, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=dense_121.input, outputs=predictions)

model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=0.0001, amsgrad=True), metrics=['accuracy'])
for layer in model.layers:

    layer.trainable = True
from keras.callbacks import EarlyStopping, ModelCheckpoint

es= EarlyStopping(monitor='val_loss', mode ='min', verbose = 1, patience = 20)

mc = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only = True, mode ='min', verbose = 1)


# train the model on the new data for a few epochs

model.fit_generator(generator=train_gen,              

                                    steps_per_epoch=len(train_gen),

                                    validation_data=test_gen,                    

                                    validation_steps=len(test_gen),

                                    epochs=50,

                                    callbacks = [es, mc], 

                                    use_multiprocessing = True,

                                    verbose=1)
from keras.models import load_model

model = load_model('model.h5')
submission_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

#submission_df['diagnosis'] = submission_df['diagnosis'].astype('str')

submission_df['filename'] = submission_df['id_code'].astype(str)+'.png'
submission_df.head(3)
submission_datagen=ImageDataGenerator(rescale=1./255)

submission_gen=submission_datagen.flow_from_dataframe(

    dataframe=submission_df,

    directory="../input/aptos2019-blindness-detection/test_images",

    x_col="filename",    

    batch_size=1,

    shuffle=False,

    class_mode=None, 

    target_size=(image_size,image_size)

)
predictions=model.predict_generator(submission_gen, steps = len(submission_gen))
max_probability = np.argmax(predictions,axis=1) 
submission_df.drop(columns=['filename'], inplace= True)

submission_df['diagnosis'] = max_probability

submission_df.to_csv('submission.csv', index=False)