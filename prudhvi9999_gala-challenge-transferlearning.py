import numpy as np 

import pandas as pd

import os

import tensorflow as tf

from tensorflow.keras.applications import ResNet152,ResNet50

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_df=pd.read_csv('/kaggle/input/hackerearth/dataset/train.csv')

test_df=pd.read_csv('/kaggle/input/hackerearth/dataset/test.csv')
train_img_path='/kaggle/input/hackerearth/dataset/Train Images/'

test_img_path='/kaggle/input/hackerearth/dataset/Test Images/'
datagen = ImageDataGenerator(

        rescale=1./255,

        validation_split=0.25,

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False,  

        rotation_range=10, 

        zoom_range = 0.1, 

        width_shift_range=0.2,

        height_shift_range=0.2,

        horizontal_flip=True,  

        vertical_flip=False

       )  

train_data_gen=datagen.flow_from_dataframe(train_df,

                                           train_img_path,

                                           x_col='Image',

                                           y_col='Class',

                                           subset='training',

                                           batch_size=32,

                                           seed=9,

                                           class_mode='categorical',

                                           target_size=(150,150),

                                           shuffle=True

                                          )
valid_data_gen=datagen.flow_from_dataframe(train_df,

                                           train_img_path,

                                           x_col='Image',

                                           y_col='Class',

                                           subset='validation',

                                           batch_size=32,

                                           seed=9,

                                           target_size=(150,150),

                                           class_mode='categorical',

                                           shuffle=True

                                          )
test_data_gen=ImageDataGenerator(rescale=1./255).flow_from_dataframe(test_df,

                                                                     test_img_path,

                                                                     target_size=(150,150),

                                                                     x_col='Image',

                                                                     y_col=None,

                                                                     class_mode=None,

                                                                     shuffle=False

                                                                    )
def get_callbacks(name):

     return [

        tf.keras.callbacks.EarlyStopping(monitor='val_acc',patience=5),

        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',

                                         factor=0.1,

                                         patience=2,

                                         cooldown=2,

                                         min_lr=0.00001,

                                         verbose=1),

        tf.keras.callbacks.TensorBoard(log_dir='logs/'+name)

    ]
def compile_and_fit(model,name,epochs=20):

    model.compile(optimizer='adam',

               loss='categorical_crossentropy',

               metrics=['accuracy']

              )

    model.fit(train_data_gen,

           epochs=epochs,

           steps_per_epoch=train_data_gen.n//train_data_gen.batch_size,

           callbacks=get_callbacks(name),

           validation_data=valid_data_gen,

           validation_steps=valid_data_gen.n//valid_data_gen.batch_size

          )

    return model
models={}
base_model = VGG16(

    weights='imagenet',

    include_top=False, 

    input_shape=[150,150,3],

    pooling='avg'

)

base_model.trainable=False
VGG16=Sequential([

    base_model,

    Dense(512,activation='relu'),

    Dense(4,activation='softmax')

])
models['VGG16']=compile_and_fit(VGG16,'VGG16')

print(models['VGG16'])
base_model=ResNet50(

    weights='imagenet',

    include_top=False, 

    input_shape=[150,150,3],

    pooling='avg')

base_model.trainable=False
Resnet50=Sequential([

    base_model,

    Dense(4,activation='softmax')

])
models['Resnet50']=compile_and_fit(Resnet50,"ResNet50")
base_model=ResNet152(

    weights='imagenet',

    include_top=False, 

    input_shape=[150,150,3],

    pooling='avg')

base_model.trainable=False
Resnet152=Sequential([

    base_model,

    Dense(4,activation='softmax')

])
models["Resnet152"]=compile_and_fit(Resnet152,"Resnet152")
base_model=VGG19(

    weights='imagenet',

    include_top=False, 

    input_shape=[150,150,3],

    pooling='avg')

base_model.trainable=False
VGG19=Sequential([

    base_model,

    Dropout(0.2),

    Dense(256,activation='relu'),

    Dense(4,activation='softmax')

])
models['VGG19']=compile_and_fit(VGG19,"VGG19")
def predictions(model):

    test_data_gen.reset()

    classes=model.predict(test_data_gen)

    predicted_class_indices=[np.argmax(i) for i in classes]

    labels = (train_data_gen.class_indices)

    labels = dict((v,k) for k,v in labels.items())

    predictions = [labels[k] for k in predicted_class_indices]

    filenames=test_data_gen.filenames

    results=pd.DataFrame({"Image":filenames,

                      "Class":predictions})

    results.to_csv(str(model)+".csv",index=False)
predictions(VGG16)
predictions(Resnet50)
predictions(Resnet152)
predictions(VGG19)