import pandas as pd

import numpy as np

import os

import keras

import matplotlib.pyplot as plt

from keras.layers import Dense,GlobalAveragePooling2D

from keras.applications import ResNet50

from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.optimizers import Adam

from keras.callbacks import Callback,ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from sklearn.metrics import cohen_kappa_score

from keras.models import model_from_json

import cv2
base_model=ResNet50(weights='../input/resnet-50-weights-file/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False)
x=base_model.output





x=GlobalAveragePooling2D()(x)

x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.

x=Dense(512,activation='relu')(x) #dense layer 3

x=Dense(128,activation='relu')(x) #dense layer 3

x=Dense(64,activation='relu')(x) #dense layer 3

x=Dense(32,activation='relu')(x) #dense layer 3

preds=Dense(5,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=preds)
# for layer in model.layers[:30]:

#     layer.trainable=False

# for layer in model.layers[30:]:

#     layer.trainable=True



for layer in model.layers:

    layer.trainable = True
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train_df["id_code"]=train_df["id_code"].apply(lambda x:x+".png")

train_df['diagnosis'] = train_df['diagnosis'].astype(str)

train_df.head()
nb_classes = 5

lbls = list(map(str, range(nb_classes)))

batch_size = 32

img_size = 224

nb_epochs = 15
# train_datagen=ImageDataGenerator(

#     rescale=1./255, 

#     validation_split=0.25,

#     horizontal_flip = True,    

#     zoom_range = 0.3,

#     width_shift_range = 0.3,

#     height_shift_range=0.3

#     )



train_datagen=ImageDataGenerator(

    rescale=1./255,

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
model.compile(optimizer=Adam(lr=0.00005),loss='categorical_crossentropy',metrics=['accuracy'])
# Callbacks



checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
# Configure the TensorBoard callback and fit your model



# tensorboard_callback = TensorBoard("logs")



# model.fit_generator(generator=train_generator,

#                     steps_per_epoch=30,

#                     validation_data=valid_generator,                    

#                     validation_steps=30,

#                     epochs=nb_epochs,

#                     callbacks=[tensorboard_callback])
checkpoint = ModelCheckpoint(

    'resnet50_model.h5', 

    monitor='val_loss', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



model.compile(optimizer=Adam(lr=0.00005),loss='categorical_crossentropy',metrics=['accuracy'])



history = model.fit_generator(

    generator=train_generator,

    steps_per_epoch=30,

    epochs=15,

    validation_data=valid_generator,

    validation_steps = 30,

    callbacks=[checkpoint])
model.compile(optimizer=Adam(lr=0.00001),loss='categorical_crossentropy',metrics=['accuracy'])



history = model.fit_generator(

    generator=train_generator,

    steps_per_epoch=30,

    epochs=10,

    validation_data=valid_generator,

    validation_steps = 30,

    callbacks=[checkpoint])
model.compile(optimizer=Adam(lr=0.000005),loss='categorical_crossentropy',metrics=['accuracy'])



history = model.fit_generator(

    generator=train_generator,

    steps_per_epoch=30,

    epochs=5,

    validation_data=valid_generator,

    validation_steps = 30,

    callbacks=[checkpoint])
model_json = model.to_json()

with open("model_resnet.json", "w") as json_file:

    json_file.write(model_json)

model.save_weights("model_resnet50.h5")
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