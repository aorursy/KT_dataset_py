import tensorflow as tf

import os

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import categorical_crossentropy

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Model

from tensorflow.keras.applications import imagenet_utils

from tensorflow.keras.layers import MaxPool2D, Dropout, Flatten, GlobalAveragePooling2D

from tensorflow.keras.applications import MobileNet

from tensorflow.keras.applications import MobileNetV2



from PIL import ImageFile

import numpy as np





ImageFile.LOAD_TRUNCATED_IMAGES = True
os.system("git clone https://ranickpatra@bitbucket.org/ranickpatra/cropped-dog-breeds.git")
base_model = MobileNet(input_shape=(224,224,3), weights='imagenet',include_top=False)

x=base_model.output

x=GlobalAveragePooling2D()(x)

x=Dense(845,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.

x=Dropout(0.2)(x)

x=Dense(205,activation='relu')(x)

preds=Dense(101,activation='softmax')(x) #final layer with softmax activation



model=Model(inputs=base_model.input,outputs=preds)

#specify the inputs

#specify the outputs

#now a model has been created based on our architecture
for i, layer in enumerate(model.layers):

    print(i, layer.name)
for layer in model.layers[:87]:

    layer.trainable=False

for layer in model.layers[87:]:

    layer.trainable=True



model.summary()
def preprocess(dd):

    return dd / 127.5 - 1
train_datagen=ImageDataGenerator(

    preprocessing_function=preprocess,

    rotation_range=40,

    zoom_range=0.15,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.15,

    horizontal_flip=True,

    fill_mode="nearest"

) #included in our dependencies



train_generator=train_datagen.flow_from_directory('cropped-dog-breeds/train_balanced/',

                                                 target_size=(224,224),

                                                 color_mode='rgb',

                                                 batch_size=200,

                                                 class_mode='categorical',

                                                 shuffle=True)
train_generator.class_indices
os.system("rm -rf cropped-dog-breeds/test/")

os.system("rm -rf cropped-dog-breeds/train/")

os.system("rm -rf cropped-dog-breeds/val/")

os.system("rm saved-model*")
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Adam optimizer

# loss function will be categorical cross entropy

# evaluation metric will be accuracy

filepath = "saved-model-{epoch:02d}-{accuracy:.5f}.hdf5"

callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)



step_size_train=train_generator.n//train_generator.batch_size

model.fit_generator(generator=train_generator,

                   steps_per_epoch=step_size_train,

                   callbacks=[callback],

                   epochs=25)