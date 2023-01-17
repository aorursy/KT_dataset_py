import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

IMAGE_HT_WID = 299
#https://keras.io/preprocessing/image/

#https://medium.com/@arindambaidya168/https-medium-com-arindambaidya168-using-keras-imagedatagenerator-b94a87cdefad

#https://github.com/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb

BATCH_SIZE = 120

import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop

train_datagen = ImageDataGenerator(

                               rotation_range=15,

#                                width_shift_range=0.1,

#                                height_shift_range=0.1,

#                                shear_range=0.01,

#                                zoom_range=[0.9, 1.25],

#                                horizontal_flip=True,

#                                vertical_flip=False,

#                                #data_format='channels_last',

                              fill_mode='reflect',

                              channel_shift_range = 30,

#                               brightness_range=[0.5, 1.5],

                               validation_split=0.4,

                              # rescale=1./255

                              samplewise_center = True,

                              samplewise_std_normalization = True,

                              preprocessing_function = tf.keras.applications.inception_resnet_v2.preprocess_input

                               )







train_generator=train_datagen.flow_from_directory(

                   

                    directory="../input/images/",

                

                   subset="training",

                    batch_size=BATCH_SIZE,

                    seed=55551,

                    shuffle=True,

                    

                    class_mode="categorical",

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))



valid_generator=train_datagen.flow_from_directory(

                    

                    directory="../input/images/",

                   subset="validation",

                    batch_size=BATCH_SIZE,

                    seed=55551,

                    shuffle=True,

                  

                    class_mode="categorical",

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))
#https://www.tensorflow.org/tutorials/images/transfer_learning

#https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24

#https://www.kaggle.com/atikur/instant-gratification-keras-starter





import tensorflow as tf

from tensorflow import keras

base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(IMAGE_HT_WID, IMAGE_HT_WID, 3),

                                               include_top=False, 

                                               weights='imagenet')

#https://keras.io/getting-started/faq/

#base_model.load_weights('../input/mobilenet-v2-keras-weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5')

base_model.trainable = False

print(base_model.summary())

model = tf.keras.Sequential([

  base_model,

  keras.layers.GlobalAveragePooling2D(),

  keras.layers.Dense(101, activation='softmax',kernel_initializer='glorot_uniform',  kernel_regularizer=keras.regularizers.l2(.0005))

])



optimizer_sgd = tf.keras.optimizers.SGD(lr=.01, momentum=.9)

#opt = tf.keras.optimizers.Adam(lr=0.001)

model.compile(optimizer=optimizer_sgd , 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])

print(model.summary())

def schedule(epoch):

    if epoch < 5:

        return .01

    elif epoch < 7:

        return .002

    else:

        return .0004

lr_scheduler = keras.callbacks.LearningRateScheduler(schedule)
checkpointer = keras.callbacks.ModelCheckpoint(filepath='modelFood_dataset.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)


csv_logger = keras.callbacks.CSVLogger('modelFood_dataset.log')
EPOCHS=8

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size + 1

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size + 1 

history = model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=EPOCHS,

                    callbacks=[lr_scheduler,checkpointer,csv_logger],              

                    verbose=2

)
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
model.save_weights('food_dataset_checkpoint1')