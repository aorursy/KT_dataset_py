def prepare_tf():

    !pip uninstall tensorflow -y

    !pip install tensorflow-gpu==2.0.0-alpha0

    from tensorflow.python.ops import control_flow_util

    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

    from tensorflow.python.client import device_lib 

    print(device_lib.list_local_devices())
#prepare_tf()

import tensorflow as tf

print(tf.__version__)
#!pip uninstall tensorflow -y
#https://stackoverflow.com/questions/55106444/attributeerror-sequential-object-has-no-attribute-total-loss

#https://www.kaggle.com/vladminzatu/cactus-detection-with-tensorflow-2-0

#!pip install tensorflow==2.0.0-alpha0

#https://medium.com/tensorflow/test-drive-tensorflow-2-0-alpha-b6dd1e522b01

#!pip install tensorflow-gpu==2.0.0-alpha0

# !pip install tf-nightly-gpu-2.0-preview tfp-nightly

# from tensorflow.python.ops import control_flow_util

# control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
#https://www.codingforentrepreneurs.com/blog/install-tensorflow-gpu-windows-cuda-cudnn/

#https://stackoverflow.com/questions/51306862/how-to-use-tensorflow-gpu

#https://medium.com/tensorflow/test-drive-tensorflow-2-0-alpha-b6dd1e522b01

# from tensorflow.python.client import device_lib 

# print(device_lib.list_local_devices())
#!pip install --upgrade tf-nightly-gpu-2.0-preview tfp-nightly
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

print(os.listdir("../input/train/"))

print(len(os.listdir("../input/train/train/")))

df_sample_submission_file = pd.read_csv("../input/sample_submission_file.csv")

df_sample_submission_file.info()
print(len(os.listdir("../input/extraimages/extraimages/")))
df_sample_submission_file.head()
CLASS_MODE = 'categorical'
#https://stackoverflow.com/questions/55663880/how-to-use-featurewise-center-true-together-with-flow-from-directory-in-imagedat

from pathlib import Path

from PIL import Image



def read_pil_image(img_path, height, width):

        with open(img_path, 'rb') as f:

            return np.array(Image.open(f).convert('RGB').resize((width, height)))



def load_all_images(dataset_path, height, width, img_ext='jpg'):

    return np.array([read_pil_image(str(p), height, width) for p in 

                                    Path(dataset_path).rglob("*."+img_ext)]) 



IMAGE_HT_WID=96

BATCH_SIZE = 70 #100 

#https://software.intel.com/en-us/articles/hands-on-ai-part-14-image-data-preprocessing-and-augmentation

from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop

train_datagen = ImageDataGenerator(

                              #  featurewise_center=True,

                               # featurewise_std_normalization= True,

                              #  zca_whitening = False,

                                samplewise_center = True,

                                rotation_range=8,

                               width_shift_range=0.1,

                               height_shift_range=0.1,

                               shear_range=0.01,

                               zoom_range=[0.9, 1.25],

                               horizontal_flip=True,

                               vertical_flip=False,

                               #data_format='channels_last',

                              channel_shift_range = 0.1,

                              fill_mode='nearest',

                              brightness_range=[0.5, 1.5],

                               validation_split=0.25,

                               #rescale=1./255

                               )



#train_datagen.fit(load_all_images('../input/extraimages/extraimages/', IMAGE_HT_WID, IMAGE_HT_WID))





test_datagen = ImageDataGenerator(

   samplewise_center = True

    #     rescale=1./255

)



train_generator=train_datagen.flow_from_directory(

                    directory="../input/train/train/",

                    subset="training",

                    batch_size=BATCH_SIZE,

                    seed=42,

                    shuffle=True,

                    class_mode=CLASS_MODE,

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))



valid_generator=train_datagen.flow_from_directory(

                    directory="../input/train/train/",

                    subset="validation",

                    batch_size=BATCH_SIZE,

                    seed=42,

                    shuffle=True,

                    class_mode=CLASS_MODE,

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))

#https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras



# from sklearn.metrics import roc_auc_score



# def auroc(y_true, y_pred):

#     return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

import tensorflow as tf

keras = tf.keras
#https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

np.random.seed(42)

import random as rn

rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,

                              inter_op_parallelism_threads=1)



from keras import backend as K



tf.set_random_seed(1234)



sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)

#https://www.tensorflow.org/tutorials/images/transfer_learning



#from tensorflow import keras



base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_HT_WID, IMAGE_HT_WID, 3),

                                               include_top=False, 

                                               weights='imagenet')

base_model.trainable = False

print(base_model.summary())

model = tf.keras.Sequential([

  base_model,

  keras.layers.GaussianNoise(0.2),  

  keras.layers.GlobalAveragePooling2D(),

  keras.layers.Dense(5, activation='sigmoid')

])



# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), 

#               loss='categorical_crossentropy', 

#               metrics=['accuracy'])



# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), 

#               loss='sparse_categorical_crossentropy', 

#               metrics=['accuracy'])



#sgd = tf.keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

#sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), metrics=['accuracy'])



#print(model.summary())

len(base_model.layers)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, 

                                                                    patience=3, verbose=2, mode='auto',

                                                                    min_lr=1e-6)



early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 

                                            # min_delta=0, 

                                             patience=6, verbose=2, mode='auto',

                                             baseline=None, restore_best_weights=True)
EPOCHS= 30

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size + 1

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size + 1

history = model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=EPOCHS,

                    verbose=2,

                    #callbacks=[reduce_lr, early_stop]             

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
train_generator2=train_datagen.flow_from_directory(

                    directory="../input/train/train/",

                    subset="training",

                    batch_size=BATCH_SIZE,

                    seed=48,

                    shuffle=True,

                    class_mode=CLASS_MODE,

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))



valid_generator2=train_datagen.flow_from_directory(

                    directory="../input/train/train/",

                    subset="validation",

                    batch_size=BATCH_SIZE,

                    seed=48,

                    shuffle=True,

                    class_mode=CLASS_MODE,

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, 

                                                                    patience=2, verbose=2, mode='auto',

                                                                    min_lr=1e-6)



early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 

                                            # min_delta=0, 

                                             patience=4, verbose=2, mode='auto',

                                             baseline=None, restore_best_weights=True)
EPOCHS= 10 #40

STEP_SIZE_TRAIN=train_generator.n//train_generator2.batch_size + 1

STEP_SIZE_VALID=valid_generator.n//valid_generator2.batch_size + 1

history2 = model.fit_generator(generator=train_generator2,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator2,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=EPOCHS,

                    verbose=2,

                    callbacks=[reduce_lr, early_stop],

                    initial_epoch = 5          

)
import matplotlib.pyplot as plt

acc = history2.history['acc']

val_acc = history2.history['val_acc']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
train_generator3=train_datagen.flow_from_directory(

                    directory="../input/train/train/",

                    subset="training",

                    batch_size=BATCH_SIZE,

                    seed=55,

                    shuffle=True,

                    class_mode=CLASS_MODE,

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))



valid_generator3=train_datagen.flow_from_directory(

                    directory="../input/train/train/",

                    subset="validation",

                    batch_size=BATCH_SIZE,

                    seed=55,

                    shuffle=True,

                    class_mode=CLASS_MODE,

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.04, 

                                                                    patience=2, verbose=2, mode='auto',

                                                                    min_lr=1e-6)



early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 

                                            # min_delta=0, 

                                             patience=4, verbose=2, mode='auto',

                                             baseline=None, restore_best_weights=True)
EPOCHS= 15 #40

STEP_SIZE_TRAIN=train_generator.n//train_generator3.batch_size + 1

STEP_SIZE_VALID=valid_generator.n//valid_generator3.batch_size + 1

history3 = model.fit_generator(generator=train_generator3,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator3,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=EPOCHS,

                    verbose=2,

                    callbacks=[reduce_lr, early_stop],

                    initial_epoch = 10          

)
import matplotlib.pyplot as plt

acc = history3.history['acc']

val_acc = history3.history['val_acc']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
train_generator4=train_datagen.flow_from_directory(

                    directory="../input/train/train/",

                    subset="training",

                    batch_size=BATCH_SIZE,

                    seed=5555,

                    shuffle=True,

                    class_mode=CLASS_MODE,

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))



valid_generator4=train_datagen.flow_from_directory(

                    directory="../input/train/train/",

                    subset="validation",

                    batch_size=BATCH_SIZE,

                    seed=5555,

                    shuffle=True,

                    class_mode=CLASS_MODE,

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))
EPOCHS= 10 #40

STEP_SIZE_TRAIN=train_generator.n//train_generator4.batch_size + 1

STEP_SIZE_VALID=valid_generator.n//valid_generator4.batch_size + 1

history3 = model.fit_generator(generator=train_generator4,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator4,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=EPOCHS,

                    verbose=2,

                    callbacks=[reduce_lr, early_stop],

                    initial_epoch = 10          

)
# base_model.trainable = True

# len(base_model.layers)
# fine_tune_at = 100

# for layer in base_model.layers[:fine_tune_at]:

#     layer.trainable =  False
#https://www.kaggle.com/atikur/instant-gratification-keras-starter





# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), 

#               loss='sparse_categorical_crossentropy', 

#               metrics=['accuracy'])
# len(model.trainable_variables)
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 

#                                                                     patience=3, verbose=2, mode='auto',

#                                                                     min_lr=1e-6)
# EPOCHS= 4 #8



# history_tune = model.fit_generator(generator=train_generator,

#                     steps_per_epoch=STEP_SIZE_TRAIN,

#                     validation_data=valid_generator,

#                     validation_steps=STEP_SIZE_VALID,

#                     epochs=EPOCHS,

#                     verbose=2,

#                     callbacks=[reduce_lr, early_stop]                 

# )
# acc += history_tune.history['acc']

# val_acc += history_tune.history['val_acc']





# epochs = range(len(acc))



# plt.plot(epochs, acc, 'r', label='Training accuracy')

# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

# plt.title('Training and validation accuracy')

# plt.legend(loc=0)

# plt.figure()





# plt.show()
test_generator=test_datagen.flow_from_directory(

                directory="../input/test/test/",

                batch_size=BATCH_SIZE,

                seed=42,

                shuffle=False,

                class_mode=None,

                target_size=(IMAGE_HT_WID,IMAGE_HT_WID))
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size + 1

test_generator.reset()

pred=model.predict_generator(test_generator,

                steps=STEP_SIZE_TEST,

                verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
len(predicted_class_indices)
predicted_class_indices[:10]

#https://www.kaggle.com/hsinwenchang/keras-mobilenet-data-augmentation-visualize

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames

results=pd.DataFrame({"Category":predictions,

                      "id":filenames})

#subm = pd.merge(df_test, results, on='file_name')[['id','Category']]

results.head()
results.Category.value_counts()
results.loc[:,'id'] = results.id.str.replace('0/','')

results.head()
results.to_csv("submission.csv",index=False)