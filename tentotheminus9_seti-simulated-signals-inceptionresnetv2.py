import os

from keras import applications

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.models import Sequential, Model 

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras import backend as k 

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from keras.preprocessing import image

import numpy as np

import math

import pandas as pd

from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

import scikitplot as skplt

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import warnings

warnings.filterwarnings('ignore')
train_dir = '../input/seti-data/primary_small/train/'

validation_dir = '../input/seti-data/primary_small/valid/'

test_dir = '../input/seti-data/primary_small/test/'



img_dim  = 197
#Generators

train_datagen = ImageDataGenerator(

  rotation_range = 180,

  horizontal_flip = True,

  vertical_flip = True,

  fill_mode = "reflect")



# Note that the validation data shouldn't be augmented!

validation_datagen = ImageDataGenerator()  

test_datagen = ImageDataGenerator()  
training_batch_size = 64

validation_batch_size = 64



train_generator = train_datagen.flow_from_directory(

  train_dir,                                                  

  classes = ('noise', 'squiggle', 'narrowband', 'narrowbanddrd', 'squarepulsednarrowband', 

            'squigglesquarepulsednarrowband', 'brightpixel'),

  target_size = (img_dim, img_dim),            

  batch_size = training_batch_size,

  class_mode = "categorical",

  shuffle = True,

  seed = 123)
validation_generator = validation_datagen.flow_from_directory(

  validation_dir,

  classes = ('noise', 'squiggle', 'narrowband', 'narrowbanddrd', 'squarepulsednarrowband', 

            'squigglesquarepulsednarrowband', 'brightpixel'),

  target_size = (img_dim, img_dim),

  batch_size = validation_batch_size,

  class_mode = "categorical",

  shuffle = True,

  seed = 123)
test_size = 700

test_batch_size = 1



test_generator = test_datagen.flow_from_directory(

  test_dir,

  classes = ('noise', 'squiggle', 'narrowband', 'narrowbanddrd', 'squarepulsednarrowband', 

            'squigglesquarepulsednarrowband', 'brightpixel'),

  target_size = (img_dim, img_dim),

  batch_size = test_batch_size,

  class_mode = "categorical",

  shuffle = False)
base = InceptionResNetV2(

  weights = '../input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',

  include_top = False,

  input_shape = (img_dim, img_dim, 3)

)
x = base.output

x = Flatten(input_shape=base.output_shape[1:])(x)

x = Dense(img_dim, activation="relu")(x)

x = Dropout(0.2)(x)

x = Dense(7, activation="softmax")(x)
model = Model(inputs=base.input, outputs=x)
for layer in model.layers:

   layer.trainable = True
model.compile(loss = "binary_crossentropy", optimizer = optimizers.rmsprop(lr=1e-4), metrics=["accuracy"])
#Train



training_step_size = 64

validation_step_size = 32



history = model.fit_generator(

  train_generator,

  steps_per_epoch = training_step_size,

  epochs = 50,

  validation_data = validation_generator,

  validation_steps = validation_step_size,

  verbose = 0,

)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
predictions = model.predict_generator(test_generator, steps = test_size, verbose = 1)
df = pd.DataFrame(predictions)
df['filename'] = test_generator.filenames
df['truth'] = ''

df['truth'] = df['filename'].str.split('/', 1, expand = True)
df['prediction_index'] = df[[0,1,2,3,4,5,6]].idxmax(axis=1)
df['prediction'] = ''

df['prediction'][df['prediction_index'] == 0] = 'noise'

df['prediction'][df['prediction_index'] == 1] = 'squiggle'

df['prediction'][df['prediction_index'] == 2] = 'narrowband'

df['prediction'][df['prediction_index'] == 3] = 'narrowbanddrd'

df['prediction'][df['prediction_index'] == 4] = 'squarepulsednarrowband'

df['prediction'][df['prediction_index'] == 5] = 'squigglesquarepulsednarrowband'

df['prediction'][df['prediction_index'] == 6] = 'brightpixel'
df.head()
cm = confusion_matrix(df['truth'], df['prediction'])

cm
cm_df = pd.DataFrame(cm)
cm_df.columns = ['noise', 'squiggle', 'narrowband', 'narrowbanddrd', 'squarepulsednarrowband', 

            'squigglesquarepulsednarrowband', 'brightpixel']
cm_df['signal'] = ('noise', 'squiggle', 'narrowband', 'narrowbanddrd', 'squarepulsednarrowband', 

            'squigglesquarepulsednarrowband', 'brightpixel')
cm_df
accuracy = accuracy_score(df['truth'], df['prediction'])

accuracy