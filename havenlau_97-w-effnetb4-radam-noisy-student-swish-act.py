# import standard libraries
import os
os.environ['TF_KERAS'] = '1'

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.io import imread
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sns.set(style='white', context='notebook', palette='deep')
import sys
print(sys.version)
!pip install -U efficientnet
!pip install keras-rectified-adam
# import keras
# from keras.models import Model
# from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
# from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# tf.keras
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten, GlobalAveragePooling2D

import efficientnet.tfkeras as enet
# load train data
train_data = pd.read_csv('/kaggle/input/syde522/train.csv')
train_dir = '/kaggle/input/syde522/train/train'
test_dir = '/kaggle/input/syde522/test'
train_data.head()
from keras.preprocessing.image import ImageDataGenerator

num_classes = len(train_data.Id)

# # detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# # instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# train_batch_size = 16 * tpu_strategy.num_replicas_in_sync
# val_batch_size = 16 * tpu_strategy.num_replicas_in_sync

train_batch_size = 64
val_batch_size = 64

val_split = 0.2 # 80:20 training to validation set
train_num_sample = int(num_classes*(1-val_split))
val_num_sample = int(num_classes*(val_split))

STEPS_PER_EPOCH = train_num_sample // train_batch_size
VALIDATION_STEPS = val_num_sample // val_batch_size

train_datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=val_split,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect'
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_data,
    directory = train_dir,
    x_col="Id",
    y_col="Category",
    target_size=(150,150),
    subset="training",
    batch_size=train_batch_size,
    shuffle=True,
    class_mode="categorical"
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe = train_data,
    directory = train_dir,
    x_col="Id",
    y_col="Category",
    target_size=(150,150),
    subset="validation",
    batch_size=val_batch_size,
    shuffle=True,
    class_mode="categorical"
)

# number of classes
num_classes = len(train_data.Category.unique())

# map class to index
integer_mapping = {x: i for i,x in enumerate(sorted(train_data.Category.unique()))}
print(integer_mapping)
# Preview 100 samples
plt.figure(figsize=(30, 30))
for idx, (_, entry) in enumerate(train_data.sample(n=100).iterrows()):
    
    plt.subplot(10, 10, idx+1)
    plt.imshow(imread(train_dir+'/'+entry.Id))
    plt.axis('off')
    plt.title(entry.Category)
    idx+=2
from tensorflow.keras.backend import sigmoid

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

print(tf.__version__)

print(tf.keras.__version__)

enet_model = enet.EfficientNetB4(include_top=False, input_shape=(150,150,3), pooling='avg', weights='noisy-student')

model = tf.keras.Sequential([
    enet_model,
    BatchNormalization(),
    Dropout(0.3),

    Dense(512),
    BatchNormalization(),
    Activation(swish_act),
    Dropout(0.5),

    Dense(128),
    BatchNormalization(),
    Activation(swish_act),
    Dense(8, activation="softmax")
])

model.summary()


# from keras_radam.training import RAdamOptimizer

# model.compile(optimizer=RAdamOptimizer(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# # instantiating the model in the strategy scope creates the model on the TPU
# # with tpu_strategy.scope():
# model = enet.EfficientNetB4(include_top=False, input_shape=(150,150,3), pooling='avg', weights='noisy-student')

# x = model.output

# x = BatchNormalization()(x)
# x = Dropout(0.7)(x)

# x = Dense(512)(x)
# x = BatchNormalization()(x)
# x = Activation(swish_act)(x)
# x = Dropout(0.5)(x)

# x = Dense(128)(x)
# x = BatchNormalization()(x)
# x = Activation(swish_act)(x)

# # Output layer, categorical one-hot output
# predictions = Dense(8, activation="softmax")(x)

# model_final = Model(inputs = model.input, outputs = predictions)

# #     model_final.summary()

# model_final.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    
# history = model_final.fit(
#     train_generator,
#     epochs = 140,
#     steps_per_epoch = STEPS_PER_EPOCH,
#     validation_data = val_generator,
#     validation_steps = VALIDATION_STEPS
# )

from keras_radam import RAdam

model.compile(optimizer=RAdam(lr=0.00008), loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30, restore_best_weights=True)
history = model.fit_generator(
        train_generator,
        epochs = 120,
        steps_per_epoch = STEPS_PER_EPOCH,
        validation_data = val_generator,
        validation_steps = VALIDATION_STEPS,
        callbacks=[es]
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.plot(epochs,acc,'bo',label = 'Training Accuracy')
plt.plot(epochs,val_acc,'b',label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
# preparing the submission

import glob
import os

test_datagen = ImageDataGenerator(
    rescale=1/255
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=1,
    shuffle=False,
    class_mode=None
)

preds = model.predict_generator(
    test_generator,
    steps=len(test_generator.filenames)
)

image_ids = [name.split('/')[-1] for name in test_generator.filenames]

# convert probability back to one-hot encoding
predictions = preds.argmax(axis=-1)
# map index to label strings
str_predictions = [sorted(train_data.Category.unique())[i] for i in predictions]

data = {'id': image_ids, 'Category':str_predictions} 
submission = pd.DataFrame(data)
print(submission.head())

submission.to_csv('submission.csv', index=False)
!ls