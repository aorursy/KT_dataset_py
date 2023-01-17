# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import tensorflow as tf
import cv2
import openslide
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kaggle_datasets import KaggleDatasets

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32*strategy.num_replicas_in_sync
EPOCHS = 20
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224,224,3)
def decode_image(filename, label=None, image_size=IMAGE_SIZE):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label
class PreparedData():
    def __init__(self):
        # Data access
        GCS_PATH = KaggleDatasets().get_gcs_path('panda-resized-train-data-512x512')
        train_df = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
        train_dir=GCS_PATH+'/train_images/train_images/'

        train,valid = train_test_split(train_df, test_size=0.2, random_state=1)
        
        train_paths = train["image_id"].apply(lambda x: train_dir + x + '.png').values
        valid_paths = valid["image_id"].apply(lambda x: train_dir + x + '.png').values
        
        self.train_labels = pd.get_dummies(train['isup_grade']).astype('int32').values
        self.valid_labels = pd.get_dummies(valid['isup_grade']).astype('int32').values


        self.train_dataset = (
            tf.data.Dataset
            .from_tensor_slices((train_paths, self.train_labels))
            .map(decode_image, num_parallel_calls=AUTO)
            .repeat()
            .cache()
            .shuffle(512)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
        )
        self.valid_dataset = (
            tf.data.Dataset
            .from_tensor_slices((valid_paths, self.valid_labels))
            .map(decode_image, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .cache()
            .prefetch(AUTO)
        )
pd_obj = PreparedData()
train_data = pd_obj.train_dataset
valid_data = pd_obj.valid_dataset
train_labels = pd_obj.train_labels
valid_labels = pd_obj.valid_labels
class DefineModels():
    def get_tl_model(self):
        tl_model = tf.keras.applications.vgg16.VGG16(weights='/kaggle/input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                                                include_top=False, input_shape=IMAGE_SHAPE)
        model = tf.keras.models.Sequential()
        model.add(tl_model)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.01))
        model.add(tf.keras.layers.Dense(6, activation='softmax'))
        return model
        
    def run(self,model,epochs):
        model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=1e-3),metrics=['accuracy'])
        #history = model.fit(train_data,epochs=epochs,validation_data=valid_data)
        #history = model.fit(train_data,epochs=epochs,validation_data=valid_data,verbose=1)
        callbacks = None#[lr_callback, Checkpoint]
        history = model.fit(
            train_data, 
            validation_data = valid_data, 
            steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,            
            validation_steps=valid_labels.shape[0] // BATCH_SIZE,            
            callbacks=callbacks,
            epochs=epochs,
            verbose=1
)
        return history
        
model_obj = DefineModels()
model = model_obj.get_tl_model()
history = model_obj.run(model,10)

test_df = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')
test_dir = "../input/prostate-cancer-grade-assessment/test_images/"
def processed_image(image_path): 
    biopsy = openslide.OpenSlide(image_path)
    img = np.array(biopsy.get_thumbnail(size=IMAGE_SIZE))
    img = np.resize(img,IMAGE_SHAPE) / 255
    return img
data = list()
for i in range(test_df.shape[0]):
    data.append(self.processed_image(test_dir + test_df['image_id'].iloc[i]+'.tiff'))
test_df=pd.DataFrame(data)
test_df.columns=['image']
test_dataset = (
            tf.data.Dataset
            .from_tensor_slices(test_df)
            .batch(BATCH_SIZE)
        )
y_pred = np.argmax(model.predict(test_dataset))
test_df['isup_grade'] = y_pred
test_df = test_df[["image_id","isup_grade"]]
test_df.to_csv('submission.csv',index=False)

