from functools import partial

from glob import glob

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import StratifiedKFold, train_test_split
AUTO = tf.data.experimental.AUTOTUNE

# tf.config.experimental_run_functions_eagerly(True)



GCS_PATH = KaggleDatasets().get_gcs_path()
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
train_csv = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
train_csv
target_counts = train_csv.target.value_counts()

print(f"0: {target_counts[0]} - {target_counts[0]*100/(target_counts[0]+target_counts[1]):.2f}% of total")

print(f"1: {target_counts[1]} - {target_counts[1]*100/(target_counts[0]+target_counts[1]):.2f}% of total")

print(f"Ratio: {target_counts[0] / target_counts[1]:.2f} : 1")
train_df_0 = train_csv[train_csv.target == 0]

train_df_1 = train_csv[train_csv.target == 1]

train_df_1_resampled = train_df_1.sample(target_counts[0], replace=True)

print(f"Upsampled counts - ")

print(f"0: {len(train_df_0)}")

print(f"1: {len(train_df_1_resampled)}")
IMG_SIZE = [1024, 1024]

BATCH_SIZE = 32
image_dir = "../input/siim-isic-melanoma-classification/jpeg"

train_dir = "train"

test_dir = "test/"
balanced_train_df = pd.concat([train_df_0, train_df_1_resampled])

balanced_train_df.image_name = balanced_train_df.image_name + ".jpg"

balanced_train_df.target = balanced_train_df.target.astype(str)

balanced_train_df
train_df, vald_df = train_test_split(balanced_train_df, test_size=0.2, stratify=balanced_train_df.target, shuffle=True, random_state=0)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, 

                                 rotation_range=360,

                                 horizontal_flip=True,

                                 vertical_flip=True)

train_dataset = train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/siim-isic-melanoma-classification/jpeg/train",

    x_col="image_name",

    y_col="target",

    class_mode="binary",

    batch_size=BATCH_SIZE,

    target_size=IMG_SIZE,

    seed=0)

vald_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

vald_dataset = train_datagen.flow_from_dataframe(

    dataframe=vald_df,

    directory="../input/siim-isic-melanoma-classification/jpeg/train",

    x_col="image_name",

    y_col="target",

    class_mode="binary",

    batch_size=BATCH_SIZE,

    target_size=IMG_SIZE,

    seed=0)
n_train, n_vald = len(train_dataset), len(vald_dataset)
def initialize_model(model_name=""):

    #pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)

    pretrained_model = tf.keras.applications.Xception(input_shape=[*IMG_SIZE, 3], include_top=False, weights='imagenet')

    #pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])

    # EfficientNet can be loaded through efficientnet.tfkeras library (https://github.com/qubvel/efficientnet)

    #pretrained_model = efficientnet.tfkeras.EfficientNetB0(weights='imagenet', include_top=False)

    

    pretrained_model.trainable = False



    model = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        #tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(8, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')

    ])



    model.compile(

        optimizer='adam',

        loss = 'binary_crossentropy',

        metrics=['AUC']

    )



    return model
with strategy.scope():

    model = initialize_model()
TRAIN_STEPS = n_train // BATCH_SIZE

VALD_STEPS = n_vald // BATCH_SIZE

EPOCHS = 10

print(f"Training steps = {TRAIN_STEPS}, Validation steps = {VALD_STEPS}")
t = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=TRAIN_STEPS,

                    validation_data=vald_dataset, validation_steps=VALD_STEPS)#, callbacks=[lr_callback])
plt.plot(t)
model.save('../working/model.h5')
test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

test_df.image_name = test_df.image_name + ".jpg"
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_dataset = test_datagen.flow_from_dataframe(

    dataframe=test_df,

    directory="../input/siim-isic-melanoma-classification/jpeg/test",

    x_col="image_name",

    y_col=None,

    shuffle=False,

    target_size=IMG_SIZE,

    class_mode=None,

    batch_size=32)
outs = model.predict(test_dataset)
pred = pd.DataFrame({'image_name': test_df['image_name'].str.rstrip(".jpg"), 'target': outs.ravel()})

pred
pred.to_csv('submissions.csv', header=True, index=False)