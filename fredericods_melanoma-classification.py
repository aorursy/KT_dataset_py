import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection

KAGGLE_PATH = "/kaggle/input/siim-isic-melanoma-classification/"
IMG_PATH_TRAIN = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
IMG_PATH_TEST = "/kaggle/input/siim-isic-melanoma-classification/jpeg/test/"

df_train = pd.read_csv(os.path.join(KAGGLE_PATH, "train.csv"))
df_train["kfold"] = -1
df_train["target"] = df_train["target"].astype(str)
df_train["image_file_name"] = df_train["image_name"] + ".jpg"
df_train = df_train.sample(frac=1).reset_index(drop=True) # shuffle dataframe
y = df_train.target.values
kf = model_selection.StratifiedKFold(n_splits=5)
for fold_, (train_idx, test_idx) in enumerate(kf.split(X=df_train, y=y)):
    df_train.loc[test_idx, "kfold"] = fold_
        
df_test = pd.read_csv(os.path.join(KAGGLE_PATH, "test.csv"))
IMG_HEIGHT, IMG_WIDTH = 96, 96
BATCH_SIZE = 32
K_FOLD = 0

train_image_generator = ImageDataGenerator(rescale=1./255)
valid_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_dataframe(
    df_train[df_train.kfold != K_FOLD],
    directory=IMG_PATH_TRAIN,
    x_col="image_file_name",
    y_col="target",
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

valid_data_gen = valid_image_generator.flow_from_dataframe(
    df_train[df_train.kfold == K_FOLD],
    directory=IMG_PATH_TRAIN,
    x_col="image_file_name",
    y_col="target",
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)
# Define base model pre-trained weights
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet')
base_model.trainable = False

# Add a classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

# Define model
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer='adam',
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.1),
    metrics=['binary_crossentropy']
)

model.summary()
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model.{epoch:02d}-{val_loss:.2f}.h5',
        save_best_only=True)
]

model.fit(
    train_data_gen,
    validation_data=valid_data_gen,
    epochs=10,
    callbacks=my_callbacks
)