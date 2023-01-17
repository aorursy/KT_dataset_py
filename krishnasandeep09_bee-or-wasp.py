import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import gc

import tensorflow as tf
from tensorflow.keras.layers import *
df = pd.read_csv('../input/bee-vs-wasp/kaggle_bee_vs_wasp/labels.csv')
df.head()
for ind in df.index:
    df.loc[ind, 'path'] = df.loc[ind, 'path'].replace('\\','/')

df.head()
labels = list(df['label'].unique())
label_cts = df['label'].value_counts()
x = range(0,4)
plt.bar(x, label_cts, tick_label=labels)
plt.title('Classes in the data')
plt.show()
qualities = list(df['photo_quality'].unique())
quality_cts = list(df['photo_quality'].value_counts())
x = range(0,2)
plt.bar(x, quality_cts, tick_label=qualities)
plt.title('Image quality')
plt.show()
df = df.query('photo_quality == 1').reset_index(drop=True)
df['label'].value_counts()
df_train = df.query('is_validation == 0 & is_final_validation == 0').reset_index(drop=True)
df_valid = df.query('is_validation == 1').reset_index(drop=True)
df_test = df.query('is_final_validation == 1').reset_index(drop=True)

data_split = [len(df_train.index), len(df_valid.index), len(df_test.index)]
data_split_labels =['Train', 'Validation', 'Test']
x = range(0,3)
plt.bar(x, data_split, tick_label=data_split_labels)
plt.title('Data split')
plt.show()
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_gen = datagen.flow_from_dataframe(
    df_train, 
    directory='../input/bee-vs-wasp/kaggle_bee_vs_wasp/', 
    x_col="path", y_col="label", target_size=(IMG_SIZE, IMG_SIZE), 
    color_mode="rgb", 
    class_mode="categorical", 
    batch_size=BATCH_SIZE, 
    shuffle=True,
)

valid_gen = datagen.flow_from_dataframe(
    df_valid, 
    directory='../input/bee-vs-wasp/kaggle_bee_vs_wasp/', 
    x_col="path", y_col="label", target_size=(IMG_SIZE, IMG_SIZE), 
    color_mode="rgb", 
    class_mode="categorical", 
    batch_size=BATCH_SIZE, 
    shuffle=True,
)

test_gen = datagen.flow_from_dataframe(
    df_test, 
    directory='../input/bee-vs-wasp/kaggle_bee_vs_wasp/', 
    x_col="path", y_col="label", target_size=(IMG_SIZE, IMG_SIZE), 
    color_mode="rgb", 
    class_mode="categorical", 
    batch_size=BATCH_SIZE, 
    shuffle=True,
)
STEP_SIZE_TRAIN = train_gen.n//train_gen.batch_size
STEP_SIZE_VALID = valid_gen.n//valid_gen.batch_size
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, 
    decay_steps=250, 
    decay_rate=0.96, 
    staircase=True,
)

es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True,
)
pretrained_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet', 
    include_top=False , 
    input_shape=[IMG_SIZE, IMG_SIZE, 3]
)

pretrained_model.trainable = False
    
model = tf.keras.Sequential([
    pretrained_model, 
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
    loss = 'categorical_crossentropy',  
    metrics=['categorical_accuracy'],
)

model.summary()
history = model.fit(
    train_gen,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_gen,
    validation_steps=STEP_SIZE_VALID,
    epochs=EPOCHS,
    callbacks=[es_callback]
)

pd.DataFrame(history.history)[['categorical_accuracy', 'val_categorical_accuracy']].plot()
pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.show()

gc.collect()
model.evaluate(test_gen)