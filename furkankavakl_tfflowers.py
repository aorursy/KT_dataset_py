import tensorflow_datasets as tfds

import numpy as np

import tensorflow as tf



tf.__version__
splits=['train[:80%]',

        'train[80%:90%]',

        'train[90%:]'

       ]
(train_ds, val_ds, test_ds),metadata = tfds.load('tf_flowers', split=splits, data_dir='./flowers', as_supervised=True, with_info=True)
metadata
split_weights=(80,10,10)

num_train, num_valid, num_test = (metadata.splits['train'].num_examples * weight/100 for weight in split_weights)

num_train
def resize_and_normalize(image, label):

    image=tf.cast(image, tf.float32)

    image=tf.image.resize(image,(128,128))

    image=image/255.0

    return image,label
def augment(image,label):

    image=tf.image.random_flip_left_right(image)

    image=tf.image.random_contrast(image, lower=0, upper=1.0)

    return image,label
train_ds=train_ds.map(resize_and_normalize)

val_ds=val_ds.map(resize_and_normalize)

test_ds=test_ds.map(resize_and_normalize)
train_ds=train_ds.map(augment)
train_ds=train_ds.shuffle(1024).batch(32)

val_ds=val_ds.batch(32)

test_ds=test_ds.batch(32)

train_ds=train_ds.prefetch(tf.data.experimental.AUTOTUNE)
def create_model():

    img_inputs = tf.keras.Input(shape=(128,128,3))

    conv_1     = tf.keras.layers.Conv2D(32,(3,3),activation='relu')(img_inputs)

    maxpool_1  = tf.keras.layers.MaxPooling2D((2,2))(conv_1)

    conv_2     = tf.keras.layers.Conv2D(64,(3,3),activation='relu')(maxpool_1)

    maxpool_2  = tf.keras.layers.MaxPooling2D((2,2))(conv_2)

    conv_3     = tf.keras.layers.Conv2D(64,(3,3),activation='relu')(maxpool_2)

    flatten    = tf.keras.layers.Flatten()(conv_3)

    dense_1    = tf.keras.layers.Dense(64, activation='relu')(flatten)

    output     = tf.keras.layers.Dense(metadata.features['label'].num_classes, activation='softmax')(dense_1)

    

    model      = tf.keras.Model(inputs=img_inputs, outputs=output)

    

    return model
model=create_model()

model.summary()
tf.keras.utils.plot_model(model,'flower_model_with_info.png',show_shapes=True)
import datetime, os



log_dir="logs/fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

os.makedirs(log_dir)
steps_per_epoch=int(num_train)//32

validation_steps=int(num_valid)//32
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model_checkpoint_callback=tf.keras.callbacks.ModelCheckpoint("training_checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5",period=5)

os.makedirs('training_checkpoints/',exist_ok=True)

early_stopping_checkpoint=tf.keras.callbacks.EarlyStopping(patience=20)
%load_ext tensorboard

%tensorboard --logdir logs
history=model.fit(train_ds.repeat(), 

                  epochs=20, 

                  steps_per_epoch=steps_per_epoch, 

                  validation_data=val_ds.repeat(),

                  validation_steps=validation_steps, 

                  callbacks=[tensorboard_callback,

                             model_checkpoint_callback,

                             early_stopping_checkpoint])