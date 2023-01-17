!pip install efficientnet
!pip install --upgrade keras

import keras
import numpy as np
import os
import random
import functools
import matplotlib.pylab as plt
from tensorflow.keras.models import load_model

# from classification_models.keras import Classifiers
import efficientnet.keras as efn
import efficientnet.tfkeras
# from tensorflow.keras.models import load_model

from PIL import Image
import cv2
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import optimizers
# from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from skimage.util import random_noise
AUTO = tf.data.experimental.AUTOTUNE

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPU ', tpu.master())
else:
    print('no tpu')
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
# 4. load callbacks
def get_callbacks(output_path, patience):
    callbacks = list()
    ckpt_path = os.path.join(output_path, '{epoch:02d}-{val_categorical_accuracy:.4f}.h5')
    print(f'get_callback: {ckpt_path}')
    callbacks.append(EarlyStopping(monitor='val_categorical_accuracy', patience=patience))
    callbacks.append(ModelCheckpoint(filepath=ckpt_path,  
                                    monitor='val_categorical_accuracy',  
                                    save_best_only=True, 
                                    verbose=1))
    return callbacks

# 5. load model
def get_model(input_size, output_classes):
    model = efn.EfficientNetB4(input_shape=(input_size, input_size, 3),  
                                classes=output_classes, weights=None)
    # model = efn.EfficientNetB4(input_shape=(input_size,input_size,3), 
    #                             classes=output_classes, weights=None)
    return model

def preprocess_func(img):
    img = Image.fromarray(img, 'RGB')
    # print(img.size)
    # img.show()
    # img = img.convert('RGB')
    w, h = img.size
    rand = random.randint(0, 1)
    if rand == 0:
        return np.asarray(img)
    rand = random.randint(70, 99) / 100
    # processed_img = img.resize((int(w * rand), int(h * rand)))
    processed_img = img.resize((224, 224))
    # processed_img.show()
    # processed_img = processed_img.resize((w, h))
    return np.asarray(processed_img)
# print(os.getcwd())
# # os.chdir('../input/generated-web-element-datagradient-image-solid/')
# !../input
# print(os.getcwd())

# train_datagen = ImageDataGenerator(rescale=1./255)#preprocessing_function=preprocess_func)
# test_datagen = ImageDataGenerator(rescale=1./255)#preprocessing_function=preprocess_func)
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
train_generator = train_datagen.flow_from_directory(f'/kaggle/input/generated-web-element-datagradient-image-solid/images/train/',
                                                target_size = (224, 224),
                                                batch_size = 2,
                                                class_mode = 'categorical',
                                                shuffle=True)

test_generator = test_datagen.flow_from_directory(f'/kaggle/input/generated-web-element-datagradient-image-solid/images/test/',
                                            target_size = (224, 224),
                                            batch_size = 2,
                                            class_mode = 'categorical',
                                            shuffle=True)
# Need this line so Google will recite some incantations
# for Turing to magically load the model onto the TPU
with strategy.scope():
    enet = efn.EfficientNetB4(
        input_shape=(224, 224, 3),
        # weights='imagenet',
        include_top=False
    )

    model = keras.Sequential([
        enet,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(3, activation='softmax')
    ]) 

    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.0001),
        # loss = 'sparse_categorical_crossentropy',
        # metrics=['sparse_categorical_accuracy']
        loss = 'categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
model.summary()
'''
# load model
model = get_model(input_size=224, output_classes=3)    
adam = optimizers.Adam(lr=0.0003)
# sgd = optimizers.SGD(lr=0.0002, clipvalue=0.5)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])

# optimize
model.fit_generator(train_generator, 
                    epochs = 30,
                    steps_per_epoch=2000,
                    callbacks=callbacks,
                    validation_data=test_generator, 
                    validation_steps=1000)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
'''
try:
    os.mkdir('./model_checkpoint')
except:
    pass
# output_path = os.getcwd()
output_path = 'model_checkpoint'
callbacks = get_callbacks(output_path, patience=3)

history = model.fit_generator(
    train_generator, 
    steps_per_epoch=1000,
    epochs=20, 
    callbacks=callbacks,
    validation_data=test_generator,
    validation_steps=200)
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)
display_training_curves(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'], 'accuracy', 212)
model = load_model("model_checkpoint/09-0.9775.h5")
answer_cnt = 0
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
valid_generator = valid_datagen.flow_from_directory(f'/kaggle/input/generated-web-element-datagradient-image-solid/images/valid/',
                                            target_size = (224, 224),
                                            batch_size = 1,
                                            class_mode = 'categorical',
                                            shuffle=True)
val_num = 1000
for i, data in enumerate(valid_generator):
    if i == val_num:
        break
    (x, gt_y) = data
    pred_y = model.predict(x)
    # print(pred_y)
    pred_y = np.argmax(pred_y, axis=1)
    ans_idx = pred_y[0]
    if gt_y[0][ans_idx] == 1.:
        answer_cnt += 1
    print(f"{i}) pred_y: {pred_y}, gt_y: {gt_y}, {gt_y[0][ans_idx] == 1.}")

print(f"{answer_cnt}/{val_num}")
import random


# model = load_model("model_checkpoint/09-0.9775.h5")
answer_cnt = 0
bad_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
bad_generator = valid_datagen.flow_from_directory(f'/kaggle/input/goodbadexamples/bad/',
                                            target_size = (224, 224),
                                            batch_size = 1,
                                            class_mode = 'categorical',
                                            shuffle=True)
bad_num = 60
for i, data in enumerate(bad_generator):
    if i == val_num:
        break
    (x, gt_y) = data
    pred_y = model.predict(x)
    # print(pred_y)
    pred_y = np.argmax(pred_y, axis=1)
    ans_idx = pred_y[0]
    if gt_y[0][ans_idx] == 1.:
        answer_cnt += 1
    print(f"{i}) pred_y: {pred_y}, gt_y: {gt_y}, {gt_y[0][ans_idx] == 1.}")

print(f"{answer_cnt}/{val_num}")