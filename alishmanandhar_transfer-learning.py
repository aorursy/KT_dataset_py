import time
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard

import numpy as np
import PIL
import os
import matplotlib.pyplot as plt
def set_trainable(model, flag = False):
    for layer in model.layers:
        layer.trainable = flag
def path_join(dirname, filenames):
    return[os.path.join(dirname,filename) for filename in filenames]
!ls ../input/final/final/
batch_size = 20
num_classes = 3
epochs = 50
train_dir = '../input/final/final/train'
val_dir = '../input/final/final/valid'
input_shape = (300,300)
data_gen_train = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=180, zoom_range=[0.9,1.3])
data_gen_val = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
generator_train = data_gen_train.flow_from_directory(directory=train_dir, target_size=input_shape, batch_size=70, shuffle=True)
generator_val = data_gen_val.flow_from_directory(directory=val_dir, target_size=input_shape, batch_size=10, shuffle=True)
pre_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(300,300,3))
pre_model.summary()
last_pre_layers = ['block5_pool']
for pre_layer in last_pre_layers:
    
    pre_layer_output = pre_model.get_layer(pre_layer)
    ref_model = keras.models.Model(inputs=pre_model.input, outputs=pre_layer_output.output)
    
    set_trainable(ref_model, flag=False)
    
    dense_values = [1024]
    
    for dense_val in dense_values:
        
        NAME = "x-lr-05-ep-200-retro-pre_layer-{}-dense-{}-time-{}".format(pre_layer, dense_val, int(time.time()))
        tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
#         transfer_model.add(tf.layers.batch_normalization(input=ref_model))
        transfer_model = keras.models.Sequential()
        transfer_model.add(ref_model)
        transfer_model.add(keras.layers.Flatten())
        transfer_model.add(keras.layers.Dense(1024, activation='relu'))
        transfer_model.add(keras.layers.Dense(512, activation='relu'))
        transfer_model.add(keras.layers.Dense(256, activation='relu'))
        transfer_model.add(keras.layers.Dense(128, activation='relu'))
        transfer_model.add(keras.layers.Dense(64, activation='relu'))
        transfer_model.add(keras.layers.Dense(3, activation='softmax'))
        optimizer = keras.optimizers.Adam(lr=0.00001)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        transfer_model.compile(optimizer=optimizer, loss = loss, metrics=metrics)

        history = transfer_model.fit_generator(generator=generator_train, epochs=epochs, steps_per_epoch=5, validation_data=generator_val, validation_steps=3)
#         transfer_model.fit(train_dataset, epochs=200, steps_per_epoch=10, validation_data=val_dataset, validation_steps=10, callbacks=[tensorboard])
#         keras.models.save_model(transfer_model, filepath='models/{}.model'.format(NAME))
        
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
