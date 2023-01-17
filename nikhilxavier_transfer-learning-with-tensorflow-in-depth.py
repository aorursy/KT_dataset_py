import os

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf



from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator
PATH = tf.keras.utils.get_file(

  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',extract=True, untar=True)
os.listdir(PATH)
train_dir = os.path.join(PATH)

validation_dir = os.path.join(PATH)



total_train = len(os.listdir(train_dir))

total_val = len(os.listdir(validation_dir))



print("Total train  class:", total_train)

print("Total validation class:", total_val)
batch_size = 32

epochs = 10

IMG_HEIGHT = 150

IMG_WIDTH = 150
# Train data set

image_gen_train = ImageDataGenerator(

                    rescale=1./255,

                    rotation_range=45,

                    width_shift_range=.15,

                    height_shift_range=.15,

                    horizontal_flip=True,

                    zoom_range=0.5

                    )



train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,

                                                     directory=train_dir,

                                                     shuffle=True,

                                                     target_size=(IMG_HEIGHT, IMG_WIDTH))
# Validation data set

image_gen_val = ImageDataGenerator(rescale=1./255)



val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,

                                                 directory=validation_dir,

                                                 target_size=(IMG_HEIGHT, IMG_WIDTH))
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
sample_training_images, _ = next(train_data_gen)

plotImages(sample_training_images[:5])
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)



base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,

                                               include_top=False,

                                               weights='imagenet')
feature_batch = base_model(train_data_gen.next())

print(feature_batch.shape)
base_model.trainable = False

base_model.summary()
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

feature_batch_average = global_average_layer(feature_batch)

print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(train_data_gen.num_classes, activation='softmax')

prediction_batch = prediction_layer(feature_batch_average)

print(prediction_batch.shape)
model = tf.keras.Sequential([

  base_model,

  global_average_layer,

  prediction_layer

])
result_batch = model.predict(sample_training_images)

result_batch.shape
base_learning_rate = 0.001

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=train_data_gen.samples // train_data_gen.batch_size,

    epochs=epochs,

    validation_data=val_data_gen,

    validation_steps=val_data_gen.samples // val_data_gen.batch_size

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()),1])

plt.xlabel('epoch')

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.tight_layout()

plt.show()
base_model.trainable = True

for layer in base_model.layers[:100]:

  layer.trainable =  False
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
fine_tune_epochs = 10

total_epochs =  epochs + fine_tune_epochs



history_fine = model.fit_generator(

    train_data_gen,

    steps_per_epoch=train_data_gen.samples // train_data_gen.batch_size,

    epochs=total_epochs,

    initial_epoch = epochs,

    validation_data=val_data_gen,

    validation_steps=val_data_gen.samples // val_data_gen.batch_size

)
acc = acc + history_fine.history['accuracy']

val_acc = val_acc + history_fine.history['val_accuracy']

loss = loss + history_fine.history['loss']

val_loss = val_loss + history_fine.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.plot([epochs-1,epochs-1],

          plt.ylim(), label='Start Fine Tuning')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.xlabel('epoch')

plt.ylim([min(plt.ylim()),1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.plot([epochs-1,epochs-1],

          plt.ylim(), label='Start Fine Tuning')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.tight_layout()

plt.show()