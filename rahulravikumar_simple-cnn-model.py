import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import os

import matplotlib.pyplot as plt
PATH ='../input/chest-xray-pneumonia/chest_xray'
train_dir = os.path.join(PATH, 'train')

validation_dir = os.path.join(PATH, 'val')

test_dir=os.path.join(PATH,'test')
train_normal_dir = os.path.join(train_dir, 'NORMAL') 

train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA') 

validation_normal_dir = os.path.join(validation_dir, 'NORMAL')  

validation_pneumonia_dir = os.path.join(validation_dir, 'PNEUMONIA') 

test_normal_dir = os.path.join(test_dir, 'NORMAL')

test_pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')
num_normal_tr = len(os.listdir(train_normal_dir))

num_pneumonia_tr = len(os.listdir(train_pneumonia_dir))



num_normal_val = len(os.listdir(validation_normal_dir))

num_pneumonia_val = len(os.listdir(validation_pneumonia_dir))



total_train = num_normal_tr + num_pneumonia_tr

total_val = num_normal_val + num_pneumonia_val
print('total training normal images:', num_normal_tr)

print('total training pneumonia images:', num_pneumonia_tr)



print('total validation normal images:', num_normal_val)

print('total validation pneumonia images:', num_pneumonia_val)

print("--")

print("Total training images:", total_train)

print("Total validation images:", total_val)
batch_size = 16

epochs = 20

IMG_HEIGHT = 150

IMG_WIDTH = 150
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

                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                     class_mode='binary')
image_gen_val = ImageDataGenerator(rescale=1./255)



val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,

                                                 directory=validation_dir,

                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                 class_mode='binary')
# Generator for our validation data

test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data



test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,

                                                              directory=test_dir,

                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                              class_mode='binary')
model = Sequential([

    Conv2D(32, 3, padding='same', activation='relu', 

           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Dropout(0.2),

    Conv2D(128, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(256, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Dropout(0.2),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(1)

])
model.compile(optimizer='adam',

                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

                  metrics=['accuracy'])



model.summary()
history = model.fit(

    train_data_gen,

    steps_per_epoch=total_train // batch_size,

    epochs=epochs,

    validation_data=val_data_gen,

    validation_steps=total_val // batch_size

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



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
print("Loss of the model is - " , model.evaluate(test_data_gen)[0])

print("Accuracy of the model is - " , model.evaluate(test_data_gen)[1]*100 , "%")