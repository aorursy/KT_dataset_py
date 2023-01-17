import os



train_crazing_dir = os.path.join('../input/neu-surface-defect-database/NEU-DET/train/images/crazing')

train_inclusion_dir = os.path.join('../input/neu-surface-defect-database/NEU-DET/train/images/inclusion/')

train_patches_dir = os.path.join('../input/neu-surface-defect-database/NEU-DET/train/images/patches/')

train_pitted_surface_dir = os.path.join('../input/neu-surface-defect-database/NEU-DET/train/images/pitted_surface/')

train_rolledin_scale_dir = os.path.join('../input/neu-surface-defect-database/NEU-DET/train/images/rolled-in_scale/')

train_scratches_dir = os.path.join('../input/neu-surface-defect-database/NEU-DET/train/images/scratches/')
train_crazing_names = os.listdir(train_crazing_dir)

train_inclusion_names = os.listdir(train_inclusion_dir)

train_patches_names = os.listdir(train_patches_dir)

train_pitted_surface_names = os.listdir(train_pitted_surface_dir)

train_rolledin_scale_names = os.listdir(train_rolledin_scale_dir)

train_scratches_names = os.listdir(train_scratches_dir)
print('total training crazing images:', len(train_crazing_names))

print('total training inclusion images:', len(train_inclusion_names))

print('total training patches images:', len(train_patches_names))

print('total training pitted_surface images:', len(train_pitted_surface_names))

print('total training rolled_in_scale images:', len(train_rolledin_scale_names))

print('total training scratches images:', len(train_scratches_names))
import tensorflow as tf
model = tf.keras.models.Sequential([

                                    # First Convolution

                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),

                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    # Second Convolution

                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    # Third Convolution

                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(512, activation='relu'),

                                    tf.keras.layers.Dense(512, activation='relu'),

                                    tf.keras.layers.Dense(6, activation='softmax'),





                            



])
model.summary()
from tensorflow.keras.optimizers import RMSprop, Adam



model.compile(loss='categorical_crossentropy',

              optimizer= 'adam',

              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Rescaling images

train_datagen = ImageDataGenerator(rescale=1/255)



train_generator = train_datagen.flow_from_directory(

    directory = '../input/neu-surface-defect-database/NEU-DET/train/images/',

    target_size = (224, 224),

    batch_size = 60,

    class_mode = 'categorical'

)
# Load the TensorBoard notebook extension

import datetime

%load_ext tensorboard

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



history = model.fit(

    train_generator,

    steps_per_epoch=8,

    epochs=15,

    verbose=1,

    callbacks=[tensorboard_callback]

)

%tensorboard --logdir logs/fit
