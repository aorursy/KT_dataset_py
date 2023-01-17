import os

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing import image

from tensorflow.keras import layers, callbacks, optimizers

import matplotlib.pyplot as plt

import numpy as np
# Define paths for train, validate, test dir

BASE_DIR = '../input/intel-image-classification'

TRAIN_DIR = os.path.join(BASE_DIR, "seg_train/seg_train")

VALIDATE_DIR = os.path.join(BASE_DIR, 'seg_test/seg_test')
LABEL_NAMES = np.array(os.listdir(TRAIN_DIR))

print(LABEL_NAMES)
# Use data augmentation for training data to prevent overfiting

SIZE = (150, 150)

BATCH_SIZE = 20



train_datagen = image.ImageDataGenerator(rescale=1/255,

                                        rotation_range=30,

                                        shear_range=0.1,

                                        zoom_range=0.1,

                                        width_shift_range=0.1,

                                        height_shift_range=0.1,

                                        horizontal_flip=True,

                                        fill_mode='reflect',)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR,

                                             target_size=SIZE,

                                             class_mode='categorical',

                                             batch_size=BATCH_SIZE)



validate_datagen = image.ImageDataGenerator(rescale=1/255)

validate_gen = validate_datagen.flow_from_directory(VALIDATE_DIR,

                                               target_size=SIZE,

                                               class_mode='categorical',

                                               batch_size=BATCH_SIZE)
# See if data is imbalanced

for label in os.listdir(TRAIN_DIR):

    images_dir = os.path.join(TRAIN_DIR, label)

    count = len(os.listdir(images_dir))



    print(label)

    print(f'Train: {count}')

    print()
# Preview images from each classes

i = 1

fig = plt.figure(figsize=((10, 10)))

for cls in os.listdir(TRAIN_DIR):

    path = os.path.join(TRAIN_DIR, cls)

    img_path = os.listdir(path)[0]

    img = plt.imread(os.path.join(path, img_path))

    fig.add_subplot(2, 3, i)

    plt.imshow(img)

    plt.xlabel(cls)

    plt.xticks([])

    plt.yticks([])

    i+=1
# Review image after augmentation

im_batch, label_batch = train_gen.next()

fig = plt.figure(figsize=(15, 15))

i = 1

row = 5

col = int(np.ceil(BATCH_SIZE / 5))

for im, label in zip(im_batch, label_batch):

    fig.add_subplot(row, col, i)

    plt.xticks([])

    plt.yticks([])

    plt.xlabel(LABEL_NAMES[np.argmax(label)])

    plt.imshow(im)

    i += 1
# Load pretrained model

conv_base = keras.applications.VGG16(include_top=False,

                                         weights='imagenet', 

                                         input_shape=SIZE+(3,))
# Freeze the pretrained model

conv_base.trainable = False
# Add classifier layers on top of feature extractor

model = keras.Sequential([

    conv_base,

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(512, activation='relu'),

    layers.Dense(512, activation='relu'),

    layers.Dense(len(LABEL_NAMES), activation='softmax')

])

model.summary()
# Configure model

model.compile(optimizer='adam', 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])



first_train_path = 'trained_top_layer.h5'

checkpoint = callbacks.ModelCheckpoint(first_train_path,

                                       monitor='val_loss',

                                       mode='min',

                                       save_best_only=True)

CALLBACKS = [checkpoint]
# Train with new data

hist = model.fit(train_gen,

                epochs=15,

                callbacks=CALLBACKS,

                validation_data=validate_gen)
# Create function to visualize loss and accuracy of training and validation

def plot_hist():

    loss = hist.history['loss']

    val_loss = hist.history['val_loss']

    acc = hist.history['accuracy']

    val_acc = hist.history['val_accuracy']

    epochs = range(len(loss))

    

    # PLot loss and accuracy for tuning

    plt.plot(epochs, loss, 'b', label='Training Loss')

    plt.plot(epochs, val_loss, 'r', label='Validation Loss')

    plt.title('Training and Validation Loss')

    plt.legend()

    plt.show()

    

    plt.plot(epochs, acc, 'b', label='Training Accuracy')

    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')

    plt.title('Training and Validation Accuracy')

    plt.legend()

    plt.show()
plot_hist()
# Load previous model with the lowest validation loss

model = keras.models.load_model(first_train_path)
# Unfreeze all layers of base model

for layer in model.layers:

    if layer.name == 'vgg16':

        layer.trainable = True

        

model.summary()
# Recompile model, apply lower learning rate

model.compile(optimizer=optimizers.Adam(lr=1e-5), 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])



classifier_path = 'intel_image_classifier.h5'

checkpoint = callbacks.ModelCheckpoint(classifier_path,

                                      monitor='val_loss',

                                      mode='min',

                                      save_best_only=True)

CALLBACKS = [checkpoint]
hist = model.fit(train_gen,

                epochs=15,

                callbacks=CALLBACKS,

                validation_data=validate_gen)
plot_hist()
best_model = keras.models.load_model(classifier_path)

best_model.evaluate(validate_gen)