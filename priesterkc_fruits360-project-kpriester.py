import tensorflow as tf
from tensorflow.keras import backend, models, layers, optimizers, regularizers
from tensorflow.keras.layers import BatchNormalization, Input, Concatenate, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

import pandas as pd
import numpy as np
np.random.seed(42)

from matplotlib import pyplot as plt

import os
# set directories for train and test datasets
base_dir = "../input/fruits/fruits-360"

train_dir = f"{base_dir}/Training"
test_dir = f"{base_dir}/Test"
# testing access to Training directory
os.listdir(train_dir)[:5]
image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

BATCH_SIZE = 50
IMG_HEIGHT = 100
IMG_WIDTH = 100
CLASS_NAMES = os.listdir(train_dir)
# create image generator with images randomized
train_data_gen = image_generator.flow_from_directory(directory=train_dir,
                                                     batch_size=BATCH_SIZE,
                                                     seed=42,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes=CLASS_NAMES,
                                                     subset='training')

valid_data_gen = image_generator.flow_from_directory(directory=train_dir,
                                                     batch_size=BATCH_SIZE,
                                                     seed=42,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes=CLASS_NAMES,
                                                     subset='validation')

test_data_gen = image_generator.flow_from_directory(directory=test_dir,
                                                    batch_size=BATCH_SIZE,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    classes=CLASS_NAMES,
                                                    shuffle=False)
# view 25 sample images from training set

image_batch, label_batch = next(train_data_gen)

plt.figure(figsize=(10,10))

for n in range(25):
    ax = plt.subplot(5,5,n+1)
    plt.imshow(image_batch[n])
    plt.axis('off')
    img_title = CLASS_NAMES[list(label_batch[n]).index(1)] # get label name at index position where value in label_batch array is 1
    plt.title(img_title)
# variables for data in model
STEPS_PER_EPOCH = train_data_gen.samples // BATCH_SIZE
VALID_STEPS = valid_data_gen.samples // BATCH_SIZE
EPOCHS = 5
# clear previous tensorflow model graphs
backend.clear_session()

# resolution size of each image
x_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

# build model structure
model = models.Sequential()

# input layer
model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=x_shape))
# hidden layers
model.add(layers.MaxPool2D())
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
#ouput layer
model.add(layers.Dense(len(CLASS_NAMES), activation='softmax'))

# show model architecture
model.summary()

# compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# create Early Stopping
callback = EarlyStopping(monitor='val_accuracy',
                         patience=2,
                         restore_best_weights=True)

# train model
history = model.fit_generator(train_data_gen,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              epochs=EPOCHS,
                              validation_data=valid_data_gen,
                              validation_steps=VALID_STEPS,
                              callbacks=[callback],
                              verbose=2)

# display test accuracy & loss
test_loss, test_acc = model.evaluate_generator(test_data_gen, steps=50)
print(f"Model test accuracy: {test_acc}")
print(f"Model test loss: {test_loss}")
# table of loss and accuracy results throughout training
hist = pd.DataFrame(history.history)

# store column values into separate variables
epochs = range(1, len(hist) + 1)
loss = hist['loss']
valid_loss = hist['val_loss']
accuracy = hist['accuracy']
valid_accuracy = hist['val_accuracy']

# Training Loss plot
plt.plot(epochs, loss, 'ko', label="Training loss")
plt.plot(epochs, valid_loss, 'b', label="Validation loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Training Accuracy plot
plt.plot(epochs, accuracy, 'ko', label="Training accuracy")
plt.plot(epochs, valid_accuracy, 'b', label="Validation accuracy")

plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

backend.clear_session()

# load in pretrained model
model_base = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# freeze model weights
model_base.trainable = False

# view base model architecture
model_base.summary()

# add top layers and output
full_model = models.Sequential()
full_model.add(model_base)
full_model.add(layers.Flatten())
full_model.add(layers.Dense(512, activation='relu'))
full_model.add(layers.Dense(len(CLASS_NAMES), activation='softmax'))


# view model architecture
print(full_model.summary())

# compile model
full_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# create Early Stopping
callback = EarlyStopping(monitor='val_accuracy',
                         patience=2,
                         restore_best_weights=True)
# train model
history = full_model.fit(train_data_gen,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=EPOCHS,
                    validation_data=valid_data_gen,
                    validation_steps=VALID_STEPS,)

# display test accuracy & loss
test_loss, test_acc = full_model.evaluate_generator(test_data_gen, steps=50)
print(f"Model test accuracy: {test_acc}")
print(f"Model test loss: {test_loss}")
# table of loss and accuracy results throughout training
hist = pd.DataFrame(history.history)

# store column values into separate variables
epochs = range(1, len(hist) + 1)
loss = hist['loss']
valid_loss = hist['val_loss']
accuracy = hist['accuracy']
valid_accuracy = hist['val_accuracy']

# Training Loss plot
plt.plot(epochs, loss, 'ko', label="Training loss")
plt.plot(epochs, valid_loss, 'b', label="Validation loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Training Accuracy plot
plt.plot(epochs, accuracy, 'ko', label="Training accuracy")
plt.plot(epochs, valid_accuracy, 'b', label="Validation accuracy")

plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
