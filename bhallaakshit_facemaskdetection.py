import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import json
import os
import random

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

import tensorflow as tf
print("Tensorflow version:", tf.__version__)
path = "../input/face-mask-dataset/DATASET"
labels = ["Mask", "NoMask"]
# Randomly view 5 images in each category

num = 10
fig, axs = plt.subplots(len(labels), num, figsize = (15, 15))

class_len = {}
for i, c in enumerate(labels):
    class_path = os.path.join(path, c)
    all_images = os.listdir(class_path)
    sample_images = random.sample(all_images, num)
    class_len[c] = len(all_images)
    
    for j, image in enumerate(sample_images):
        img_path = os.path.join(class_path, image)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axs[i, j].imshow(img)
        axs[i, j].set(xlabel = c, xticks = [], yticks = [])

fig.tight_layout()

# Make a pie-chart to visualize the percentage contribution of each category.
fig, ax = plt.subplots()
ax.pie(
    class_len.values(),
    labels = class_len.keys(),
    autopct = "%1.1f%%"
)
fig.show()
# The dataset is imbalance so we will have to take care of that later.
# We do not have separate folders for training and validation. 
# We need to read training and validation images from the same folder such that:
# 1. There is no data leak i.e. Training images should not appear as validation images.                 
# 2. We must be able to apply augmentation to training images but not validation images.  
# We shall adopt the following strategy:
# 1. Use the same validation_split in ImageDataGenerator for training and validation.
# 2. Use the same seed when using flow_from_directory for training and validation. 
# To veify the correctness of this approach, you can print filenames from each generator and check for overlap.

# Note that we use simple augmentation to avoid producing unsuitable images.

img_size = 100
batch = 64

datagen_train = ImageDataGenerator(
    rescale = 1./255, 
    validation_split = 0.2,
    rotation_range = 5,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    zoom_range = 0.01
)

datagen_val = ImageDataGenerator(
    rescale = 1./255, 
    validation_split = 0.2 
)    

train_generator = datagen_train.flow_from_directory(
    directory = path,
    classes = labels,
    target_size=(img_size, img_size),
    seed = 42,
    batch_size = batch, 
    shuffle = True,
    subset = 'training'
)

val_generator = datagen_val.flow_from_directory(
    directory = path,
    classes = labels,
    target_size=(img_size, img_size),
    seed = 42,
    batch_size = batch, 
    shuffle = True,
    subset = 'validation'
)
# To veify the correctness of this approach (empty set is expected)
set(val_generator.filenames).intersection(set(train_generator.filenames))
# Check out labeling
LabelMap = val_generator.class_indices
LabelMap
# Previously we found that there was class imbalance. 
# We shall use class weights to tackle this before moving to training.

total_wt = sum(class_len.values())

weights = {
    0: 1 - class_len[labels[0]]/total_wt,
    1: 1 - class_len[labels[1]]/total_wt
}
weights
# Initialising the CNN

# Use base model
basemodel = MobileNetV2(
    weights="imagenet", 
    include_top=False,
    input_tensor=Input(shape=(img_size, img_size, 3))
)

# Fine tuning
basemodel.trainable = True

# Add classification head to the model
headmodel = basemodel.output
headmodel = GlobalAveragePooling2D()(headmodel)
headmodel = Flatten()(headmodel) 
headmodel = Dense(256, activation = "relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(128, activation = "relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(len(labels), activation = "softmax")(headmodel) 

model = Model(inputs = basemodel.input, outputs = headmodel)

model.compile(
    optimizer=Adam(lr=0.0005), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
model.summary()
%%time

epochs = 32
steps_per_epoch = train_generator.n//train_generator.batch_size
val_steps = val_generator.n//val_generator.batch_size

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1,
    patience=2, 
    min_lr=0.00001, 
    mode='auto'
)

checkpoint = ModelCheckpoint(
    "model_weights.h5", 
    monitor='val_accuracy',
    save_weights_only=True, 
    mode='max', 
    verbose=1
)

callbacks = [checkpoint, reduce_lr]

history = model.fit_generator(
    train_generator,
    validation_data = val_generator,
    validation_steps = val_steps,
    class_weight = weights,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    callbacks = callbacks
)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# Plotting training and validation accuracy per epoch

train_acc = history.history["accuracy"]
valid_acc = history.history["val_accuracy"]

epochs = range(len(train_acc)) 

plt.plot(epochs, train_acc)
plt.plot(epochs, valid_acc)
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.title("Training and Validation Accuracy")
# Plotting training and validation loss per epoch

train_loss = history.history["loss"]
valid_loss = history.history["val_loss"]

epochs = range(len(train_loss)) 

plt.plot(epochs, train_loss)
plt.plot(epochs, valid_loss)
plt.legend(["Training Loss", "Validation Loss"])
plt.title("Training and Validation Loss")
# Confusion Matrix 

# Since we do not have a lot of data, we did not split into training-validation-testing.
# Instead we split into training-validation.
# Strictly speaking, we should verify performance against new images from testing dataset.
# However, we shall use images in validation dataset for testing. 

# There is one problem. Previously, we set shuffle = True in our generator.
# This makes it difficult to obtain predictions and their corresponding ground truth labels.
# Thus, we shall call the generator again, but this time set shuffle = False.

val_generator = datagen_val.flow_from_directory(
    directory = path,
    classes = labels,
    target_size=(img_size, img_size),
    seed = 42,
    batch_size = batch, 
    shuffle = False,
    subset = 'validation'
)

# Obtain actual labels
actual = val_generator.classes

# Obtain predictions
pred = model.predict_generator(val_generator) # Gives class probabilities
pred = np.round(pred) # Gives one-hot encoded classes
pred = np.argmax(pred, axis = 1) # Gives class labels
    
# Now plot matrix
cm = confusion_matrix(actual, pred, labels = list(LabelMap.values()))
sns.heatmap(
    cm, 
    cmap="Blues",
    annot = True, 
    fmt = "d"
)
plt.show()
# Classification Report
print(classification_report(actual, pred))
# Store 10 images with their predicted and actual lables
img_list = []
title_list = []
for i in range(10):    
    X, y = next(val_generator)
    image = X[0]
    actual = labels[np.argmax(y[0])]
    
    img = image.reshape(1, img_size, img_size, 3) # Simple pre-processing. Sometimes it can be more complex than this.
    predict = model.predict(img)
    predict = np.argmax(predict)
    predict = labels[predict]
        
    img_list.append(img)
    title_list.append("Actual: " + actual + " , Predicted: " + predict)
# Plot images with their predicted and actual lables
img_list = np.array(img_list)
title_list = np.array(title_list)

img_list = img_list.reshape(2, 5, 100, 100, 3)
title_list = title_list.reshape(2, 5)

fig, axs = plt.subplots(2, 5, figsize = (20, 20))
for i in range(2):
    for j in range(5):
        axs[i, j].imshow(img_list[i, j])
        axs[i, j].set(xlabel = title_list[i, j], xticks = [], yticks = [])
        
fig.tight_layout()
