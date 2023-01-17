# Import libraries and tools
# Data preprocessing and linear algebra
import os, re, random
from os.path import join
import zipfile
from pathlib import Path
import shutil
from sklearn.datasets import load_files
import pandas as pd
import numpy as np
np.random.seed(2)

# Visualisation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

# Tools for cross-validation, error calculation
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical

# Machine Learning
from keras.models import Model
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
print(os.listdir('../input/'))
INPUT_PATH = '../input/flowers-recognition/flowers/flowers/'
print(os.listdir(INPUT_PATH))
img_folders = [join(INPUT_PATH, dir) for dir in os.listdir(INPUT_PATH)]
list(img_folders)
# Load images into NumPy array
images = load_files(INPUT_PATH, random_state=42, shuffle=True)
X = np.array(images['filenames'])
y = np.array(images['target'])
labels = np.array(images['target_names'])

# Remove unnecessary .pyc and .py files
pyc_file = (np.where(file==X) for file in X if file.endswith(('.pyc','.py')))
for i in pyc_file:
    X = np.delete(X, i)
    y = np.delete(y, i)
# Our array summary
print(f'Target labels (digits) - {y}')
print(f'Target labels (names) - {labels}')
print(f'Number of uploaded images : {X.shape[0]}')
# Draw random image directly from dataset for aesthetic reasons only
img = plt.imread('../input/flowers-recognition/flowers/daisy/100080576_f52e8ee070_n.jpg')
plt.imshow(img);
# Check our target y variable
flowers = pd.DataFrame({'species': y})
flowers.count()
# Correspond species and flowers and form digit labels
flowers['flower'] = flowers['species'].astype('category')
labels = flowers['flower'].cat.categories
labels
# Let's implement a constant - standard image size for Inception model input, which is 150 px
image_size = 150
# Write images into NumPy array using sklearn's img_to_array() method
def imageLoadConverter(img_paths):
    # Load
    images = [load_img(img_path, target_size=(image_size, image_size)) for img_path in img_paths]
    # Write into array
    images_array = np.array([img_to_array(img) for img in images])
    
    return(images_array)

# Convert into NumPy array
X = np.array(imageLoadConverter(X))
# Print result
print(f'Function worked with following output (images, width, height, color): {X.shape}')
# Convert classes in digit form
num_classes = len(np.unique(y))
print(f'Classes: {num_classes} and corresponding labels: {labels}')
# One-Hot Encoding
y = to_categorical(y, num_classes)
print(y.shape)
# Split data on train, validation and test subsets
# Using 10% or 20% from train data is classical approach

# First, split X into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=2)

# Second, split test into test and validation subsets in equal proportion
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=2)
# Count number of elements in subsets
total_X_train = X_train.shape[0]
total_X_val = X_val.shape[0]
total_X_test = X_test.shape[0]
print(f'Train: {total_X_train}')
print(f'Validation: {total_X_val}')
print(f'Test: {total_X_test}')
# Delete X since it will not be needed further
del X
# By default, the InceptionV3 model expects images as input with the size 150x150 px with 3 channels
input_shape = (image_size, image_size, 3)
# Define model constants
batch_size = 8
epochs = 20
# Define our pre-trained model, downloading weights from Imagenet
# pre_trained_model = InceptionV3(input_shape = input_shape, include_top = False, weights = 'imagenet')

# Define our pre-trained model, using weights, uploaded from Kaggle's Keras Inception dataset
local_weights = "../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
pre_trained_model = InceptionV3(input_shape = input_shape, include_top = False, weights = None)
# Load weights into network
pre_trained_model.load_weights(local_weights)
# Print models summary table
print(pre_trained_model.summary())
# Print number of models layers
len(pre_trained_model.layers)
# Set layers to be not trainable since they are already are
for layer in pre_trained_model.layers:
     layer.trainable = False
# Add custom layers
x = pre_trained_model.output
# Add Pooling layer
x = Flatten()(x)
# Add a fully connected layer with 1024 nodes and ReLU activation
x = Dense(1024, activation="relu")(x)
# Add a dropout with rate 0.5
x = Dropout(0.2)(x)
# Specify final output layer with SoftMax activation
predictions = Dense(5, activation="softmax")(x)
pre_trained_model.input
predictions
# Build the final model 
inception_model = Model(inputs=pre_trained_model.input, 
                        outputs=predictions
                       )
# Compile model
inception_model.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                        metrics=['accuracy']
                       )
# Implement train ImageDataGenerator and specify some preprocessing
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)
# Upload and peprocess images
train_generator = train_datagen.flow(
        X_train, y_train, 
        batch_size=batch_size,
        shuffle=False)  
# Implement validation ImageDataGenerator
validation_datagen = ImageDataGenerator(
    rescale=1./255
)
validation_generator = validation_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False) 
test_datagen = ImageDataGenerator(
    rescale=1./255
)
test_generator = test_datagen.flow(
        X_test, y_test,
        batch_size=batch_size,
        shuffle=False
)
# Stop model learning after 10 epochs in which val_loss value not decreased
early_stop = EarlyStopping(patience=10, 
                          verbose=1, 
                          mode='auto'
                         )
# Reduce the learning rate when accuracy, for example, not increase for two continuous steps
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001
                                           )
# Save callbacks
callbacks = [early_stop, learning_rate_reduction]
callbacks
hist = inception_model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_X_val//batch_size,
    steps_per_epoch=total_X_train//batch_size,
    callbacks=callbacks
)
# poch 1/20
# 432/432 [==============================] - 25s 57ms/step - loss: 0.9134 - accuracy: 0.6843 - val_loss: 0.7519 - val_accuracy: 0.7616 - lr: 1.0000e-04
# Epoch 2/20
# 432/432 [==============================] - 23s 53ms/step - loss: 0.6030 - accuracy: 0.7841 - val_loss: 0.6113 - val_accuracy: 0.7801 - lr: 1.0000e-04
# Epoch 3/20
# 432/432 [==============================] - 23s 54ms/step - loss: 0.5084 - accuracy: 0.8154 - val_loss: 0.5370 - val_accuracy: 0.8079 - lr: 1.0000e-04
# Epoch 4/20
# 432/432 [==============================] - 22s 52ms/step - loss: 0.4568 - accuracy: 0.8336 - val_loss: 0.5346 - val_accuracy: 0.8009 - lr: 1.0000e-04
# Epoch 5/20
# 432/432 [==============================] - 24s 54ms/step - loss: 0.4375 - accuracy: 0.8435 - val_loss: 0.5041 - val_accuracy: 0.8148 - lr: 1.0000e-04
# Epoch 6/20
# 432/432 [==============================] - 22s 52ms/step - loss: 0.3996 - accuracy: 0.8528 - val_loss: 0.4885 - val_accuracy: 0.8218 - lr: 1.0000e-04
# Epoch 7/20
# 432/432 [==============================] - 24s 54ms/step - loss: 0.3680 - accuracy: 0.8620 - val_loss: 0.5028 - val_accuracy: 0.8194 - lr: 1.0000e-04
# Epoch 8/20
# 432/432 [==============================] - ETA: 0s - loss: 0.3690 - accuracy: 0.8667
# Epoch 00008: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.
# 432/432 [==============================] - 25s 57ms/step - loss: 0.3690 - accuracy: 0.8667 - val_loss: 0.5066 - val_accuracy: 0.8194 - lr: 1.0000e-04
# Epoch 9/20
# 432/432 [==============================] - 23s 52ms/step - loss: 0.3142 - accuracy: 0.8945 - val_loss: 0.4857 - val_accuracy: 0.8287 - lr: 5.0000e-05
# Epoch 10/20
# 432/432 [==============================] - 24s 55ms/step - loss: 0.2965 - accuracy: 0.8867 - val_loss: 0.4750 - val_accuracy: 0.8380 - lr: 5.0000e-05
# Epoch 11/20
# 432/432 [==============================] - 23s 54ms/step - loss: 0.2845 - accuracy: 0.9043 - val_loss: 0.4753 - val_accuracy: 0.8403 - lr: 5.0000e-05
# Epoch 12/20
# 432/432 [==============================] - 23s 53ms/step - loss: 0.2874 - accuracy: 0.8939 - val_loss: 0.4844 - val_accuracy: 0.8310 - lr: 5.0000e-05
# Epoch 13/20
# 431/432 [============================>.] - ETA: 0s - loss: 0.2748 - accuracy: 0.9030
# Epoch 00013: ReduceLROnPlateau reducing learning rate to 2.499999936844688e-05.
# 432/432 [==============================] - 25s 57ms/step - loss: 0.2744 - accuracy: 0.9032 - val_loss: 0.4928 - val_accuracy: 0.8356 - lr: 5.0000e-05
# Epoch 14/20
# 432/432 [==============================] - 23s 54ms/step - loss: 0.2689 - accuracy: 0.9058 - val_loss: 0.4824 - val_accuracy: 0.8333 - lr: 2.5000e-05
# Epoch 15/20
# 432/432 [==============================] - ETA: 0s - loss: 0.2675 - accuracy: 0.9064
# Epoch 00015: ReduceLROnPlateau reducing learning rate to 1.249999968422344e-05.
# 432/432 [==============================] - 24s 56ms/step - loss: 0.2675 - accuracy: 0.9064 - val_loss: 0.4757 - val_accuracy: 0.8333 - lr: 2.5000e-05
# Epoch 16/20
# 432/432 [==============================] - 23s 53ms/step - loss: 0.2451 - accuracy: 0.9130 - val_loss: 0.4748 - val_accuracy: 0.8333 - lr: 1.2500e-05
# Epoch 17/20
# 432/432 [==============================] - ETA: 0s - loss: 0.2615 - accuracy: 0.9145
# Epoch 00017: ReduceLROnPlateau reducing learning rate to 1e-05.
# 432/432 [==============================] - 23s 53ms/step - loss: 0.2615 - accuracy: 0.9145 - val_loss: 0.4769 - val_accuracy: 0.8287 - lr: 1.2500e-05
# Epoch 18/20
# 432/432 [==============================] - 24s 57ms/step - loss: 0.2516 - accuracy: 0.9107 - val_loss: 0.4685 - val_accuracy: 0.8333 - lr: 1.0000e-05
# Epoch 19/20
# 432/432 [==============================] - 24s 55ms/step - loss: 0.2439 - accuracy: 0.9186 - val_loss: 0.4724 - val_accuracy: 0.8333 - lr: 1.0000e-05
# Epoch 20/20
# 432/432 [==============================] - 24s 55ms/step - loss: 0.2454 - accuracy: 0.9145 - val_loss: 0.4674 - val_accuracy: 0.8426 - lr: 1.0000e-05
# Plot accuracy and loss curves
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))

ax1.plot(hist.history['loss'], color='r', label="Train loss")
ax1.plot(hist.history['val_loss'], color='b', label="Validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
legend = ax1.legend(loc='best', shadow=True)

ax2.plot(hist.history['accuracy'], color='r', label="Train accuracy")
ax2.plot(hist.history['val_accuracy'], color='b',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))
legend = ax2.legend(loc='best', shadow=True)

plt.tight_layout()
plt.show()
# Predict on validation X_val_resnet
y_pred_val = inception_model.predict_generator(validation_generator)
# Prepare y_true and y_pred on validation by taking the most likely class
y_true_val = y_val.argmax(axis=1)
y_pred_val = y_pred_val.argmax(axis=1)
# Check datatypes
print(f'y_true datatype: {y_true_val.dtype}')
print(f'y_pred datatype: {y_pred_val.dtype}')
# Evaluate on validation dataset
loss, acc = inception_model.evaluate_generator(validation_generator, verbose=0)
print(f'Validation loss: {loss:.2f}%')
print(f'Validation accuracy: {acc*100:.2f}%')
# Compute and plot the Confusion matrix
confusion_mtx_resnet = confusion_matrix(y_true_val, y_pred_val) 

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx_resnet, annot=True, fmt='d', cmap=plt.cm.Blues)
plt.xlabel("Predicted Label")
plt.ylabel("Validation (aka True) Label")
plt.title("Confusion Matrix")
plt.show()
samples = total_X_test
predict = inception_model.predict_generator(test_generator, steps=np.ceil(samples/batch_size))
predict.shape
X_test.shape
# Evaluate on test dataset
loss, acc = inception_model.evaluate_generator(test_generator, verbose=0)
print(f'Test loss: {loss:.2f}%')
print(f'Test accuracy: {acc*100:.2f}%')
# Get most likely class as y_pred and y_test
y_pred = predict.argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Show classification report
print(metrics.classification_report(y_true, y_pred))
# Compute and plot the Confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred) 

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True)
plt.xlabel("Predicted Label")
plt.ylabel("Validation (aka True) Label")
plt.title("Confusion Matrix")
plt.show()