# Import libraries and tools
# Data preprocessing and linear algebra
import os, re, random
import pandas as pd
import numpy as np
import zipfile
np.random.seed(2)

# Visualisation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

# Tools for cross-validation, error calculation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical

# Machine Learning
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D, Activation
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import optimizers
# from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.applications import VGG16
from keras.applications.resnet50 import ResNet50
# Save datasets path
PATH = "../input/dogs-vs-cats-redux-kernels-edition"
PATH_TRAIN = "train_images/"
PATH_TEST = "test_images/"
# Check datasets in file system
os.listdir(PATH)
# Save archives paths and names
train_image_path = os.path.join(PATH, "train.zip")
test_image_path = os.path.join(PATH, "test.zip")
# Create subfolder for train dataset
# Disclaimer. It is need because ImageDataGenerator() which we will use later correctly reads images only from subfolders
os.mkdir('/kaggle/working/train_images')
# Create subfolder for test dataset
os.mkdir('/kaggle/working/test_images')
# Unzip train dataset
archive = zipfile.ZipFile(train_image_path,"r")
for file in archive.namelist():
        archive.extract(file, 'train_images/')
# Unzip test dataset
archive = zipfile.ZipFile(test_image_path,"r")
for file in archive.namelist():
        archive.extract(file, 'test_images/')
# Check if our kittens and puppies are in right place
# os.listdir('/kaggle/working/train_images/train')
# If you prefer work with file system, unzip can be done using bash commands.
# Just leave this code here, it will give same result - unzipped images in two folders.
# Unzip train dataset
# !unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip
# Check unzipped dataset in file system
#! ls -l train/
# Unzip test dataset
# !unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip
# Save images names to variable
train_images = os.listdir(f'{PATH_TRAIN}/train/')
# Then extract images names and save them into Numpy array
imagenames = np.array([f'{f}' for f in train_images])
# Check our image names array
imagenames
# Assign labels to images according to competitions task (0-cat, 1-dog)
# Implement array of image categories
categories = []
for imagename in imagenames:
    # Loop through data and split our images names
    split_category = imagename.split('.')[0]
    # Assign labels
    if split_category == 'cat':
        categories.append(str(0))
    else:
        categories.append(str(1))
# Save our filenames 
animals = pd.DataFrame({
    'Image name': imagenames,
    'Category': categories
})
animals.head(5)
# Check total amount of 0 and 1 labels
animals['Category'].value_counts()
# Draw a cat
# Don't forget to install 'pillow' module (conda install pillow) to give a 'pyplot' ability of working with '.jpg'
img = plt.imread(f'{PATH_TRAIN}/train/{imagenames[1]}')
plt.imshow(img);
# Split data on train and validation subsets
# Using 10% or 20% from train data is classical approach
X_train, X_val = train_test_split(animals, test_size=0.2,  random_state=2)
X_train = X_train.reset_index()
X_val = X_val.reset_index()

# We may want use only 1800 images because of CPU computational reasons. If so, this code should be run
# X_train = X_train.sample(n=1800).reset_index()
# X_val = X_val.sample(n=100).reset_index()
# Count
total_X_train = X_train.shape[0]
total_X_val = X_val.shape[0]
total_X_train
total_X_val
# By default, the VGG16 model expects images as input with the size 224 x 224 pixels with 3 channels (e.g. color).
image_size = 224
input_shape = (image_size, image_size, 3)
# Define CNN model constants
epochs = 5
batch_size = 16
# Define our pre-trained model
pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
# Print models summary table
# Note that it expects input pictures in 224 size and 3 channels, as we mensioned before. So we didn't lie.
print(pre_trained_model.summary())
# Use this if want to see a Big Bang. Downloads VGG16 with total defaults (Total params: 138,357,544).
# Very huge.
# model = VGG16()
# print(model.summary())
# Add some micro-tuning 
# Set above layers to be not traianble since using pre-trained model - they are already trained
for layer in pre_trained_model.layers[:15]:
    layer.trainable = False

for layer in pre_trained_model.layers[15:]:
    layer.trainable = True

# Specify networks output    
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
    
# Flatten the output layer to one dimension
x = GlobalMaxPooling2D()(last_output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = Dropout(0.5)(x)

# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

# Form our model
model_mod = Model(pre_trained_model.input, x)
# Compile model
model_mod.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy']
                 )
# Implement train ImageDataGenerator and specify some small preprocessing
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
# Upload images from file system using flow_from_dataframe() method and use our datagen
# to make parallel preprocessing. We obtain uploaded and preprocessed images.
train_generator = train_datagen.flow_from_dataframe(
    X_train, 
    "/kaggle/working/train_images/train",
    x_col='Image name',
    y_col='Category',
    class_mode='binary',
    target_size=(image_size, image_size),
    batch_size=batch_size,
    #validate_filenames = False
)
# Implement validation ImageDataGenerator
validation_datagen = ImageDataGenerator(
    rescale=1./255
)
# Upload and peprocess images
validation_generator = validation_datagen.flow_from_dataframe(
    X_val, 
    "/kaggle/working/train_images/train",
    x_col='Image name',
    y_col='Category',
    class_mode='binary',
    target_size=(image_size, image_size),
    batch_size=batch_size,
    #validate_filenames = False
)
# Check one sample generated image
# Create generator for test sample image
generated_example_df = X_train.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    generated_example_df, 
    "/kaggle/working/train_images/train", 
    x_col='Image name',
    y_col='Category',
    class_mode='categorical',
    #validate_filenames = False
)
# Plot sample
plt.figure(figsize=(10, 10))
for i in range(0, 9):
    plt.subplot(3, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
earlystop = EarlyStopping(patience=10, 
                          verbose=1, 
                          mode='auto'
                         )
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001
                                           )
# Save our callbacks
callbacks = [earlystop, learning_rate_reduction]
callbacks
# Just leave it here
# def fixed_generator(generator):
#     for batch in generator:
#         yield (batch, batch)
# Fit the model
history = model_mod.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_X_val//batch_size,
    steps_per_epoch=total_X_train//batch_size,
    callbacks=callbacks
)
# Epoch 1/5
# 1250/1250 [==============================] - 404s 323ms/step - loss: 0.3144 - accuracy: 0.8517 - val_loss: 0.1308 - val_accuracy: 0.9451 - lr: 1.0000e-04
# Epoch 2/5
# 1250/1250 [==============================] - 385s 308ms/step - loss: 0.1555 - accuracy: 0.9362 - val_loss: 0.1066 - val_accuracy: 0.9527 - lr: 1.0000e-04
# Epoch 3/5
# 1250/1250 [==============================] - 382s 305ms/step - loss: 0.1305 - accuracy: 0.9467 - val_loss: 0.1012 - val_accuracy: 0.9599 - lr: 1.0000e-04
# Epoch 4/5
# 1250/1250 [==============================] - 381s 305ms/step - loss: 0.1125 - accuracy: 0.9553 - val_loss: 0.0866 - val_accuracy: 0.9619 - lr: 1.0000e-04
# Epoch 5/5
# 1250/1250 [==============================] - 381s 305ms/step - loss: 0.1051 - accuracy: 0.9577 - val_loss: 0.0886 - val_accuracy: 0.9649 - lr: 1.0000e-04
# Save calculated weigthts (approx. 60 Mb)
model_mod.save_weights('model_wieghts.h5')
model_mod.save('model_keras.h5')
# Plot accuracy and loss curves
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))

ax1.plot(history.history['loss'], color='r', label="Train loss")
ax1.plot(history.history['val_loss'], color='b', label="Validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
legend = ax1.legend(loc='best', shadow=True)

ax2.plot(history.history['accuracy'], color='r', label="Train accuracy")
ax2.plot(history.history['val_accuracy'], color='b',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))
legend = ax2.legend(loc='best', shadow=True)

plt.tight_layout()
plt.show()
# Prepare Y_val
Y_val = X_val['Category']
# Predict on validation data
Y_pred =  model_mod.predict_generator(validation_generator)
# Define treshold
threshold = 0.5
# Convert
Y_pred_conv = np.where(Y_pred > threshold, 1,0)
Y_pred_conv[:,0]
# Plot probability histogram
pd.Series(Y_pred_conv[:,0]).hist()
# Check datatypes
Y_pred_conv.dtype
Y_val.dtype
# Convert to int
Y_val_str = Y_val.astype(int)
# Compute and plot the Confusion matrix
confusion_mtx = confusion_matrix(Y_val_str, Y_pred_conv) 

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True)
plt.xlabel("Predicted Label")
plt.ylabel("Validation (aka True) Label")
plt.title("Confusion Matrix")
plt.show()
test_images = os.listdir('/kaggle/working/test_images/test')
X_test = pd.DataFrame({
    'test_imagename': test_images
})
samples = X_test.shape[0]
X_test.count()
# If we use only 1800 images because of CPU computational reasons:
# X_test_cpu = X_test.sample(n=1800).reset_index()
# X_test_cpu.count()
test_datagen = ImageDataGenerator(
    rescale=1./255
)
test_generator = test_datagen.flow_from_dataframe(
    X_test, 
    "/kaggle/working/test_images/test", 
    x_col='test_imagename',
    y_col=None,
    class_mode=None,
    batch_size=batch_size,
    target_size=(image_size, image_size),
    shuffle=False
)
test_generator
predict = model_mod.predict_generator(test_generator, steps=np.ceil(samples/batch_size))
predict.shape
X_test.shape
threshold = 0.5
X_test['Category'] = np.where(predict > threshold, 1,0)
# Save results using competitions format
submit = X_test.copy()
submit['id'] = submit['test_imagename'].str.split('.').str[0]
submit['label'] = submit['Category']
submit.drop(['test_imagename', 'Category'], axis=1, inplace=True)
submit.to_csv('submission_vgg16.csv', index=False)
# Check how our answer looks
plt.figure(figsize=(10,5))
sns.countplot(submit['label'])
plt.title("(Final answer on test data (Model - VGG16))")
# Define model constants
# image_size = 224 like in previous model
num_classes = 2
num_epochs = 5
num_batch_size = 64
WEIGHTS_PATH = "../input/resnet-weights-uploaded/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
# Define model
model_resnet = Sequential()
# Add pre-trained weights
model_resnet.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
# The last dense layer must specify the number of labels (or classes) and activation f-n
model_resnet.add(Dense(num_classes, activation='softmax'))
# Since we load pre-trained model we must specify first layer as non-trainable
model_resnet.layers[0].trainable = True
# Compile model
model_resnet.compile(
    optimizer='sgd', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
# Print models summary
model_resnet.summary()
# Before we begin lets save our train and validation subsets into threir copies
X_train_resnet = X_train
X_val_resnet = X_val
Y_val_resnet = Y_val
Y_train_resnet = animals['Category']
total_X_train_resnet = X_train_resnet.shape[0]
total_X_val_resnet = X_val_resnet.shape[0]
X_train_resnet.head()
total_X_train_resnet
X_train_resnet.shape
X_val_resnet.shape
train_resnet_datagen = ImageDataGenerator(
    rescale=1./255,
)
train_resnet_generator = train_resnet_datagen.flow_from_dataframe(
    X_train_resnet,
    "/kaggle/working/train_images/train",
    x_col='Image name',
    y_col='Category',
    class_mode='categorical',
    target_size=(image_size, image_size),
    batch_size=num_batch_size
)
validation_resnet_datagen = ImageDataGenerator(
    rescale=1./255
)
validation_resnet_generator = validation_resnet_datagen.flow_from_dataframe(
    X_val_resnet, 
    "/kaggle/working/train_images/train", 
    x_col='Image name',
    y_col='Category',
    class_mode='categorical',
    target_size=(image_size, image_size),
    batch_size=num_batch_size
)
# Fit the model
history_resnet = model_resnet.fit_generator(
        train_resnet_generator,
        epochs = num_epochs,
        validation_data = validation_resnet_generator,
        validation_steps = total_X_val_resnet//num_batch_size,
        steps_per_epoch = total_X_train_resnet//num_batch_size,
        callbacks = callbacks
)
# Leave this output here to outline that we obtain low accuracy when train on 5 epochs and
# first layer trainable = False param
# Epoch 1/5
# 312/312 [==============================] - 190s 611ms/step - loss: 5.4424 - accuracy: 0.5646 - val_loss: 3.1316 - val_accuracy: 0.6012 - lr: 0.0050
# Epoch 2/5
# 312/312 [==============================] - 195s 624ms/step - loss: 5.1490 - accuracy: 0.5693 - val_loss: 5.3533 - val_accuracy: 0.5655 - lr: 0.0050
# Epoch 3/5
# 312/312 [==============================] - ETA: 0s - loss: 5.5350 - accuracy: 0.5592
# Epoch 00003: ReduceLROnPlateau reducing learning rate to 0.0024999999441206455.
# 312/312 [==============================] - 194s 622ms/step - loss: 5.5350 - accuracy: 0.5592 - val_loss: 5.5042 - val_accuracy: 0.5615 - lr: 0.0050
# Epoch 4/5
# 312/312 [==============================] - 193s 617ms/step - loss: 1.8847 - accuracy: 0.6557 - val_loss: 1.7388 - val_accuracy: 0.6354 - lr: 0.0025
# Epoch 5/5
# 312/312 [==============================] - 192s 615ms/step - loss: 2.0500 - accuracy: 0.6221 - val_loss: 5.0342 - val_accuracy: 0.5294 - lr: 0.0025
# Save calculated weigthts
model_resnet.save_weights('model_resnet_wieghts.h5')
model_resnet.save('model_resnet_keras.h5')
# Plot accuracy and loss curves
fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(7, 7))
history_resnet = history

ax3.plot(history.history['loss'], color='r', label="Train loss")
ax3.plot(history.history['val_loss'], color='b', label="Validation loss")
ax3.set_xticks(np.arange(1, epochs, 1))
legend = ax3.legend(loc='best', shadow=True)

ax4.plot(history.history['accuracy'], color='r', label="Train accuracy")
ax4.plot(history.history['val_accuracy'], color='b',label="Validation accuracy")
ax4.set_xticks(np.arange(1, epochs, 1))
legend = ax4.legend(loc='best', shadow=True)

plt.tight_layout()
plt.show()
# Predict on validation X_val_resnet
Y_pred_resnet =  model_resnet.predict_generator(validation_resnet_generator)
# Define treshold
threshold = 0.5
# Convert
Y_pred_conv_res = np.where(Y_pred_resnet > threshold, 1,0)
Y_pred_conv_res[:,0]
# Plot probability histogram
pd.Series(Y_pred_conv_res[:,0]).hist()
# Prepare Y_val in str dtype
# Convert to int
Y_val_resnet_str = Y_val_resnet.astype(int)
# Compute and plot the Confusion matrix
confusion_mtx_resnet = confusion_matrix(Y_val_resnet_str, Y_pred_conv_res[:,0]) 

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx_resnet, annot=True)
plt.xlabel("Predicted Label")
plt.ylabel("Validation (aka True) Label")
plt.title("Confusion Matrix")
plt.show()
# Add X_test_resnet
X_test_resnet = X_test
test_resnet_datagen = ImageDataGenerator(
    rescale=1./255
)
test_resnet_generator = test_resnet_datagen.flow_from_dataframe(
    X_test_resnet, 
    "/kaggle/working/test_images/test", 
    x_col='test_imagename',
    y_col=None,
    class_mode=None,
    batch_size=num_batch_size,
    target_size=(image_size, image_size),
    shuffle=False
)
predict_resnet = model_resnet.predict_generator(test_resnet_generator, steps=np.ceil(samples/num_batch_size))
# Save predictions
submit_resnet = X_test_resnet.copy()
submit_resnet['id'] = submit_resnet['test_imagename'].str.split('.').str[0]
submit_resnet['label'] = submit_resnet['Category']
submit_resnet.drop(['test_imagename', 'Category'], axis=1, inplace=True)
submit_resnet.to_csv('submission_resnet_50.csv', index=False)
# Check answer
plt.figure(figsize=(10,5))
sns.countplot(submit_resnet['label'])
plt.title("(Final answer on test data (Model - ResNet))")
