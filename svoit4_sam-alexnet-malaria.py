import os
import numpy as np
from shutil import copyfile
from random import seed, sample

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks.callbacks import EarlyStopping
from keras.optimizers import Adam
# create train, test, validation directory structure
# include directories for images that are too small
cell_types = ["Parasitized/", "Uninfected/"]
data_set_type = ["train", "test", "validation"]

for ds in data_set_type:
    # skip if directory exists
    if ds in os.listdir("./cell_images/"):
        continue
    os.mkdir("./cell_images/" + ds + "/")
    for ct in cell_types:
        os.mkdir("./cell_images/" + ds + "/" + ct)
# get two lists of all image files
p_path = "./cell_images/Parasitized"
u_path = "./cell_images/Uninfected"

# remove database files from list if present
parasitized = [i for i in os.listdir(p_path) if i != "Thumbs.db"]
uninfected = [i for i in os.listdir(u_path) if i != "Thumbs.db"]

n_parasitized = len(parasitized)
n_uninfected = len(uninfected)

# train, test, validation split
train_frac = 0.6
test_frac = 0.2
valid_frac = 1 - train_frac - test_frac
train_size = int(n_parasitized * train_frac)
test_size = int(n_parasitized * test_frac)
valid_size = n_parasitized - train_size - test_size
### randomly sample the image lists for assignment to train, test, or validation ###
# set random seed
seed(1)

# sample the training set
p_train_samp = sample(range(n_parasitized), train_size)
u_train_samp = sample(range(n_uninfected), train_size)
train_parasit = []
train_uninf = []

# generate new lists of training images
for s in p_train_samp:
    train_parasit.append(parasitized[s])
for s in u_train_samp:
    train_uninf.append(uninfected[s])

# generate new sample list for test set
parasitized = [i for i in parasitized if i not in train_parasit]
uninfected = [i for i in uninfected if i not in train_uninf]
n_parasitized = len(parasitized)
n_uninfected = len(uninfected)

p_test_samp = sample(range(n_parasitized), test_size)
u_test_samp = sample(range(n_uninfected), test_size)
test_parasit = []
test_uninf = []

# generate new lists of test images
for s in p_test_samp:
    test_parasit.append(parasitized[s])
for s in u_test_samp:
    test_uninf.append(uninfected[s])

# remaining values will be used as validation set
valid_parasit = [i for i in parasitized if i not in test_parasit]
valid_uninf = [i for i in uninfected if i not in test_uninf]
### copy files to the appropriate directory ###

# train
for p in train_parasit:
    copyfile("./cell_images/Parasitized/" + p, "./cell_images/train/Parasitized/" + p)
for u in train_uninf:
    copyfile("./cell_images/Uninfected/" + u, "./cell_images/train/Uninfected/" + u)

# test
for p in test_parasit:
    copyfile("./cell_images/Parasitized/" + p, "./cell_images/test/Parasitized/" + p)
for u in test_uninf:
    copyfile("./cell_images/Uninfected/" + u, "./cell_images/test/Uninfected/" + u)

# validation
for p in valid_parasit:
    copyfile("./cell_images/Parasitized/" + p, "./cell_images/validation/Parasitized/" + p)
for u in valid_uninf:
    copyfile("./cell_images/Uninfected/" + u, "./cell_images/validation/Uninfected/" + u)
# train batch size and image dims
batch = 32
inp_height = 224
inp_width = 224

input_shape = (inp_height, inp_width, 3)

# data generator/ preprocessor
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    validation_split=0.0
)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    fill_mode='nearest',
    data_format='channels_last',
    validation_split=0.0,
    dtype='float64')

test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
# uniform key word arguments passed to each iterator
ffd_kwargs = {"target_size" : (inp_height, inp_width),
              "color_mode" : "rgb",
              "class_mode" : "binary",
              "batch_size" : batch,
              "shuffle" : True,
              "seed" : 1,
              "subset" : None,
              "interpolation" : 'nearest'}
# specify train, test, validation generators to flow from the directories we created
train_generator = train_datagen.flow_from_directory(
    "./cell_images/train",
    **ffd_kwargs)

test_generator = test_datagen.flow_from_directory(
    "./cell_images/test",
    **ffd_kwargs)

validation_generator = validation_datagen.flow_from_directory(
    "./cell_images/validation",
    **ffd_kwargs)
#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding="valid"))
model.add(Activation("relu"))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="valid"))
model.add(Activation("relu"))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
model.add(Activation("relu"))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
model.add(Activation("relu"))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid"))
model.add(Activation("relu"))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

# Passing it to a Fully Connected layer
model.add(Flatten())

# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation("relu"))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation("relu"))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation("relu"))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.summary()
# specify early stopping conditions
es = EarlyStopping(monitor="val_binary_accuracy",
                   min_delta=0,
                   patience=15,
                   verbose=0,
                   mode='auto',
                   baseline=None,
                   restore_best_weights=True)
opt = Adam(learning_rate=1e-6, beta_1=0.9, beta_2=0.999, amsgrad=False)
# compile and train the model
epochs = 300

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=opt,
              metrics=["binary_accuracy"])

history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    shuffle=True,
    callbacks=[es])
# Plot training & validation accuracy values
plt.plot(history.history['binary_2accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
model.evaluate_generator(validation_generator)
1 / (5511 * (0.025**2) * 4)
