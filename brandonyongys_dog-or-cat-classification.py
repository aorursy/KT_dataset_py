# Import packages

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import os



import random # Randomly select a filename for viewing

from keras.preprocessing.image import load_img # View image

from PIL import Image



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator



from sklearn.model_selection import train_test_split



# Directory

main_folder = r'../input'

input_file_train = r'train/train'

input_file_test = r'test1/test1'
# Import training file

## All training images are in a folder

## Each image is labeled with cat or dog

## The categories are converted to 0 (Cat) or 1 (Dog)

filenames = os.listdir(os.path.join(main_folder, input_file_train))

categories = []

for file in filenames:

	category = file.split(".")[0].lower()

	categories.append(category)



dataframe = pd.DataFrame({"filename":filenames, "categories":categories})

# View sample data

sample = random.choice(filenames)

plt.imshow(load_img(os.path.join(main_folder, input_file_train, sample)))



from PIL import Image

im = Image.open(os.path.join(main_folder, input_file_train, sample))

im.size
# Split the dataframe into train and val dataframe

random_seed = 10

batch_size = 100

image_height, image_width, image_channels = 128, 128, 3

train_df, val_df = train_test_split(dataframe, test_size = 0.2, random_state = random_seed)

train_df.reset_index(inplace = True, drop = True)

val_df.reset_index(inplace = True, drop = True)
# ImageDataGenerator to load the images for training

## Train datagen and generator

train_datagen = ImageDataGenerator(

	rotation_range = 15,

	rescale = 1./255,

	width_shift_range = 0.2, 

	height_shift_range = 0.2, 

	horizontal_flip = True, 

	shear_range = 0.2, 

	zoom_range = 0.2)



train_generator = train_datagen.flow_from_dataframe(

	dataframe = train_df,

	directory = os.path.join(main_folder, input_file_train),

	x_col = "filename",

	y_col = "categories",

	target_size = (image_height, image_width),

	class_mode = "categorical",

	batch_size = batch_size)



## Val datagen and generator

val_datagen = ImageDataGenerator(rescale = 1./255)

val_generator = val_datagen.flow_from_dataframe(

	dataframe = val_df,

	directory = os.path.join(main_folder, input_file_train),

	x_col = "filename",

	y_col = "categories",

	target_size = (image_height, image_width),

	class_mode = "categorical",

	batch_size = batch_size)

# Model hyperparameters

dropout_rate = 0.25

fc_units_1 = 512

fc_units_2 = 256

output_units = 2

epochs = 50



# Build the CNN model

model = Sequential()



## Conv layer 1

model.add(Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same",

	input_shape = (image_height, image_width, image_channels), data_format = "channels_last"))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(rate = dropout_rate))



## Conv layer 2

model.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same"))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(rate = dropout_rate))



## Conv layer 3

model.add(Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same"))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(rate = dropout_rate))



## Output layer

model.add(Flatten())

model.add(Dense(units = fc_units_1, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(rate = dropout_rate))

model.add(Dense(units = fc_units_2, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(rate = dropout_rate))

model.add(Dense(units = output_units, activation = "softmax"))
# Model summary

model.summary()

# Optimize and compile

optimizer = RMSprop()

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
# Create callbacks

## Early Stop

### Stop training if the val_loss does not decrease 

earlystop = EarlyStopping(monitor = "val_acc", patience = 10)



## Reduce learning rate

reduce_lr = ReduceLROnPlateau(monitor = "val_acc", factor = 0.75, verbose = 1, patience = 2, min_lr = 0.00001)



## Save best model

checkpoint = ModelCheckpoint(filepath = os.path.join(main_folder, "best weights.h5"), monitor = "val_acc", save_best_only = True)



callbacks = [earlystop, reduce_lr]
# Train the model

history = model.fit_generator(

	generator = train_generator, 

	steps_per_epoch = len(train_df)//batch_size,

	epochs = epochs,

	callbacks = callbacks,

	validation_data = val_generator, 

	validation_steps = len(val_df)//batch_size)
model.save_weights("model.h5")
# Visualize the training and validation loss and accuracy

fig, ax = plt.subplots(2, 1, figsize = (12,12))

ax[0].plot(history.history["loss"], color = "b", label = "Training loss")

ax[0].plot(history.history["val_loss"], color = "r", label = "Validation loss")

ax[0].set_xticks(np.arange(1, epochs, 1))

ax[0].set_yticks(np.arange(0, 2.3, 0.2))

ax[0].legend(loc = "best", shadow = True)



ax[1].plot(history.history["acc"], color = "b", label = "Training accuracy")

ax[1].plot(history.history["val_acc"], color = "r", label = "Valication accuracy")

ax[1].set_xticks(np.arange(1, epochs))

ax[1].legend(loc = "best", shadow = True)



plt.tight_layout()

plt.show()
##### TEST SET

# Read and set up the generator

test_files = os.listdir(os.path.join(main_folder, input_file_test))

test_df = pd.DataFrame({"filename":test_files})

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_dataframe(

	dataframe = test_df,

	directory = os.path.join(main_folder, input_file_test),

	x_col = "filename",

	y_col = None,

	target_size = (image_height, image_width),

	class_mode = None,

	batch_size = batch_size,

	shuffle = False)
# Prediction on the test set

prediction = model.predict_generator(

	generator = test_generator,

	steps = np.ceil(len(test_df)/batch_size))
# Convert from probability to label (0 or 1)

test_df["category"] = np.argmax(prediction, axis = 1)
# Save prediction to csv for submission

submission = test_df.copy()

submission["filename"] = submission["filename"].apply(lambda x: x.split(".")[0])

submission.rename(columns = {"filename":"id", "category":"label"}, inplace = True)

submission.to_csv("submission.csv", index = False)