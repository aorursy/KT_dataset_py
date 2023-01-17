# Import packages

import pandas as pd 

import numpy as np 

import os



from PIL import Image

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
#Directory

main_folder = "../input"

subfolder_input = r'all/All'

csv_file = r'GTruth.csv'



# Files

files = [f for f in os.listdir(os.path.join(main_folder, subfolder_input)) if "csv" not in f]

dataframe = pd.read_csv(os.path.join(main_folder, subfolder_input, csv_file))

dataframe["Id"] = dataframe["Id"].apply(lambda x: str(x) + r".jpeg")

dataframe["Ground_Truth"] = dataframe["Ground_Truth"].apply(lambda x: str(1-x)) # Change the label to 0 = normal, 1 = pneumonia
# Visualize an image

nrows = 4

ncols = 4

idx = np.random.randint(0, len(files), nrows*ncols)

fig, axes = plt.subplots(nrows, ncols, figsize = (10, 10))

for ii, ax in zip(idx, axes.flatten()):

	img = Image.open(os.path.join(main_folder, subfolder_input, files[ii]))

	ax.imshow(img)

	ax.xaxis.set_visible(False)

	ax.yaxis.set_visible(False)

	ax.title.set_text("Class label: %s" %dataframe.loc[dataframe["Id"] == files[ii], "Ground_Truth"].values)
dataframe.head()
# Display the number of cases for "normal" (1) and "pneumonia" (1)

dataframe["Ground_Truth"].value_counts().plot.bar()



"""

72% - normal (0), 28% - pneumonia (1)

There is an imbalance of classes with about 72% of the samples are normal

There is a need to balance out the classes prior to training

"""
# Split the dataframe into train, val and test set

train_set, test_set = train_test_split(dataframe, test_size = 500)

train_set, val_set  = train_test_split(train_set, test_size = 500)



print(" No of training samples: %s \n" %len(train_set),

     "No of validation samples: %s \n" %len(val_set),

     "No of test samples: %s" %len(test_set))
# Create generator for the images

batch_size = 128

image_height = 256

image_width = 256

image_channels = 3



## Train generator

train_datagen = ImageDataGenerator(

	rescale = 1./255,

	horizontal_flip = True,

	rotation_range = 15,

	width_shift_range = 0.1,

	height_shift_range = 0.1

	)



train_generator = train_datagen.flow_from_dataframe(

	dataframe = train_set,

	directory = os.path.join(main_folder, subfolder_input),

	x_col = "Id",

	y_col = "Ground_Truth",

	class_mode = "binary",

	batch_size = batch_size,

	target_size = (image_height, image_width)

	)



## Val generator

val_datagen = ImageDataGenerator(rescale = 1./255)

val_generator = val_datagen.flow_from_dataframe(

	dataframe = val_set,

	directory = os.path.join(main_folder, subfolder_input),

	x_col = "Id",

	y_col = "Ground_Truth",

	class_mode = "binary",

	batch_size = batch_size,

	target_size = (image_height, image_width)

	)



## Test generator

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_dataframe(

	dataframe = test_set,

	directory = os.path.join(main_folder, subfolder_input),

	x_col = "Id",

	y_col = "Ground_Truth",

	class_mode = "binary",

	batch_size = batch_size,

	target_size = (image_height, image_width)

	)
# CNN model hyperparameters

dropout_rate = 0.25

fc_units_1 = 1024

fc_units_2 = 512

output_units = 1



# Build CNN model

model = Sequential()



## Convolutional layer

### Layer 1

model.add(Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same",

	input_shape = (image_height, image_width, image_channels), data_format = "channels_last"))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(rate = dropout_rate))



### Layer 2

model.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same"))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(rate = dropout_rate))



### Layer 3

model.add(Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same"))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(rate = dropout_rate))



## Fully connected layer

model.add(Flatten())

### FC1

model.add(Dense(units = fc_units_2, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(rate = dropout_rate))



### Output layer

model.add(Dense(units = output_units, activation = "sigmoid"))



## CNN model summary

model.summary()
learning_rate = 0.002

# Optimizer

optimizer = Adam(lr = learning_rate)
# Compile

model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
# Callbacks

## Reduce learning rate

reduce_lr = ReduceLROnPlateau(

	monitor = "val_acc",

	factor = 0.5,

	patience = 3,

	verbose = 1,

	min_lr = 0.00001

	)



## Save model on checkpoint

checkpoint = ModelCheckpoint(

	filepath = "../checkpoint/weights.h5",

	monitor = "val_acc",

	verbose = 1,

	save_best_only = True,

	)



## Early stop if no improvement

early_stop = EarlyStopping(

	monitor = "val_acc",

	min_delta = 0.05,

	patience = 2,

	restore_best_weights = True

	)



## List of callbacks for training

callbacks = [reduce_lr, checkpoint, early_stop]
epochs = 100

# Train the model

history = model.fit_generator(

	generator = train_generator,

	steps_per_epoch = len(train_set)//batch_size,

	epochs = epochs,

	validation_data = val_generator,

	validation_steps = len(val_set)//batch_size

	)
# Visualization of training and loss loss and acc

## Plot the loss

fig, axes = plt.subplots(2, 1, figsize = (10, 8))

axes[0].plot(history.history["loss"], label = "Training Loss", color = "b", marker = "x")

axes[0].plot(history.history["val_loss"], label = "Validation loss", color = "r", marker = "v")

axes[0].title.set_text("Loss chart")

axes[0].legend(loc = "best")



## Plot the accuracy

axes[1].plot(history.history["acc"], label = "Training Acc", color = "b", marker = "x")

axes[1].plot(history.history["val_acc"], label = "Validation Acc", color = "r", marker = "v")

axes[1].title.set_text("Accuracy chart")

axes[1].legend(loc = "best")
# Provide prediction on validation and test set

## Output the probability

val_pred = model.predict_generator(

	generator = val_generator,

	steps = len(val_generator)

	)



test_pred = model.predict_generator(

	generator = test_generator,

	steps = len(test_generator)

	)



## Convert from probability to binary (0 or 1)

criteria = val_pred > 0.5

val_pred[criteria] = 1

val_pred[~criteria] = 0



criteria = test_pred > 0.5

test_pred[criteria] = 1

test_pred[~criteria] = 0
# Plot the confusion matrix

## Confusion matrix calculation

cm_val = confusion_matrix(

	y_true = val_set["Ground_Truth"].values.astype(int),

	y_pred = val_pred

	)



cm_val = pd.DataFrame(

	cm_val, 

	index = ["Normal", "Pneumonia"],

	columns = ["Normal", "Pneumonia"])



cm_test = confusion_matrix(

	y_true = test_set["Ground_Truth"].values.astype(int),

	y_pred = test_pred

	)



cm_test = pd.DataFrame(

	cm_test, 

	index = ["Normal", "Pneumonia"],

	columns = ["Normal", "Pneumonia"])



## Visualization

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))

sns.heatmap(data = cm_val, annot = True, ax = ax[0], fmt = "d")

sns.heatmap(data = cm_test, annot = True, ax = ax[1], fmt = "d")

ax[0].title.set_text("Confusion matrix for validation set")

ax[1].title.set_text("Confusion matrix for test set")