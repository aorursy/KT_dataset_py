from __future__ import absolute_import, division, print_function, unicode_literals



%matplotlib inline 



# Importing modules

import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib.pyplot as plt

import matplotlib.cm as cm

from matplotlib import image as mp_image



import imageio

import imgaug as ia

from imgaug import augmenters



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from skimage import morphology

from skimage import exposure



from mlxtend.plotting import plot_confusion_matrix



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Importing data

train_data = pd.read_csv(r"../input/digit-recognizer/train.csv")

test_data = pd.read_csv(r"../input/digit-recognizer/test.csv")



# Selecting train data and labels

X_train_raw = train_data.loc[:, "pixel0":].to_numpy() / 255 # Transformed pixel values

y_train_raw = train_data.loc[:, "label"].to_numpy() # Non-encoded labels

# Test data

X_test_raw = test_data.loc[:, "pixel0":].to_numpy() / 255
# Recreating images



X_train = [] # List of train images

for pixels in X_train_raw:

    image = pixels.reshape(28,28)

    X_train.append(image)

X_train = np.array(X_train)



X_test = [] # List of test images

for pixels in X_test_raw:

    image = pixels.reshape(28,28)

    X_test.append(image)

X_test = np.array(X_test)



# Review Images

width = 6

height = 6

fig = plt.figure(figsize = (width, height))

columns = 4

rows = 5

for i in range(1, columns * rows + 1):

    fig.add_subplot(rows, columns, i)

    plt.imshow(X_train[np.random.randint(low = 0, high = 784)], cmap='Greys_r')

plt.show()



print("Shape of the training data:", X_train.shape)

print("Shape of the test data:", X_test.shape)

print("Shape of the labels:", y_train_raw.shape)
# Adding image rotation, shift, zoom

augs_one = []

for i in X_train:

    rotate = augmenters.Affine(translate_percent={"x": (-0.12, 0.12), "y": (-0.12, 0.12)}, rotate=(-12, 12),

                               scale={"x": (0.8, 1.1), "y": (0.8, 1.1)})

    aug_image = rotate.augment_image(i)

    augs_one.append(aug_image)



# Review Augmented Images

width = 6

height = 6

fig = plt.figure(figsize = (width, height))

columns = 4

rows = 5

for i in range(1, columns * rows + 1):

    fig.add_subplot(rows, columns, i)

    plt.imshow(augs_one[np.random.randint(low = 0, high = 784)], cmap='Greys_r')

plt.show()
# Adding erosion

augs_two = []

for i in X_train:

    aug_image = morphology.binary_erosion(i)

    augs_two.append(aug_image)
X_augs = np.concatenate((X_train, augs_one), axis=0) # Combine original and augmented images

y_augs = np.concatenate((y_train_raw, y_train_raw), axis=0)

print("Shape of the augmented and original data:", X_augs.shape)

print("Shape of the augmented and original labels:", y_augs.shape)
X_train_eq = []

for i in X_augs:

    # Histogram Equalization

    image_eq = exposure.equalize_hist(i)

    X_train_eq.append(image_eq)



# Review Equalized Images

width = 6

height = 6

fig = plt.figure(figsize = (width, height))

columns = 4

rows = 5

for i in range(1, columns * rows + 1):

    fig.add_subplot(rows, columns, i)

    plt.imshow(X_train_eq[np.random.randint(low = 0, high = 784)], cmap='Greys_r')

plt.show()
# Transform y-labels into categories for categorical crossentropy

y_train = tf.keras.utils.to_categorical(y_augs, num_classes=10, dtype='float32')



# Add channels (reshaping data for CNN)

X_train = np.expand_dims(X_augs, -1)

X_test = np.expand_dims(X_test, -1)
tf.keras.backend.clear_session()



# Tensorflow CNN

# 1st Stage

X_input = keras.Input(shape=(28, 28, 1), name="input")

data = layers.Conv2D(filters = 32, kernel_size = (5, 5), strides = (1, 1),

                     padding = "Same", activation = "relu")(X_input)

data = layers.Conv2D(filters = 64, kernel_size = (5, 5), strides = (1, 1),

                     padding = "Same", activation = "relu")(data)

data = layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2),

                        padding = "Same")(data)

data = layers.Dropout(rate = 0.25)(data)

# 2nd Stage

data = layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1),

                     padding = "Same", activation = "relu")(data)

data = layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1),

                     padding = "Same", activation = "relu")(data)

data = layers.Dropout(rate = 0.25)(data)

data = layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2),

                        padding = "Same")(data)

# 3rd Stage

data = layers.Conv2D(filters = 256, kernel_size = (5, 5), strides = (1, 1),

                     padding = "Same", activation = "relu")(data)

data = layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2),

                        padding = "Same")(data)

data = layers.Dropout(rate = 0.25)(data)

# 4th Stage

data = layers.Flatten()(data)

data = layers.Dense(units = 512, activation = "relu")(data)

data = layers.Dropout(rate = 0.5)(data)

data = layers.Dense(units = 10, activation = "softmax")(data)



# 

model = Model(inputs = X_input, outputs = data)



# Compile model

model.compile(optimizer = tf.compat.v2.optimizers.Adam(learning_rate=0.0005), loss = "categorical_crossentropy",

              metrics = ["accuracy"])

# Summarize

print(model.summary())
# Defining callbacks

# Getting model with lowest validation accuracy

accurate_model = ModelCheckpoint("model_weights.h5", monitor="val_accuracy", verbose=1,

                                 save_best_only=True, mode="max")



# Stopping learning with no improvment

loss_stop = EarlyStopping(monitor="val_loss", min_delta=0.00001,

                          patience = 10, mode="min", verbose=1)



# Reducing learning rate with increase of validation accuracy

reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.4,

                              patience=3, min_lr=0.0000001, mode = "auto", verbose = 1)
history = model.fit(x = X_train, y = y_train, epochs = 10, verbose = 1,

                    validation_split = 0.2, 

                    callbacks = [reduce_lr, loss_stop, accurate_model])
# DataFrame results of learning

info = {'Loss': history.history["loss"], 'Validation Loss': history.history["val_loss"],\

        "Accuracy" : history.history["acc"], "Validation Accuracy" : history.history["val_acc"]}



deep_statistics = pd.DataFrame(data=info)



# Top 5 best results from learning based on validation loss

best_results = deep_statistics[(deep_statistics["Validation Accuracy"] >= 0.90) & (deep_statistics["Accuracy"] >= 0.90)\

                               & (deep_statistics["Validation Loss"] <= 0.1) & (deep_statistics["Loss"] <= 0.1)]

best_results.sort_values("Validation Loss").head()
df = pd.DataFrame(history.history)

df.rename(columns={"acc":"Training", "val_acc":"Validation",

                   "loss":"Train Loss", "val_loss":"Validation Loss"}, inplace = True)



sns.set(font_scale=1.2)

sns.set_style("whitegrid")



palette = sns.diverging_palette(10, 220, sep=80, n=2)



# Training VS Validation plot

figure, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20,6))

axes[0].set(xlabel="Epoch", ylabel="Accuracy")

sns.lineplot(data=df[["Training", "Validation"]], marker = "o", ax=axes[0], palette=palette, legend="full")

axes[1].set(xlabel="Epoch", ylabel="Loss")

sns.lineplot(data=df[["Train Loss", "Validation Loss"]], marker = "d", ax=axes[1], palette=palette, legend="full")
# Re-evaluating train set

train_predictions = model.predict(X_train)

numbers = [np.where(number == np.amax(number)) for number in train_predictions] # Get predictions labels

numbers_ravel = np.concatenate(numbers).ravel() # List of predicted numbers



# Comparing labels

result = confusion_matrix(y_augs, numbers_ravel)



classes = ["0","1","2","3","4","5","6","7","8","9"]



fig, ax = plot_confusion_matrix(conf_mat = result,

                                colorbar = True,

                                show_absolute = True,

                                show_normed = True,

                                class_names = classes, figsize = (10, 10))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
# Gettting predictions of test set

predictions = model.predict(X_test)

numbers = [np.where(number == np.amax(number)) for number in predictions] # Get predictions labels

numbers = np.concatenate(numbers).ravel() # List of predicted numbers

ImageId = np.arange(1, X_test.shape[0]+1)