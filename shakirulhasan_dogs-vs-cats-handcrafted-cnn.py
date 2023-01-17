# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import zipfile
import os
import shutil
# Removing files that are allready unzippped
for folder in os.listdir("/kaggle/working/"):
    if folder != "__notebook_source__.ipynb":
        shutil.rmtree("/kaggle/working/"+folder)

# Unzip train data
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip") as zip_ref:
    zip_ref.extractall("/kaggle/working/train")

# Unzip test data
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/test1.zip") as zip_ref:
    zip_ref.extractall("/kaggle/working/test")
TRAIN_DIR = "/kaggle/working/train/"
TEST_DIR = "/kaggle/working/test/"

# Creating separate folder for cat and dog images
# Using try catch so that if the folder is already created, then it won't throw any error
try:
    os.mkdir(TRAIN_DIR + "cat")
    os.mkdir(TRAIN_DIR + "dog")
except os.error:
    pass

# Seperating training data
for image in os.listdir(TRAIN_DIR + "train"):
    source = TRAIN_DIR + "train/" + image
    destination = TRAIN_DIR + image.split(".")[0]
    shutil.move(source, destination)

# Test data can't be sepearated, so putting them in one folder
for image in os.listdir(TEST_DIR + "test1"):
    source = TEST_DIR + "test1/" + image
    destination = TEST_DIR
    shutil.move(source, destination)

# Removing the empty folders
os.rmdir(TRAIN_DIR + "train")
os.rmdir(TEST_DIR + "test1")
DOG_DIR = "/kaggle/working/train/dog/"
CAT_DIR = "/kaggle/working/train/cat/"

TEST_FILES = os.listdir(TEST_DIR)
CAT_FILES = os.listdir(CAT_DIR)
DOG_FILES = os.listdir(DOG_DIR)

TEST_DATA_SIZE = len(TEST_FILES)
DOG_DATA_SIZE = len(DOG_FILES)
CAT_DATA_SIZE = len(CAT_FILES)

print(f"There are {DOG_DATA_SIZE} dog, {CAT_DATA_SIZE} cat and {TEST_DATA_SIZE} test images.")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
# Getting random indexes
cat1, cat2, dog1, dog2 = [random.randint(0, CAT_DATA_SIZE-1) for i in range(4)]

# Showing the images on a subplot
figure, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(8,8))

axes[0, 0].imshow(mpimg.imread(CAT_DIR + CAT_FILES[cat1]))
axes[0, 1].imshow(mpimg.imread(CAT_DIR + CAT_FILES[cat2]))
axes[1, 0].imshow(mpimg.imread(DOG_DIR + DOG_FILES[dog1]))
axes[1, 1].imshow(mpimg.imread(DOG_DIR + DOG_FILES[dog2]))
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from tensorflow.keras.models import Sequential
model = Sequential([
    Input((150, 150, 3)),
    Conv2D(16, 3, activation="relu"),
    MaxPool2D(2, 2),
    Conv2D(32, 3, activation="relu"),
    MaxPool2D(2, 2),
    Conv2D(64, 3, activation="relu"),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(512, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
BATCH_SIZE = 32
train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2,
                                   rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR,
                                              target_size=(150, 150),
                                              batch_size=BATCH_SIZE,
                                              class_mode="binary",
                                              subset="training")

validation_gen = train_datagen.flow_from_directory(TRAIN_DIR,
                                                   target_size=(150, 150),
                                                   batch_size=BATCH_SIZE,
                                                   class_mode="binary",
                                                   subset="validation")
history = model.fit(train_gen,
                    epochs=10,
                    steps_per_epoch=train_gen.samples // BATCH_SIZE,
                    validation_data=validation_gen,
                    validation_steps=validation_gen.samples // BATCH_SIZE)
# Plotting loss and accuracy
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
axes[0].plot(history.history["loss"], label="Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[1].plot(history.history["accuracy"], label="Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[0].legend()
axes[1].legend()
fig.show()
import pandas as pd
test_df = pd.DataFrame({'filename': TEST_FILES})

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_dataframe(test_df,
                                            TEST_DIR,
                                            x_col="filename",
                                            y_col=None,
                                            class_mode=None,
                                            target_size=(150, 150),
                                            shuffle=False,
                                            batch_size=BATCH_SIZE)
predict = model.predict(test_gen, steps=np.ceil(test_df.shape[0] / BATCH_SIZE))
output = (predict >= 0.5).astype(int)
# As output was a 2D array, I had to squeeze it
output = np.squeeze(output).tolist()
# Creating the label map for categories
label_map = dict((v, k) for k, v in train_gen.class_indices.items())
CATEGORY = [label_map[i] for i in output]

TEST_ID = []
for file in TEST_FILES:
    TEST_ID.append(file.split('.')[0])
# Creating the dataframe for submission
submission_df = pd.DataFrame({
    "id": TEST_ID,
    "label": CATEGORY
})

submission_df.to_csv("submission.csv", index=False)