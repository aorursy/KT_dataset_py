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
# Imports

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
# Assign Default Variables

raw_csv = "/kaggle/input/digit-recognizer/train.csv"
test_csv = "/kaggle/input/digit-recognizer/test.csv"
# Load Dataset

raw_df = pd.read_csv(raw_csv)
test_df = pd.read_csv(test_csv)
def get_image_and_label(data_frame):
    
    IMGs = data_frame.drop(["label"], axis=1).values if 'label' in data_frame.columns else data_frame.values
    IMGs = np.array([image.reshape((28, 28)) for image in IMGs])
    IMGs = np.expand_dims(IMGs, axis=3)

    labels = data_frame['label'].values if 'label' in data_frame.columns else None
    
    return IMGs, labels
raw_IMGs, raw_labels = get_image_and_label(raw_df)
test_IMGs, _ = get_image_and_label(test_df)
classes = len(set(raw_labels))
classes
raw_labels = to_categorical(raw_labels, num_classes=classes)
raw_labels
train_IMGs, validation_IMGs, trian_labels, validation_labels = train_test_split(raw_IMGs, raw_labels, test_size=0.1, random_state=42)
print("Training Set Details:")
print(train_IMGs.shape)
print(trian_labels.shape)

print("\n")

print("Validation Set Details:")
print(validation_IMGs.shape)
print(validation_labels.shape)

print("\n")

print("Testing Set Details:")
print(test_IMGs.shape)
model = Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(1024, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(int(classes), activation="softmax")
])
model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adam(learning_rate=1e-4),
              metrics=['accuracy'])
model.summary()
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)

validation_datagen = ImageDataGenerator(rescale=1/255)

test_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow(train_IMGs, trian_labels, batch_size=32)
validation_generator = train_datagen.flow(validation_IMGs, validation_labels, batch_size=32)
test_generator = test_datagen.flow(test_IMGs, batch_size=32, shuffle=False)
history = model.fit_generator(train_generator, epochs=100, validation_data=validation_generator, verbose=1)
model.evaluate(validation_IMGs, validation_labels)
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'r', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()
pred_labels = model.predict_generator(test_generator)
pred_labels = np.argmax(pred_labels, axis=-1)
pred_labels
# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 2
ncols = 5

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

random_numbers = np.random.choice(range(len(pred_labels)), 10, replace=False)

rand_IMGs = [test_IMGs[num] for num in random_numbers]
rand_labels = [pred_labels[num] for num in random_numbers]

for i, img, label in zip(range(len(random_numbers)), rand_IMGs, rand_labels):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)
    
    plt.imshow(img[:,:,0])
    plt.title(label)

plt.show()
my_submission = pd.DataFrame({'ImageId': test_df.index + 1, 'Label': pred_labels})
my_submission.head()
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
