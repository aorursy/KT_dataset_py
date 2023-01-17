# Libraries

import os

# numpy
import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
from sklearn.model_selection import train_test_split

# Charts
import matplotlib.pyplot as plt

# Image IO
import skimage.io
import skimage.transform

# Deep learning
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import tensorflow

# Set random seed to make results reproducable
np.random.seed(21)
tensorflow.set_random_seed(21)
# Parameters
training_dataset_path = "../input/dataset/dataset_updated/training_set"
test_dataset_path = "../input/dataset/dataset_updated/validation_set"

# categories to use
# categories = ['drawings', 'engraving', 'iconography', 'painting']
categories = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']
n_categories = len(categories)
category_embeddings = {
    'drawings': 0,
    'engraving': 1,
    'iconography': 2,
    'painting': 3,
    'sculpture': 4
}

# After computing the mean image size, we can set a default width and a default height to resize the images
# Warning : this is a convention that I decided to use
width = 128 # 368
height = 128 # 352
n_channels = 3
# training dataset metadata
n_imgs = []
for cat in categories:
    files = os.listdir(os.path.join(training_dataset_path, cat))
    n_imgs += [len(files)]
    
plt.bar([_ for _ in range(n_categories)], n_imgs, tick_label=categories)
plt.show()
fig, axes = plt.subplots(nrows=1, ncols=n_categories, figsize=(15, 3))

cat_cpt=0
for cat in categories:
    category_path = os.path.join(training_dataset_path, cat)
    img_name = os.listdir(category_path)[0]
    img = skimage.io.imread(os.path.join(category_path, img_name))
    img = skimage.transform.resize(img, (width, height, n_channels), mode='reflect')
    axes[cat_cpt].imshow(img, resample=True)
    axes[cat_cpt].set_title(cat, fontsize=8)
    cat_cpt += 1

plt.show()
training_data = []
for cat in categories:
    files = os.listdir(os.path.join(training_dataset_path, cat))
    for file in files:
        training_data += [(os.path.join(cat, file), cat)]

test_data = []
for cat in categories:
    files = os.listdir(os.path.join(test_dataset_path, cat))
    for file in files:
        test_data += [(os.path.join(cat, file), cat)]
# Load all images to the same format (takes some time)
def load_dataset(tuples_list, dataset_path):
    indexes = np.arange(len(tuples_list))
    np.random.shuffle(indexes)
    
    X = []
    y = []
    n_samples = len(indexes)
    cpt = 0
    for i in range(n_samples):
        t = tuples_list[indexes[i]]
        try:
            img = skimage.io.imread(os.path.join(dataset_path, t[0]))
            img = skimage.transform.resize(img, (width, height, n_channels), mode='reflect')
            X += [img]
            y_tmp = [0 for _ in range(n_categories)]
            y_tmp[category_embeddings[t[1]]] = 1
            y += [y_tmp]
        except OSError:
            pass
        
        cpt += 1
        
        if cpt % 1000 == 0:
            print("Processed {} images".format(cpt))

    X = np.array(X)
    y = np.array(y)
    
    return X, y

X_train, y_train = load_dataset(training_data, training_dataset_path)
X_val, y_val = load_dataset(test_data, test_dataset_path)
# creation of a keras image generator
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True)

train_datagen.fit(X_train)
# CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=5, input_shape=(width, height, n_channels), activation='relu'))
# model.add(Conv2D(8, kernel_size=5, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(48, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(n_categories, activation='softmax'))

# Don't forget to compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit using a train and a validation generator
# train_generator = DataGenerator(training_data, training_dataset_path)
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
# test_generator = DataGenerator(test_data, test_dataset_path)

training_result = model.fit_generator(generator=train_generator,
                                      validation_data=(X_val, y_val),
                                      epochs=20,
                                      verbose=1,
                                      steps_per_epoch=len(X_train) / 32)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))

axes[0].plot(training_result.history['loss'], label="Loss")
axes[0].plot(training_result.history['val_loss'], label="Validation loss")
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Accuracy
axes[1].plot(training_result.history['acc'], label="Accuracy")
axes[1].plot(training_result.history['val_acc'], label="Validation accuracy")
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
plt.tight_layout()

plt.show()
# Let's look at more metrics
from sklearn.metrics import classification_report

X_test = []
y_test = []
for t in test_data:
    try:
        img = skimage.io.imread(os.path.join(test_dataset_path, t[0]))
        img = skimage.transform.resize(img, (width, height, n_channels), mode='reflect')
        X_test += [img]
        y_test += [category_embeddings[t[1]]]
    except OSError:
        pass

X_test = np.array(X_test)
y_test = np.array(y_test)

pred = model.predict(X_test, verbose=1)

y_pred = np.argmax(pred, axis=1)
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix

c_matrix = confusion_matrix(y_test, y_pred)
plt.imshow(c_matrix, cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
plt.show()
print(c_matrix)