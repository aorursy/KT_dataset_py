# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline
training_set = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
testing_set = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
submissions = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

nrows, ncols, channels = 28, 28, 1
training_set.dropna(axis = 0, inplace = True)
seed = 20

X_train, X_val, y_train, y_val = train_test_split(training_set.drop(['label'], axis=1), training_set[['label']],
                                                    random_state = seed, test_size=0.2)

print("Training set size = {}, Validation set size = {}".format(len(X_train), len(X_val)))
print("Number of images in testing set = {}".format(training_set.shape[0]))
X_train = X_train / 255
X_val = X_val / 255
testing_set = testing_set / 255

# reshape the entire data to images of size 28x28
X_train = X_train.values.reshape(-1, nrows, ncols, channels)
X_val = X_val.values.reshape(-1, nrows, ncols, channels)
testing_set = testing_set.values.reshape(-1, nrows, ncols, channels)

# convert numerical classes into categorical classes
y_train = keras.utils.to_categorical(y_train, num_classes = 10)
y_val = keras.utils.to_categorical(y_val, num_classes = 10)
def visualize_data(rows, cols, data, predictions=None):
    img = 0
    
    plt.rcParams['figure.figsize'] = (15, 15)
    fig, axes = plt.subplots(rows, cols)

    for i in range(0, rows):
        for j in range(0, cols):
            axes[i, j].imshow(data[img, :, :, 0], cmap='gray')
            axes[i, j].grid(False)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if predictions is not None:
                axes[i, j].set_xlabel('Predcited = {}'.format(predictions[img]))
            img += 1
            
visualize_data(10, 10, X_train)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=5, width_shift_range=3,
                            height_shift_range=3, shear_range=0.3, zoom_range=0.08)

train = datagen.flow(X_train, y_train, batch_size=64)
val = datagen.flow(X_val, y_val, batch_size=64)

train.n
model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(nrows, ncols, channels)),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam', metrics = ['accuracy'])

model.summary()
history = model.fit(x=train, epochs=5, validation_data=val)
print("Shape of testing dataset:", testing_set.shape)
arr_probabilities = model.predict(testing_set, batch_size=500)
predictions = []

for probabilities in arr_probabilities:
    predictions.append(np.argmax(probabilities))
    
predictions[:10]
visualize_data(2, 5, testing_set, predictions)
submissions['Label'] = predictions
submissions.head()