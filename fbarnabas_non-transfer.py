# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import layers, models, optimizers, metrics, losses
from sklearn.model_selection import train_test_split
from keras.preprocessing import image

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
"""
full_image_prefixes = os.listdir('/kaggle/input/football-player-number-13/images')

image_prefixes = []

for string in full_image_prefixes:
    image_prefixes.append(string.split('_')[0])

unique_image_prefixes = list(set(image_prefixes))
"""
df_train = pd.read_csv('/kaggle/input/football-player-number-13/train_solutions.csv')

train_samples = df_train['Sample'].to_list()

train_samples = set(train_samples)

print(train_samples)
print(len(train_samples))
df_test = pd.read_csv('/kaggle/input/football-player-number-13/sampleSubmissionAllZeros.csv')

test_samples = df_test['Sample'].to_list()

test_samples = set(test_samples)

print(test_samples)
print(len(test_samples))

"""
img = image.load_img('/kaggle/input/football-player-number-13/images/0dd0ec11e0b4818ec9d478236915cd95_7.jpg')

img_2 = image.load_img('/kaggle/input/football-player-number-13/images/0dd0ec11e0b4818ec9d478236915cd95_8.jpg')

array = image.img_to_array(img)

array_2 = image.img_to_array(img_2)

a = []

a.append(array)

print(len(a))

a.append(array_2)

print(len(a))

a = np.array(a)

print(a.shape)
"""


input_path = '/kaggle/input/football-player-number-13/images/'
"""
X_train = []

for unique_image_prefix in unique_image_prefixes:
    image_slices = []
    image_labels = []
    for i in range(1,17):
        image_path = input_path + unique_image_prefix + '_' + str(i) + '.jpg'
        img = image.load_img(image_path, target_size=(480, 270))
        array = image.img_to_array(img)
        image_slices.append(array)
    image_slices = np.array(image_slices)
    X_train.append(image_slices)
"""
X_train = []
y_train = []

input_shape = (240, 135)


for sample in train_samples:
    for i in range(1,17):
        image_path = input_path + sample + '_' + str(i) + '.jpg'
        img = image.load_img(image_path, target_size=input_shape)
        array = image.img_to_array(img)
        X_train.append(array)
        y_train.append(df_train[(df_train.Sample == sample) & (df_train.Label == i)].Predicted.to_string(index=False).strip())
X_train = np.array(X_train)
print(X_train.shape)

# [f(x) if condition else g(x) for x in sequence]


y_train = [1 if x == 'True' else 0 for x in y_train]
print(y_train)
y_train = np.array(y_train)
print(y_train.shape)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
"""
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
"""

X_test = []


for sample in test_samples:
    for i in range(1,17):
        image_path = input_path + sample + '_' + str(i) + '.jpg'
        img = image.load_img(image_path, target_size=input_shape)
        array = image.img_to_array(img)
        X_test.append(array)

X_test = np.array(X_test)
model = models.Sequential()

model.add(layers.Conv2D(4, (8, 8), activation='relu',input_shape=(input_shape[0], input_shape[1], 3)))
model.add(layers.MaxPooling2D((8, 8)))
model.add(layers.Dropout(0.5))
#model.add(layers.BatchNormalization())


model.add(layers.Conv2D(4, (4, 4), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Dropout(0.5))
#model.add(layers.BatchNormalization())

#model.add(layers.Conv2D(32, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((3, 3)))
#model.add(layers.Dropout(0.5))

model.add(layers.Flatten())

model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
opt = optimizers.RMSprop(lr=0.0004)

rec = metrics.Recall()
prec = metrics.Precision()
#acc = metrics.Accuracy()


model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[rec, prec, 'accuracy'])

class_weights = {0: 1.0, 1: 13.5}
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=8,
    validation_split=0.3,
    class_weight=class_weights,
    #validation_data=(X_val, y_val),
    #callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(X_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test recall:', score[1])
print('Test precision:', score[2])
print('Test accuracy:', score[3])
