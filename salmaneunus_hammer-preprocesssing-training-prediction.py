import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tqdm import tqdm

%matplotlib inline
train = pd.read_csv('../input/mechanical-tools-dataset/labels_hammer-dataset_2020-10-10-11-57-53.csv')    # reading the csv file

train.head()  
train.columns
train_image = []

for i in tqdm(range(train.shape[0])):

    img = image.load_img('../input/mechanical-tools-dataset/Mechanical Tools Image dataset-20201009T110652Z-001/Mechanical Tools Image dataset/Hammer/'+train['id'][i],target_size=(400,400,3))

    img = image.img_to_array(img)

    img = img/255

    train_image.append(img)

X = np.array(train_image)
X.shape
plt.imshow(X[5])
plt.imshow(X[15])
y = np.array(train.drop(['id', 'Hammer'],axis=1))
y.shape
y
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(6, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)
img = image.load_img('../input/mechanical-tools-dataset/Mechanical Tools Image dataset-20201009T110652Z-001/Mechanical Tools Image dataset/Hammer/000050.jpg',target_size=(400,400,3))

img = image.img_to_array(img)

img = img/255
classes = np.array(train.columns[0:])

proba = model.predict(img.reshape(1,400,400,3))

top_3 = np.argsort(proba[0])[:-4:-1]

for i in range(3):

    print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))

plt.imshow(img)

proba
img = image.load_img('../input/mechanical-tools-dataset/Mechanical Tools Image dataset-20201009T110652Z-001/Mechanical Tools Image dataset/Hammer/000060.jpg',target_size=(400,400,3))

img = image.img_to_array(img)

img = img/255
classes = np.array(train.columns[0:])

proba = model.predict(img.reshape(1,400,400,3))

top_3 = np.argsort(proba[0])[:-4:-1]

for i in range(3):

    print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))

plt.imshow(img)

proba