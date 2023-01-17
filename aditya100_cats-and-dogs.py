# Importing the required libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

import cv2

from tqdm import tqdm



from sklearn.model_selection import train_test_split



import keras

from keras import layers

from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dense, Dropout, Flatten

from keras.models import Sequential
# Importing and Processing the dataset



train_dir = '../input/train/'



train_images = []

train_labels = []



for img in tqdm(os.listdir(train_dir)):

    try:

        img_r = cv2.imread(os.path.join(train_dir, img), cv2.IMREAD_GRAYSCALE)

        train_images.append(np.array(cv2.resize(img_r, (50, 50), interpolation=cv2.INTER_CUBIC)))

        if 'dog' in img:

            train_labels.append(1)

        else:

            train_labels.append(0)

    except Exception as e:

        print('broken image')

# Visualising the image



plt.title(train_labels[0])

_ = plt.imshow(train_images[0])
x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
x_train = np.array(x_train)

x_test = np.array(x_test)
print("Train Shape:" + str(x_train.shape))

print("Test Shape:" + str(x_test.shape))
plt.title(y_train[0])

_ = plt.imshow(x_train[0])
def baseline_model():

    model = Sequential()

    

    model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 1), activation='relu'))

    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    

    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    

    model.add(Dropout(0.2))



    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    

    model.add(Dropout(0.2))

    

    model.add(Dense(1, activation='sigmoid'))

    

    return model
model = baseline_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
x_train = x_train.reshape(-1, 50, 50, 1)

x_test = x_test.reshape(-1, 50, 50, 1)
history = model.fit(np.array(x_train), y_train, validation_data=(np.array(x_test), y_test), epochs=20, verbose=1)
hist = history.history
plt.plot(hist['loss'], 'green', label='Training Loss')

plt.plot(hist['val_loss'], 'blue', label='Validation Loss')

_ = plt.legend()
plt.plot(hist['acc'], 'green', label='Training Accuracy')

plt.plot(hist['val_acc'], 'blue', label='Validation Accuracy')

_ = plt.legend()
test_dir = '../input/test/'

test_images = []



for img in tqdm(os.listdir(test_dir)):

    try:

        img_r = cv2.imread(os.path.join(test_dir, img), cv2.IMREAD_GRAYSCALE)

        test_images.append(np.array(cv2.resize(img_r, (50, 50), interpolation=cv2.INTER_CUBIC)))

    except Exception as e:

        print('broken image')
_ = plt.imshow(test_images[0])
test_images = np.array(test_images)

test_images = test_images.reshape(-1, 50, 50, 1)

predictions = model.predict(test_images)
predictions.shape
counter = range(1, len(test_images) + 1)

solution = pd.DataFrame({"id": counter, "label":list(predictions)})

cols = ['label']



for col in cols:

    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)



solution.to_csv("dogsVScats.csv", index = False)
