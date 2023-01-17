import numpy as np 

import pandas as pd 

import os

import tensorflow as tf

import keras

from keras.applications import VGG19

from keras.models import Sequential

from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import cv2
# Load the images into a dataframe



df = []



crops = ['jute', 'maize', 'rice', 'sugarcane', 'wheat']



# Useful dict for switching between crop names and labels

crop_to_label = {}

for i, crop in enumerate(crops):

    crop_to_label[crop] = i

    

label_to_crop = {value:key for (key, value) in crop_to_label.items()}



for crop in crops:

    subdir = '../input/agriculture-crop-images/kag2/' + crop

    for path in os.listdir(subdir):

        df.append([os.path.join(subdir, path), crop])

    

df = pd.DataFrame(df, columns=['path', 'label'])

df = df.sample(frac=1, random_state=0).reset_index(drop=True) # shuffle the rows

df.head()
def preprocess_image(path):

    """Helper function to read, resize and rescale an image from its path"""

    im = plt.imread(path)

    im = cv2.resize(im, (224,224), interpolation=cv2.INTER_CUBIC)    

    return im/255.



# Test on first image in dataset

im = preprocess_image(df.loc[0, 'path'])

label = df.loc[0, 'label']



# Get the image dimensions to a variable

img_size, _, channels = im.shape



# Show the image with its label

plt.title(label)

plt.imshow(im)

plt.show()
n_examples = len(df.index)

n_classes = len(crops)



# Initialize X and y

X = np.zeros(shape=(n_examples, img_size, img_size, channels))

y = np.zeros(shape=(n_examples, n_classes))



# Loop through dataset to set values of X and y

for i, idx in enumerate(df.index):

    path, label = df.loc[idx, :]

    X[i, :, :, :] = preprocess_image(path)

    y[i, crop_to_label[label]] = 1
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=0)
num_classes = 5



model = Sequential()

vgg = VGG19(input_shape=(img_size,img_size,channels),include_top=False,weights = 'imagenet',pooling='avg')

model.add(vgg)

model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))

model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model.add(Dense(num_classes, activation='softmax'))



model.layers[0].trainable = False
batch_size = 16

epochs = 30

learning_rate = 1e-3
model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, decay=learning_rate/epochs), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x=X_train, y=y_train, validation_data=(X_dev, y_dev), epochs=epochs, batch_size=batch_size)
plt.figure(figsize=(10, 10))



plt.subplot(2, 2, 1)

plt.title('Training accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.plot(history.history['accuracy'])



plt.subplot(2, 2, 2)

plt.title('Validation accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.plot(history.history['val_accuracy'])



plt.subplot(2, 2, 3)

plt.title('Training loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.plot(history.history['loss'])



plt.subplot(2, 2, 4)

plt.title('Validation loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.plot(history.history['val_loss'])



plt.show()
test_df = pd.read_csv('../input/testssss/testdata.csv', index_col=0)



test_df.head()
X_test = np.zeros(shape=(len(test_df.index), img_size, img_size, channels))

y_test = np.zeros(shape=(len(test_df.index), 5))



for i, idx in enumerate(test_df.index):

    path, crop = test_df.loc[idx, ['testpath', 'crop']]

    X_test[i, :, :, :] = preprocess_image(path)

    y_test[i, crop_to_label[crop]] = 1

    

plt.imshow(X_test[0, :, :, :])

plt.title(label_to_crop[np.argmax(y_test[0, :])] + ' - Model predicts ' + label_to_crop[model.predict_classes(np.array([X_test[0,:, :, :]]))[0]])

plt.show()
print('Test accuracy - {}%'.format(model.evaluate(X_test, y_test)[1] * 100))
def model_predict(path):

    im = np.array([preprocess_image(path)])

    prediction = model.predict_classes(im)

    return np.vectorize(label_to_crop.get)(prediction)[0], model.predict(im)[0, prediction[0]]



plt.figure(figsize=(20, 20))



for n, path in enumerate(os.listdir('../input/new-images')): 

    prediction, confidence = model_predict('../input/new-images/' + path)

    plt.subplot(2, 2, n+1)

    plt.title('Model predicts {}, confidence={}'.format(prediction, confidence))

    plt.imshow(preprocess_image('../input/new-images/' + path))
submission = pd.DataFrame(np.array([list(test_df['testpath'].values), list(model.predict_classes(X_test))]).T, columns=['pathname', 'label'])

submission.to_csv('submission.csv', index = False)