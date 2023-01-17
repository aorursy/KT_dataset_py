import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from glob import glob

from skimage.io import imread

import keras.backend as k

import tensorflow as tf

import os



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer
df = pd.DataFrame({'path': glob(os.path.join('../input/train', '*.tif'))})

df['id'] = df.path.map(lambda x: x.split('/')[3].split(".")[0])

labels = pd.read_csv('../input/train_labels.csv')

df = df.merge(labels, on="id")

df.head()
df0 = df[df.label == 0].sample(500, random_state=42)

df1 = df[df.label == 1].sample(500, random_state=42)

df = pd.concat([df0, df1], ignore_index=True).reset_index()

df = df[["path", "id", "label"]]

df.shape
df['image'] = df['path'].map(imread)

df.head()
image = (df['image'][500], df['label'][500])

_ = plt.imshow(image[0])

_ = plt.title(image[1])
input_images = np.stack(list(df.image), axis=0)

input_images.shape
Y = LabelBinarizer().fit_transform(df.label)

X = input_images
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model
def model(input_shape):

    # Defining the input placeholder

    X_input = Input(input_shape)

    

    # Padding the borders

    X = ZeroPadding2D((3, 3))(X_input)

    

    # Applying the first block

    X = Conv2D(32, (7, 7), strides= (1, 1), name='conv0')(X)

    X = BatchNormalization(axis=3, name='bn0')(X)

    X = Activation('relu')(X)

    

    # MaxPool

    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    

    # Applying the second block

    X = Conv2D(64, (7, 7), strides= (1, 1), name='conv1')(X)

    X = BatchNormalization(axis=3, name='bn1')(X)

    X = Activation('relu')(X)

    

    # MaxPool

    X = MaxPooling2D((2, 2), name='max_pool2')(X)

      

    # Applying the third block

    X = Conv2D(128, (7, 7), strides= (1, 1), name='conv2')(X)

    X = BatchNormalization(axis=3, name='bn2')(X)

    X = Activation('relu')(X)

    

    # MaxPool

    X = MaxPooling2D((2, 2), name='max_pool3')(X)  

    

    

    # Flatten and FullyConnected Layer

    X = Flatten()(X)

    X = Dense(1, activation='sigmoid', name='fc')(X)

    

    model = Model(inputs=X_input, outputs=X, name='Model')

    

    return model
model_final = model(train_X.shape[1:])
model_final.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model_final.fit(train_X, train_Y, epochs=10, batch_size=50)
evals = model_final.evaluate(test_X, test_Y, batch_size=32, verbose=1)



print('Test accuracy: '+str(evals[1]*100)+'%')
test_data = pd.DataFrame({'path': glob(os.path.join('../input/test', '*.tif'))})

test_data['id'] = test_data.path.map(lambda x: x.split('/')[3].split(".")[0])

test_data['image'] = test_data['path'].map(imread)
test_images = np.stack(test_data.image, axis=0)

test_images.shape
predicted_labels = [model_final.predict(np.expand_dims(tensor, axis=0))[0][0] for tensor in test_images]

predictions = np.array(predicted_labels)

test_data['label'] = predictions

submission = test_data[["id", "label"]]

submission.head()
submission.to_csv("submission.csv", index = False, header = True)