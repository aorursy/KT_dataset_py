import math
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

import os
%matplotlib inline
fashion_train_set_original = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
fashion_test_set_original = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
fashion_train_set_original.head()
LABELS = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
def to_28_by_28_gray_img_matrix(imgrow):
    """Transform a row from the dataset into a numpy matrix of shape (28,28) containing the image's pixel values"""
    return imgrow[1:].values.reshape((28,28))
fig, axes = plt.subplots(3, 5)
fig.set_size_inches(10,4)
fig.tight_layout()

for i, ax in enumerate(axes.flatten()):
    imgrow = fashion_train_set_original.iloc[i]
    ax.imshow(to_28_by_28_gray_img_matrix(imgrow), cmap='gray')
    ax.axis('off')
    ax.set_title(LABELS[imgrow['label']])
def reshape_and_normalize_inputs(X):
    X = X.reshape((X.shape[0], 28, 28, 1))
    X = X/255.
    return X
X_train, X_dev, y_train, y_dev = train_test_split(
    fashion_train_set_original.values[:,1:], 
    fashion_train_set_original.values[:,0], 
    test_size=0.20, 
    random_state=2
)
X_train = reshape_and_normalize_inputs(X_train)
X_dev = reshape_and_normalize_inputs(X_dev)
# transform the label integers into one-hot vectors
enc = OneHotEncoder()
enc.fit(y_train.reshape(-1,1))
y_train_OH = enc.transform(y_train.reshape(-1,1)).toarray()
y_dev_OH = enc.transform(y_dev.reshape(-1,1)).toarray()
def fashion_mnist_model(input_shape):
    """Create a CNN 
    
    Args:
        input_shape (tuple of int): the height, width and channels as a tuple
    
    Returns:
        a trainable CNN model
    """

    X_input = Input(input_shape)

    X = Conv2D(20, (3, 3), strides = (1, 1), name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool_1')(X)
    
    X = Conv2D(46, (2, 2), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='fc1')(X)
    X = Dense(10, activation='softmax', name='fc2')(X)

    model = Model(inputs = X_input, outputs = X, name='fashion_mnist_model')

    return model
model = fashion_mnist_model(X_train.shape[1:])
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(x = X_train, y = y_train_OH, epochs = 10, batch_size = 64)
preds = model.evaluate(x = X_dev, y = y_dev_OH)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
y_dev_predict = np.argmax(model.predict(X_dev), axis=1)
print(confusion_matrix(y_dev, y_dev_predict))
plt.imshow((X_dev[(y_dev == 6) & (y_dev_predict != 6)]*255.)[6].reshape((28,28)), cmap='gray')
X_test = fashion_test_set_original.values[:,1:].copy()
y_test = fashion_test_set_original.values[:,0].copy()
X_test = reshape_and_normalize_inputs(X_test)
y_test_OH = enc.transform(y_test.reshape(-1,1))
preds = model.evaluate(x = X_test, y = y_test_OH)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))