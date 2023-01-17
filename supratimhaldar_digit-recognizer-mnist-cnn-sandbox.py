import numpy as np

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model



import keras.backend as K

K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



%matplotlib inline



import pandas as pd

import numpy as np

import os

import time
mnist = pd.read_csv(os.path.join("../input/", "train.csv"))

print("Dataset size =", mnist.shape)



# Prepare features(X) and target(y)

X = mnist.drop("label", axis=1)

y = mnist["label"]
# Split dataset into train and test set

from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(mnist, test_size=0.2, random_state=42)

print("Length of train_set =", train_set.shape[0], "AND test_set =", test_set.shape[0])



# Now shuffle the train dataset

split_idx = len(train_set)

shuffle_index = np.random.permutation(split_idx)

train_set = train_set.iloc[shuffle_index]

print("Length of shuffled train_set =", train_set.shape[0])



# Extract features(X) and target(y) from train and test set

X_train = train_set.drop("label", axis=1)

y_train = train_set["label"]



X_test = test_set.drop("label", axis=1)

y_test = test_set["label"]



# Scale features

X_train = X_train/255

X_test = X_test/255

X=X/255



print("Shapes =", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# Reshape the data

from keras.utils import to_categorical

n_H, n_W = 28, 28



m_train = X_train.shape[0]

X_train_reshaped = X_train.values.reshape(m_train, n_H, n_W, 1)

y_train_reshaped = to_categorical(y_train)



m_test = X_test.shape[0]

X_test_reshaped = X_test.values.reshape(m_test, n_H, n_W, 1)

y_test_reshaped = to_categorical(y_test)
def DigitRecognizerModel(input_shape):

    """

    Implementation of the DigitRecognizerModel.

    

    Arguments:

    input_shape -- shape of the images of the dataset



    Returns:

    model -- a Model() instance in Keras

    """

    X_input = Input(input_shape)

    

    # Layer

    X = ZeroPadding2D((3,3))(X_input)

    X = Conv2D(32, (7,7), strides=(1,1), name='conv0')(X)

    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    X = Activation('relu')(X)

    

    # Layer

    X = MaxPooling2D((5, 5), strides=(3,3), name='max_pool0')(X)

    

    # Layer

    X = ZeroPadding2D((3,3))(X_input)

    X = Conv2D(128, (3,3), strides=(1,1), name='conv1')(X)

    X = BatchNormalization(axis = 3, name = 'bn1')(X)

    X = Activation('relu')(X)

    

    # Layer

    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    

    # Layer

    X = Flatten()(X)

    X = Dense(10, activation='softmax', name='fc')(X)

    

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.

    model = Model(inputs=X_input, outputs=X, name='DigitRecognizerModel')

    

    return model
model = DigitRecognizerModel(X_train_reshaped.shape[1:])

model.compile(loss="categorical_crossentropy",

                   optimizer="Adamax",

                   metrics=["accuracy"])
#model.fit(x=X_train_reshaped, y=y_train_reshaped, epochs=100, batch_size=128)
m = X.shape[0]

X_reshaped = X.values.reshape(m, n_H, n_W, 1)

y_reshaped = to_categorical(y)

model.fit(x=X_reshaped, y=y_reshaped, epochs=50, batch_size=32)
preds_train = model.evaluate(x=X_train_reshaped, y=y_train_reshaped)

preds_test = model.evaluate(x=X_test_reshaped, y=y_test_reshaped)



print()

print ("Train set : Loss = " + str(preds_train[0]))

print ("Train set : Accuracy = " + str(preds_train[1]))

print()

print ("Test set : Loss = " + str(preds_test[0]))

print ("Test set : Accuracy = " + str(preds_test[1]))
##################################################################################

# Test on Kaggle test dataset and prepare submission file

##################################################################################

mnist_test = pd.read_csv(os.path.join("../input/", "test.csv"))



mnist_test = mnist_test/255

m_test_data = mnist_test.shape[0]

X_test_data_reshaped = mnist_test.values.reshape(m_test_data, n_H, n_W, 1)



submission_out_path = os.path.join(".", 

                                   "Submission_05Mar19_CNN1_clf.csv")



test_pred_proba = model.predict(X_test_data_reshaped)

test_pred_classes = np.argmax(test_pred_proba, axis=1)



my_submission = pd.DataFrame({ \

            'ImageId': np.arange(1, m_test_data+1), \

            'Label': test_pred_classes})

my_submission.to_csv(submission_out_path, index=False)
model.summary()