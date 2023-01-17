# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
#the data is present as a flat file; for cnn we need the input as a 28 * 28 matrix
#the first col in df is the label
X_train_df = train_df[train_df.columns[1:]].as_matrix()
#reshape the input to a m * len * height matrix; m -> # train examples; len and height -> 28 in this case
X_train = X_train_df.reshape(42000,28,28,1)
X_train_reshaped = X_train / 255.0
Y_train = train_df[train_df.columns[0:1]]
from keras.utils import np_utils
y_train_reshaped = np_utils.to_categorical(Y_train, num_classes=10)
def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (6, 6), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(10, activation='softmax', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    ### END CODE HERE ###
    
    return model
happyModel = HappyModel(input_shape=X_train_reshaped[0].shape)
happyModel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
happyModel.fit(x=X_train_reshaped, y=y_train_reshaped,epochs = 5, batch_size = 16)
#try new architecture: add dropout
test_df = pd.read_csv('../input/test.csv')
X_test = test_df.as_matrix()
X_test = X_test.reshape(28000,28,28,1)
X_test_norm = X_test / 255.0
preds = happyModel.predict(x=X_test_norm)
# select the indix with the maximum probability
results = np.argmax(preds,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)