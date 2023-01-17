import numpy as np
import pandas as pd
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()

test= pd.read_csv("../input/test.csv")
print(test.shape)
test.head()
X_train = (train.ix[:,1:].values).astype('float32') # all pixel values
Y_train_org = train.ix[:,0].values.astype('int32') # only labels i.e targets digits
X_test_org = test.values.astype('float32')
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test_org.reshape(X_test_org.shape[0], 28, 28,1)

Y_train = to_categorical(Y_train_org)
print(Y_train.shape)
def DigitModel(input_shape):
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
    X_input = Input(input_shape)
    X = ZeroPadding2D((2,2))(X_input)
    
    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 2, name = 'bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    
    X = Conv2D(20, (3, 3), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
        
    X = Conv2D(12, (3, 3), strides = (1, 1), name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2, 2), name='max_pool3')(X)



    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(10, activation='softmax', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='DigitModel')
    ### END CODE HERE ###
    
    return model
print(X_train[0].shape)
digitModel = DigitModel(X_train[0].shape)
digitModel.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
print(Y_train.shape)
digitModel.fit(X_train, Y_train, epochs=2, batch_size=32)
#X_test_show = X_test_org.reshape(test.shape[0],28,28)
#print(X_test[0:2].shape)
#plt.imshow(X_test_show[1], cmap=plt.get_cmap('gray'))
#print(np.argmax(digitModel.predict(X_test[0:2],verbose=1), axis=1))
predictions = digitModel.predict(X_test, verbose=1)
print(X_test.shape)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": np.argmax(predictions, axis=1)})
submissions.to_csv("DR4.csv", index=False, header=True)



