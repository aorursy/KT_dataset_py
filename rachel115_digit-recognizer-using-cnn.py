# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
# load train and test dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# check train and test dataset shape
print("train dataset's shape",train.shape)
print("test dataset's shape",test.shape)
# check missing values
print("Existing null in train data: ", train.isnull().values.any())
print("Existing null in test data: ", test.isnull().values.any())
# extract features and label from train dataset
X_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
# show picture and corresponding label
plt.imshow(X_train.iloc[5].values.reshape(28,28))
print ("the corresponding label is: ", y_train[5])
# reshape X_train and test data and conver y_train to one-hot encode
# our image data is channel_last
X_train = X_train.values.reshape(-1,28,28,1)
X_test = test.values.reshape(-1,28,28,1)
print("X_train shape",X_train.shape)
print("X_test data shape",X_test.shape)
# one hot encode
n_classes = 10
y_train = to_categorical(y_train, n_classes)
print("y_train shape",y_train.shape)
# split train dataset into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train, 
                                                 test_size = 0.2, 
                                                 random_state = 42)
print("Training Data Shape = ", X_train.shape, y_train.shape)
print("Validation Data Shape = ", X_val.shape, y_val.shape)
# normalize data: divided by 255 since the range of pixel is 0~255
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0
# code reference Keras - Tutorial - Happy House v2 (coursera) and make some changes
# build a CNN model using Keras
def CNN_Model(input_shape):
    """
    Implementation of CNN model.    
    Argument:
    input_shape -- shape of the images
    Return:
    model -- a CNN model
    """
    # Define the input placeholder as a tensor with shape input_shape. 
    X_input = Input(input_shape)
    
    # Zero-Padding: pads the border of X_input with zeroes
    #the original shape of one image is (28,28,1). After this ZeroPadding2D, the shape of one image is (32,32,1)
    X = ZeroPadding2D((2,2))(X_input)

    # this CNN structure is: CONV layer -> BN  layer in channel dimension -> RELU activation
    # our case is channel_last so BatchNormalization(axis=3). It means normalize follow channel axis.
    X = Conv2D(32, (3, 3), strides = (1, 1))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X + FC
    X = Flatten()(X)
    X = Dense(n_classes, activation='softmax')(X)

    # Create model.
    model = Model(inputs = X_input, outputs = X)
    
    return model
#Step1: create the model using X_train 
cnn_Model = CNN_Model(X_train[0].shape)
#Step2: compile the model
cnn_Model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
#Step3: train the model
cnn_Model.fit(x = X_train, y = y_train, epochs = 50, batch_size = 32)
# evaluate model on training dataset and validation dataset.
trainEva = cnn_Model.evaluate(X_train, y_train)
valEva = cnn_Model.evaluate(X_val, y_val)
print("Training Loss = ", trainEva[0], "Training Acc = ", trainEva[1])
print("Validation Loss = ", valEva[0], "Validation Acc = ", valEva[1])
# predict class for test data
y_pred = cnn_Model.predict(X_test)
# select the indix with the maximum probability
# axis=0 means select maximum by column, axis=1 means select maximum by row
y_pred = np.argmax(y_pred,axis = 1)
# show picture and predict label in test
plt.imshow(test.iloc[0].values.reshape(28,28))
print ("the corresponding label is: ", y_pred[0])

y_pred = pd.DataFrame(y_pred,columns=['Label'])
y_pred.head()
imageId = pd.Series(range(1,(test.shape[0]+1)),name = 'ImageId')
result = pd.concat([imageId,y_pred],axis = 1)

result.to_csv("cnn_digitRecognizer.csv",index = False)
