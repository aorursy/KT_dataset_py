#USING A SAMPLE OF MNIST DIGITS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
# import geojson
import warnings
warnings.filterwarnings('ignore')

from keras.utils import np_utils

import os
%matplotlib inline
plt.style.use('fivethirtyeight') #style
from IPython.display import display

pd.set_option('display.max_columns', 500)
print(os.listdir("../input"))

trainData = pd.read_csv('../input/digits_train.csv')
testData = pd.read_csv('../input/digits_test.csv')
submission = pd.read_csv('../input/digits_sample_submission.csv')
trainData.head()
testData.head()
train_df = trainData.drop(['Unnamed: 0'], axis=1)
test_df = testData.drop(['Unnamed: 0'], axis=1)
test_df = np.array(test_df)
test_df

X_train=train_df.iloc[:, 0:-1].values
y_train=train_df.iloc[:, -1].values

print("Size of training data: {}".format(X_train.shape))
print("Size of test data {}".format(test_df.shape))
print("Size of a single entry in X_train {}".format(X_train[:1].shape))
# Next, we will normalize the values to be from 0 to 1
# normalize pixel values from 0-255 to 0-1
X_train = X_train.astype('float32')
X_test = test_df.astype('float32')

X_train = X_train / 255
X_test = X_test / 255
print(X_train.shape)
print(X_test.shape)
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
num_classes = y_train.shape[1]

print(num_classes)
print(y_train[1:5])
img_width=28
img_height=28
img_depth=1

plt.figure(figsize=(12,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i].reshape(8,8), cmap='gray', interpolation='none')
    plt.title("Label {}".format((np.where(y_train[i]==1))))
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
X_train = X_train.reshape(X_train.shape[0], 1, 8, 8)
X_test = X_test.reshape(X_test.shape[0], 1, 8, 8)

print("Size of training data: {}".format(X_train.shape))
print("Size of test data {}".format(X_test.shape))
from sklearn.model_selection import train_test_split
X_tr, X_ts, y_tr, y_ts = train_test_split(X_train, y_train, train_size = 0.8,random_state = 42)
def mnist_cnn(input_shape=(1, 8, 8), num_classes=10, lr=1E-3, dropout_rate=0.25):
    
    model = Sequential()
    
    #Layer 1
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #Layer 2
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(124, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    #Activation Layer
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr, decay=1e-6),
                  metrics=['accuracy'])
    
    model.summary()
    
    return model


# create model
# model = mnist_cnn()
model = mnist_cnn(lr=1E-3, dropout_rate=0.4)

#model summary
model.summary()
X_tr.reshape((-1, 1))
# fit the model - 50 epochs
model.fit(X_tr, y_tr, epochs=50, batch_size=200, verbose=2)
scores = model.evaluate(X_ts, y_ts, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

# now to predict the model's accuracy on the test data set
from keras.models import load_model
model.save('digit_model_final.h5')
model=load_model('digit_model_final.h5')
predicted=model.predict_classes(X_test)
expected = y_ts
# from sklearn import  metrics
# print('accurcy :',metrics.accuracy_score(expected, predicted))

# print (np.mean(predicted == expected))head.accuracy

predicted
# trainData.drop(['Unnamed: 0'], axis=1)

submission0 = pd.DataFrame({
        "id": testData["Unnamed: 0"],
        "labels": predicted
    })
submission0
submission0.to_csv('digit_test_labelled_final2.csv', index=False, header=True)
print(submission0.head(10))