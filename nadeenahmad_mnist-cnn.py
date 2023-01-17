import pandas as pd 
import numpy as np
train= pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# print a sample (first picture) of training set 
train.head()


print ("Total number of images in the training set, Number of pixels length & width per image")
print(train.shape)
# we notice the training set has an extra column they are supposed to be 784 only therefore we need to drop
# this column with the header "label" which is supposed to be the Y_train column used to test our model to test
# its accuracy
print ("Total number of images in the test set, Number of pixels length & width per image")
print(test.shape)

#dropping the label column 
training_data = train.drop('label', axis=1)
#save the label column in a seperate variable to use it later as Y_train
label = train[['label']]
print("Extra column successfully dropped", training_data.shape)
print("Label column was saved ", label.shape)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.1)
#Now lets get at a look at the split data used for training
X_train.shape , y_train.shape

#Now lets get at a look at the split data used for testing
X_test.shape , y_test.shape



#restrict classes of y train to 10 classes 0 to 9 
from keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

from keras.models import Sequential
CNNmodel = Sequential()

from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D
# the input is in format [length][width][pixel] so we must explicitly imply that in the function 
# because our layer expect the input in [pixels][width][height] format
CNNmodel.add(Conv2D(128, (3,3), padding='same', input_shape=(28,28,1), data_format='channels_last', activation='relu'))
# Layer two
from keras.layers import MaxPooling2D
CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
#Layer three the dropout layer 
from keras.layers import Dropout
CNNmodel.add(Dropout(0.2))
#Layer four
#Then a fully connected layer with 128 neurons and rectifier activation function.
CNNmodel.add(Dense(128, activation='relu'))
#layer Five
#Flatten
from keras.layers import  Flatten
CNNmodel.add(Flatten())
#Finally, the output layer has 10 neurons for the 10 classes
#and a softmax activation function to output probability-like predictions for each class:
# First argument given to dense function represents the number of classes = 10
CNNmodel.add(Dense(10, activation='softmax'))

# Compile model
CNNmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#Normalize values of x to range from 0 to 1 instead of from 0 to 255 
New_x = X_train.values.reshape(X_train.shape[0],28,28,1) / 255
print(New_x.shape)


CNNmodel.fit(x=New_x, y=y_train, batch_size=1000, epochs=32, verbose=1, validation_split=0.2)


# Final evaluation of the model
New_xTest = X_test.values.reshape(X_test.shape[0],28,28,1) / 255
scores = CNNmodel.evaluate(New_xTest, y_test, verbose=0)
print("Percentage of Error in our model " , (100-scores[1]*100) ," %")

Accuracy = CNNmodel.evaluate(verbose=2, x=New_xTest, y=y_test)
print ("The previously developed model has " , Accuracy[1]*100 ," % accuracy")

