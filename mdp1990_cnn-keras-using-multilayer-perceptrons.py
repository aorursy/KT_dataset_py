import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop

from sklearn.metrics import classification_report, confusion_matrix
trainData = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
testData = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
Y_train = trainData['label']
X_train = trainData.drop('label',axis=1)
del trainData
"""
Flatten 28*28 images into 784 vectors of images.
"""
num_pixels = X_train.shape[1]
num_pixels

g = sns.countplot(Y_train)
Y_train.value_counts()
"""
Checking for Missing Values
There is no missing values in the data.
"""
print("Missing Values in Training Data",X_train.isnull().sum())
print("Missing Values in Testing Data",testData.isnull().sum())

"""
Reshape Images
Flatten 28*28 images to a 784 vector for each image
Train and test images (28px x 28px) .Dataframe as 1D vectors of 784 values. 
We reshape all data to 28x28x1 3D matrices.
Keras requires an extra dimension in the end which correspond to channels. 
MNIST images are gray scaled so it use only one channel. 
For RGB images, there is 3 channels, we would have reshaped 784px vectors to 28x28x3 3D matrices.
"""


"""
Data Normalization
We perform a grayscale normalization. CNN converges faster on [0.....1] data than on [0......255].
The pixel values are gray scale between 0 and 255.
"""

X_train = X_train / 255.0
testData = testData / 255.0
"""
Encoding Label:
Output variable is an integer from 0 to 9. 
This is a multi-class classification problem. As such, it is good practice to use a one hot encoding of the 
class values, transforming the vector of class integers into a binary matrix.
We can easily do this using the built-in np_utils.to_categorical() helper function in Keras.
"""

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
num_classes = Y_train.shape[1]
# Split the train and the validation set for the fitting
# Set the random seed
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)



model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history_baselineModel = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=200, verbose=2)
history_baselineModel


# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 

conf_matrix = confusion_matrix(Y_true,Y_pred_classes)
conf_matrix
# predict results
results = model.predict(testData)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("sample_submission.csv",index=False)