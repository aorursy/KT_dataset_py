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


X_train = X_train.values.reshape(-1,28,28,1)
testData = testData.values.reshape(-1,28,28,1)
"""
Data Normalization
We perform a grayscale normalization. CNN converges faster on [0.....1] data than on [0......255].
The pixel values are gray scale between 0 and 255.
"""
# Normalize the data
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

# Split the train and the validation set for the fitting
# Set the random seed
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


"""
ReLU is computed after the convolution and therefore a nonlinear activation function like tanh or sigmoid. I choosed to set 32 filters for the two firsts conv2D layers and 64 filters for the two last ones
#Input -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
input _shape = [pixel,width,height]

"""

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_val, Y_val), verbose = 2)
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


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