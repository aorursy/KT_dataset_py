# Import libraries for data preparation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
np.random.seed(2)
# loading training data into pandas dataframe object
train_data = pd.read_csv("../input/train.csv")
train_data.head()
# loading test data into pandas dataframe object
test_data = pd.read_csv("../input/test.csv")
test_data.head()
# Getting training label from training data
Y_train = train_data['label']
X_train = train_data.drop(columns=['label'])
del train_data
print("Shape of training image data"+str(X_train.shape))
print("Shape of training label data"+str(Y_train.shape))
#Visualizing the training label
sns.countplot(Y_train)
# checking missing value in train data
X_train.isnull().any().any()
# checking missing value in test data
test_data.isnull().any().any()
# performing grayscale normalization of test and train data
X_train = X_train/255.0
test_data = test_data/255.0 
#reshaping train and test images to 28*28*1 pixels
X_train = X_train.values.reshape(-1,28,28,1)
test_data = test_data.values.reshape(-1,28,28,1)
# Some examples
index = np.random.randint(42000)
g = plt.imshow(X_train[index][:,:,0])
print("label : "+str(Y_train[index]))
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
# Set the random seed
random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
## Importing keras libraries to model CNN
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
# creating CNN model 
model = Sequential()
model.add(Conv2D(filters = 8, kernel_size = (4,4), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
          
model.add(Conv2D(filters = 16, kernel_size = (2,2), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
          
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dense(10, activation = "softmax"))
# defining an optimizer for the model
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# train the model
batch_size = 100
epochs = 30
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val), verbose = 2)
# datagen = ImageDataGenerator(rotation_range = 20,  # randomly rotate images in the range (degrees, 0 to 180)
#                             zoom_range = 0.2, # Randomly zoom image 
#                             width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
#                             height_shift_range=0.2)
# datagen.fit(X_train)
# Fit the model
# num_iteration = X_train.shape[0]/batch_size
# history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
#                               epochs = epochs, validation_data = (X_val,Y_val),
#                               verbose = 2, steps_per_epoch = num_iteration)
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# predict results
predictions = model.predict(test_data)
predictions = np.argmax(predictions,axis = 1)

predictions = pd.Series(predictions, name = "Label")
image_id = pd.Series(range(1,28001),name = "ImageId")

submission = pd.concat([image_id,predictions],axis = 1)
submission.to_csv("submission.csv",index=False)

submission.head()