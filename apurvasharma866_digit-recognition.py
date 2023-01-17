# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
# read train 
train = pd.read_csv("../input/digit-recognizer/train.csv")
print("Shape of train data : " ,train.shape)
print("Train data :")

train.head()
# read test 
test= pd.read_csv("../input/digit-recognizer/test.csv")
print("Shape of test data : " ,test.shape)
print("Test data :")

test.head()
x_train, x_val, y_train, y_val = train_test_split(train.drop("label",axis=1) , train["label"], test_size=0.1)
# visualize number of digits classes
import seaborn as sns
plt.figure(figsize=(15,7))
g = sns.countplot(y_train, palette="icefire")
plt.title("Number of digit classes")
y_train.value_counts()
fig= plt.figure( figsize=(30,6))
plt.plot(x_train.iloc[0])
plt.title('784x1 data')
# plot some samples
fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(ncols=2,nrows=2 , figsize=(14,14))
img = x_train.iloc[0].to_numpy()
img = img.reshape((28,28))
ax1.imshow(img,cmap='gray')

img = x_train.iloc[1].to_numpy()
img = img.reshape((28,28))
ax2.imshow(img,cmap='gray')

img = x_train.iloc[28].to_numpy()
img = img.reshape((28,28))
ax3.imshow(img,cmap='gray')

img = x_train.iloc[30].to_numpy()
img = img.reshape((28,28))
ax4.imshow(img,cmap='gray')

# Normalize the data
x_train = x_train / 255.0
test = test / 255.0
print("x_train shape: ",x_train.shape)
print("test shape: ",test.shape)
# Reshape
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape: ",x_train.shape)
print("test shape: ",test.shape)
# Label Encoding 
from keras.utils.np_utils import to_categorical 
# convert to one-hot-encoding
y_train = to_categorical(y_train, num_classes = 10)
y_train
# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.33, random_state=2)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)
# data augmentation
datagen = ImageDataGenerator( 
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False, # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,   # dimesion reduction
        rotation_range=0.5,    # randomly rotate images in the range 5 degrees
        zoom_range = 0.5,      # Randomly zoom image 5%
        width_shift_range=0.5, # randomly shift images horizontally 5%
        height_shift_range=0.5,# randomly shift images vertically 5%
        horizontal_flip=False, # randomly flip images
        vertical_flip=False)   # randomly flip images

datagen.fit(X_train)
# plot some samples
fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(ncols=2,nrows=2 , figsize=(14,14))
img = X_train[0]
img = img.reshape((28,28))
ax1.imshow(img,cmap='gray')

img = X_train[1]
img = img.reshape((28,28))
ax2.imshow(img,cmap='gray')

img = X_train[280]
img = img.reshape((28,28))
ax3.imshow(img,cmap='gray')

img = X_train[200]
img = img.reshape((28,28))
ax4.imshow(img,cmap='gray')
# Conv2D -> MaxPool2D -> Dropout -> Conv2D -> MaxPool2D -> Dropout 
# -> FULLY CONNECTED LAYER(Flatten -> Dense -> Dropout -> Dense)

 
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
#
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected layer 
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
earlystop = EarlyStopping(monitor='loss',patience=2,verbose=0,mode='min')
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=86),
                              epochs = 15, validation_data = (X_val,Y_val), 
                              steps_per_epoch=X_train.shape[0]//86 ,callbacks = [learning_rate_reduction] )
loss = pd.DataFrame(model.history.history)
print("Model History :")
loss
loss[['val_loss',"loss"]]
# Plot the loss curves for training and validation 
loss[['val_loss',"loss"]].plot()
plt.title("Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# Plot the accuracy curves for training and validation 
loss[['val_accuracy',"accuracy"]].plot()
plt.title("Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(14,14))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="icefire",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
# Display some error results 

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize = (13,13))
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)),cmap= "gray")
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

# Right are difference between predicted labels and true labels
right = (Y_pred_classes - Y_true == 0)

Y_pred_classes_errors = Y_pred_classes[right]
Y_pred_errors = Y_pred[right]
Y_true_errors = Y_true[right]
X_val_errors = X_val[right]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize = (13,13))
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)),cmap= "gray")
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)