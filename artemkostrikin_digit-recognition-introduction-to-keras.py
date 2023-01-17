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
# Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
mnist_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
mnist_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
# divide the data into input and output for training, so that the model learns from what is needed and what needs to be discarded
mnist_train_data = mnist_train.loc[:, 'pixel0':]
mnist_train_label = mnist_train.loc[:, 'label']

# There are many ways to normalize data. The most common is zero-mean normalization, for which the new value W’ = (W – mean) / std. Subtracting the mean of the set and dividing by standard deviation. Such normalization is usually used when the minimum and the maximum values in the set are not specified. But we know what the minimum (0) and the maximum (255) values are, so we can apply “min-max” normalization. The formula for this normalization is quite extensive, although simple – I encourage you to search for it. 
# However, for min = 0 and max = 255, we can simplify this formula significantly and simply divide the value of each pixel by the maximum value, i.e. 255

mnist_train_data = mnist_train_data / 255.0
mnist_test = mnist_test / 255.0
# Check our data
digits = mnist_train.loc[8, 'pixel0':]
array = np.array(digits) 

# reshape(a, (28,28))
image_array = np.reshape(array, (28,28))

digit_image = plt.imshow(image_array, cmap=plt.cm.binary)
plt.colorbar(digit_image)
print('Image Label: {}'.format(mnist_train.loc[8, 'label']))
# Check our data
digits = mnist_train.loc[9, 'pixel0':]
array = np.array(digits) 

# reshape(a, (28,28))
image_array = np.reshape(array, (28,28))

digit_image = plt.imshow(image_array, cmap=plt.cm.binary)
plt.colorbar(digit_image)
print('Image Label: {}'.format(mnist_train.loc[9, 'label']))
# Let's count the number of our labels
sns.countplot(mnist_train.label)
print(list(mnist_train.label.value_counts().sort_index()))
# Converting our data to an array
mnist_train_data = np.array(mnist_train_data)
mnist_train_label = np.array(mnist_train_label)
# Reshape the input shapes to give them the shape the model expects in the future
mnist_train_data = mnist_train_data.reshape(mnist_train_data.shape[0], 28, 28, 1)
print(mnist_train_data.shape, mnist_train_label.shape)
# TensorFlow is a library from Google, we use it to create our model
# and also use the keras package
# Remember keras runs on top of Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D
from tensorflow.keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# When the accuracy or loss starts to plateau during training we can implement the following callbacks 
# to lower the learning rate and hence make smaller steps as it gets closer to the global optimum
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
nclasses = mnist_train_label.max() - mnist_train_label.min() + 1
mnist_train_label = to_categorical(mnist_train_label, num_classes = nclasses)
print('Shape of y_train after encoding: ', mnist_train_label.shape)
# Let's prepare our model for training
def build_model(input_shape=(28, 28, 1)):
    model = Sequential() 
    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = input_shape)) # First 2D Convolutional layer
    model.add(BatchNormalization()) # Activation is Rectified Linear Unit of ReLU for all layers
    model.add(Conv2D(32, kernel_size = 3, activation='relu')) # Batch Normalization is used along with Dropout
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # Dropout Regularization of 0.4 in order to avoid overfitting
    model.add(Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size = 4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax')) # Softmax activation used as this is a multiclass classification task
    return model # The number of units is 10 as there are 10 different classes of digits
def compile_model(model, optimizer='adam', 
                  loss='categorical_crossentropy'):
    
    model.compile(optimizer=optimizer, 
                  loss=loss, metrics=['accuracy']) # using adam optimization, RMSProp works fine too
    # Categorical crossentropy is used as the multiclass loss
    
def train_model(model, train, test, epochs, split):
    history = model.fit(train, test, shuffle=True, epochs=epochs, validation_split=split)
    return history # Data is shuffled during training to avoid inherent bias to the sequence of occurence of an image
# Let's start training our model, I will use 80 Epochs
cnn_model = build_model((28, 28, 1)) # The input is an image odf size 28 X 28
compile_model(cnn_model, 'adam', 'categorical_crossentropy')

# You can use as many eras as you like
# But during the test I found that training above 100 Epochs will not lead to effectiveness
model_history = train_model(cnn_model, mnist_train_data, mnist_train_label, 100, 0.2)
def plot_model_performance(metric, validations_metric):
    plt.plot(model_history.history[metric],label = str('Training ' + metric))
    plt.plot(model_history.history[validations_metric],label = str('Validation ' + metric))
    plt.legend()
# Plotting the loss
plot_model_performance('loss', 'val_loss')
# We change the shape of the test arrays, we have already done this, at the very beginning
mnist_test_arr = np.array(mnist_test)
mnist_test_arr = mnist_test_arr.reshape(mnist_test_arr.shape[0], 28, 28, 1)
print(mnist_test_arr.shape)
# Now, since the model is trained, it's time to find the results for the unseen test images.
predictions = cnn_model.predict(mnist_test_arr)
# Let's print forecasts
print(predictions)
# Let's use the argmax function and choose a random number by index
print(np.argmax(predictions[4]))
# Let's execute the forecast
# Using the (squeeze) function to remove one dimension from a tensor shape
plt.imshow((tf.squeeze(mnist_test_arr[4])), cmap=plt.cm.binary)
plt.show()
# Finally, making the final submissions
predictions_test = []

for i in predictions:
    predictions_test.append(np.argmax(i))
# Create a csv file with the result
submission =  pd.DataFrame({
        "ImageId": mnist_test.index+1,
        "Label": predictions_test
    })

submission.to_csv('submission.csv', index=False)