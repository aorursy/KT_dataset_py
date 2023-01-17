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
import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras import backend as K

import matplotlib.pyplot as plt

from IPython.display import SVG

from keras.utils.vis_utils import plot_model



img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)

num_classes = 10

batch_size = 32

epochs= 12
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

x_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train.head()



x_train = df.loc[:,'pixel0':]

y_train = df['label']



x_train = x_train/255.0

x_test = x_test/255.0



print(x_train.shape)

print(x_test.shape)
# Let's make some beautiful plots.

digit_array = train.loc[3, "pixel0":]

arr = np.array(digit_array) 



#.reshape(a, (28,28))

image_array = np.reshape(arr, (28,28))



digit_img = plt.imshow(image_array, cmap=plt.cm.binary)

plt.colorbar(digit_img)

print("IMAGE LABEL: {}".format(train.loc[3, "label"]))
from keras.utils.np_utils import to_categorical



# Converting dataframe into arrays

mnist_train_data = np.array(x_train)

mnist_train_label = np.array(y_train)

mnist_test_data = np.array(x_test)



mnist_train_label = to_categorical(mnist_train_label, num_classes = num_classes)
# Reshaping the input shapes to get it in the shape which the model expects to recieve later.

mnist_train_data = mnist_train_data.reshape(mnist_train_data.shape[0], 28, 28, 1)



x_test_data = mnist_test_data.reshape(mnist_test_data.shape[0], 28, 28, 1)

print(mnist_train_data.shape, mnist_train_label.shape, x_test_data.shape)

# Define a CNN model that takes several inputs. 

# Having a parameterized CNN model helps in performing several experiments

# by chaning the feature maps, kernel size and by adding a dropout layer



def build_cnn_model_with_max_poooling(prefix='', feature1 = 6, feature2 = 16, kernel_size = 3, add_dropout= False, dropout_rate = 0.0):

  # Build a sequential neural network model that has two convolutional 

  # layers and 2 max pooling layers

  model = Sequential()

  model.add(Conv2D(feature1, kernel_size=(kernel_size, kernel_size),

                  activation='relu',

                  input_shape=input_shape))

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(feature2, (kernel_size, kernel_size), activation='relu'))

  model.add(MaxPooling2D(pool_size=(2, 2)))



  # After conv and max pooling layers the output is flattened and connected 

  # to a couple of dense layers

  model.add(Flatten())

  model.add(Dense(120, activation='relu'))



  # Add a dropout layer

  if add_dropout:

    model.add(Dropout(dropout_rate))

  model.add(Dense(84, activation='relu'))



  # This is the final layer of the model that uses softmax and is responsible to

  # predict the class labels for the input images

  model.add(Dense(num_classes, activation='softmax'))



  # Plot the model using Keras vis_utils 

  plot_model(model, to_file= prefix + 'model_plot.png', show_shapes=True, show_layer_names=True)



  # https://keras.io/optimizers/ 

  # 

  model.compile(loss=keras.losses.categorical_crossentropy,

                optimizer=keras.optimizers.Adadelta(lr=0.1, rho=0.95, decay=0.0),

                metrics=['accuracy'])

  return model
# This function plots the graphs for model loss and accuracy per epoch

# It also saves the output to a file

def plot_graphs(prefix, history):

  # Plot the graphs



  # list all data in history

  print(history.history.keys())



  # Plot the graph of accuracy vs epoch. 

  # The graph shows the training accuracy in blue and the testing accurancy in orange.

  plt.plot(history.history['accuracy'])

  plt.plot(history.history['val_accuracy'])

  plt.title('model accuracy')

  plt.ylabel('accuracy')

  plt.xlabel('epoch')

  plt.legend(['train', 'test'], loc='upper left')

  plt.savefig(prefix + 'model_accuracy.png')

  plt.show()



  # Plot the graph of model loss vs epoch

  # The graph shows the training accuracy in blue and the testing accurancy in orange.

  plt.plot(history.history['loss'])

  plt.plot(history.history['val_loss'])

  plt.title('model loss')

  plt.ylabel('loss')

  plt.xlabel('epoch')

  plt.legend(['train', 'test'], loc='upper left')

  plt.savefig(prefix + 'model_loss.png')

  plt.show()
# Train and evalutate model.

# This method takes the neural network as input and does the following

# - trains the model using training data

# - plots the graphs for loss and accuracy

# - Runs predictions on the test set

# - Prints the testing loss and accuracy

def train_and_evaluate_model(prefix, model):



  # Print model summary

  model.summary()



  # Train the model using training data

  history = model.fit(mnist_train_data, mnist_train_label,

            batch_size=batch_size,

            epochs=epochs,

            verbose=1,

            validation_split=0.2)



  # Plot the graphs using matplotlib

  plot_graphs(prefix, history)

  

  # Evaluate the model using test data



  # Report test loss and accuracy

  print('Test loss:', score[0])

  print('Test accuracy:', score[1])
# Take the baseline model and use feature maps of 64 and 64 respectively. 

# Keep using kernel size 5 as it performed better than kernel size 3

model = build_cnn_model_with_max_poooling('experiment5_', 64, 64, 5)

train_and_evaluate_model('experiment5_', model)
predictions = model.predict(x_test_data)
predictions_arr = []



for i in predictions:

    predictions_arr.append(np.argmax(i))



submission =  pd.DataFrame({

        "ImageId": x_test.index+1,

        "Label": predictions_arr

    })



submission.to_csv('my_submission.csv', index=False)