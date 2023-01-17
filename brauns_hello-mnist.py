# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Copy the file into the directory which allows writing into it as well

! cp ../input/sample_submission.csv /kaggle/working/sample_submission.csv



# Any results you write to the current directory are saved as output.
# keras is an optimized deep learning library supporting Tensorflow, Theano and CNTK at its backend

import keras

from keras import layers



# Sequential model because the model built here has no skipped connections or inception modules

from keras.models import Sequential



# To choose the model's state when the best validation accuracy is obtained

from keras.callbacks import ModelCheckpoint



# To visualize the structure of the model built, layer by layer

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



# To visualize figures

import matplotlib.pyplot as plt
# Model built by adding layers to the object of Sequential class

def LeNet(input_shape):

    # The sequential model, a linear class of layers, is used because the model is a sequence of layers

    model = Sequential()

    

    # Padding each item of the dataset with 2 columns and 2 rows of zeros for an image to be a (32,32,1) image

    model.add(layers.ZeroPadding2D((2,2)))

    

    

    # Adding the first convolutional layer with 6 filters of size (5,5)

    # and stride_size=1 and activation=relu

    model.add(layers.Conv2D(filters=6, kernel_size=(5,5), strides=1, activation="tanh", input_shape=input_shape))



    

    # Adding the first average pooling layer with pool_size (2,2)

    # and stride_size=2

    model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))



    

    # Adding the second convolutional layer with 16 filters of size (5,5)

    # and stride_size=1 and activation=relu

    model.add(layers.Conv2D(filters=16, kernel_size=(5,5), strides=1, activation="tanh"))



    

    # Adding the second average pooling layer with pool_size (2,2)

    # and stride_size=2

    model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))



    

    # Unrolling the matrix row-wise into a vector for feeding into a fully connected layer

    model.add(layers.Flatten())



    # Feeding into a fully connected layer containing 120 neurons and activating with relu

    model.add(layers.Dense(units=120, activation="relu"))

    

    # Feeding into the next fully connnected layer containing 84 neurons and activating with relu

    model.add(layers.Dense(units=84, activation="relu"))



    # The output is a fully connnected layer containing 10 neurons 

    # with class probabilities calculated by using softmax activation

    model.add(layers.Dense(units=10, activation="softmax"))



    return model
# Reading the file using pandas read_csv function

df = pd.read_csv(os.path.join('../input', 'train.csv'))

#df.head(5)



# Assigning the columns from index 1 to the end as the 1st column in the training set is the label(digit 0-9)

inputs = df.iloc[:, 1:].values



# Reshaping the input images into a shape accepted by the LeNet5

inputs = np.array([i.reshape((28,28,1)) for i in inputs])

#print("Input shape is", inputs.shape)



# 1st column assigned as label; the label is one-hot encoded as our predictions will be encoded(softmax) similarly

outputs = pd.get_dummies(df.loc[:, 'label']).values

#print("Output shape is", outputs.shape)
# Call to the function with params as accepted by the input_shape paramter of a layer

model = LeNet(inputs.shape[1:])
# Configure the model with an optimization process, an objective and metric(s)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model and validate it over some epochs

epochs = 20

checkpoint = ModelCheckpoint('model.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='max')

his = model.fit(x=inputs, y=outputs, validation_split=0.3, epochs=epochs, callbacks = [checkpoint], verbose=0)
# To visualise the model thus built

SVG(model_to_dot(model).create(prog='dot', format='svg'))
# To give info about the layers used, their configurations

model.summary()
# ind is the index of the state of the model with the best validation accuracy

ind = his.history['val_acc'].index(max(his.history['val_acc']))

print(ind)
# The row with the best value of validation accuracy

keys = ['val_acc', 'val_loss', 'acc', 'loss']

a = {str(key): his.history[key][ind] for key in keys}

print(a)
# Test the model built, on the training set itself

plt.subplots(2, 3, figsize=(20, 10))

for i in range(6):

    # Choose any example

    a = np.random.randint(0, 42000, 1)[0]

    pred = model.predict(np.array([inputs[a]]))

    plt.subplot(2,3, i+1)

    plt.imshow(inputs[a].reshape((28,28)))

    plt.title("Predicted label: {}\n Actual label: {}".format(np.argmax(pred, axis=1)[0], df.loc[:,'label'][a]), fontsize=16)

plt.subplots_adjust(hspace=0.7, wspace=0.4)
# Test the model over the test data

test_df = pd.read_csv(os.path.join('../input', 'test.csv'))



test_inputs = test_df.values



test_inputs = np.array([i.reshape((28,28,1)) for i in test_inputs])

test_inputs.shape
pred_proba = model.predict(test_inputs)

pred_labels = np.argmax(pred_proba, axis=1)

print(pred_labels[:10])
# To store the values in the prescribed format

submission_df = pd.read_csv(os.path.abspath('sample_submission.csv'))

pred_label_series = pd.Series(pred_labels, dtype='int32')

submission_df['Label'] = pred_label_series



submission_df = submission_df.set_index(['ImageId'])



print(submission_df.head(10))
# Plot figures and find out 

plt.subplots(2, 3, figsize=(20, 10))

for i in range(6):

    # Choose any example

    a = np.random.randint(0, 28000, 1)[0]

    pred = model.predict(np.array([test_inputs[a]]))

    plt.subplot(2,3, i+1)

    plt.imshow(test_inputs[a].reshape((28,28)))

    plt.title("Predicted label: {}".format(np.argmax(pred, axis=1)[0]), fontsize=16)

plt.subplots_adjust(hspace=0.7, wspace=0.4)
# Writing into the submission file

submission_df.to_csv(os.path.abspath('sample_submission.csv'))
# Analysing the training and validation accuracies 



plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.plot(range(20), his.history['acc'], label='Train accuracy')

plt.plot(range(20), his.history['val_acc'], label='Validation accuracy')

plt.legend(loc='best')

plt.grid()

plt.show()



# Analysing the training and validation losses



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.plot(range(20), his.history['loss'], label='Train loss')

plt.plot(range(20), his.history['val_loss'], label='Validation loss')

plt.legend(loc='best')

plt.grid()

plt.show()