import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib.cm as cm  # colormaps

%matplotlib inline



import tensorflow

from tensorflow.python.keras.models import Sequential, Model

from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout

from tensorflow.python.keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Concatenate

from tensorflow.python.keras.optimizers import RMSprop # for optimisation

from tensorflow.python.keras.utils.np_utils import to_categorical # one-hot encoding
train_records = pd.read_csv("../input/train.csv")

test_records = pd.read_csv("../input/test.csv")
y_train = train_records["label"]

X_train = train_records.drop(labels = ["label"],axis = 1)

del train_records 
# we need to reshape the training data to have 4D, since we want each digit to be read as an image

# with only one colour channel

X_train = X_train.values.reshape(-1,28,28,1)

test_records = test_records.values.reshape(-1,28,28,1)

X_train.shape, test_records.shape
def plot_images(images_to_plot, titles=None, ncols=6, thefigsize=(18,18)):

    n_images = images_to_plot.shape[0]

    nrows = np.ceil(n_images/ncols).astype(int)

    

    fig,ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=thefigsize)

    ax = ax.flatten()



    for i in range(n_images):

        ax[i].imshow( images_to_plot[i,:,:,0], cmap=cm.Greys ) 

            # cmap=cm.Greys plots in Grey scale so the image looks as if it were written

        ax[i].axis('off')  

        if titles is not None:

            ax[i].set_title(titles[i])
plot_images( X_train[:36,:,:,:], titles= y_train[:36])
# to use softmax, we need to alter the class representations from integers to 'one-hot' 

y_train = to_categorical(y_train)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1, random_state=2)
# This model is specified using the Keras Sequential API, which allows one to specify a network which is a 

# linear sequence of layers, the output of one being the input of the next

# Please look at the Keras documentation on the API



model_1=Sequential()

model_1.add(Conv2D(6,(3,3),padding='same',input_shape=(28,28,1)))

model_1.add(Activation('relu'))

model_1.add(Conv2D(6,(3,3),padding='same'))  # A depth of only 6 convolutional filters here

model_1.add(Activation('relu'))

model_1.add(MaxPool2D(pool_size=(2,2),strides=None,padding='valid')) 

model_1.add(Dropout(0.25))

# you might want to try adding more convolutions and pooling here 

model_1.add(Flatten()) # reshapes the square array of the image into a vector

model_1.add(Dense(100)) # this is a pure linear transform

model_1.add(BatchNormalization())   

model_1.add(Activation('tanh'))     # Activation is a layer that applies a non-linearity to its inputs

model_1.add(Dense(100))

#model_1.add(BatchNormalization())

model_1.add(Activation('tanh'))

model_1.add(Dense(100))

#model_1.add(BatchNormalization()) # we can drop BatchNormalization into or out of the model at will. 

model_1.add(Activation('tanh'))

model_1.add(Dense(100))

model_1.add(BatchNormalization())

model_1.add(Activation('tanh'))

model_1.add(Dense(100))

#model_1.add(BatchNormalization())

model_1.add(Activation('tanh'))

model_1.add(Dense(10))

model_1.add(Activation('softmax'))
model_1.summary()
model_1.compile(optimizer='RMSprop',loss='categorical_crossentropy', metrics=['accuracy'])
history_1 = model_1.fit( X_train, y_train, epochs=30, batch_size=86, shuffle=True,

                       validation_data=(X_valid,y_valid))
valid_predictions = model_1.predict( X_valid )

valid_predicted_classes = np.argmax(valid_predictions, axis=1)

valid_predicted_classes.shape

y_valid_classes = np.argmax(y_valid,axis=1)
valid_prediction_errors = np.not_equal( valid_predicted_classes, y_valid_classes )
np.sum( valid_prediction_errors)

# around 2% error rate
# lets look at the image that were wrongly classified

error_images = X_valid[ valid_prediction_errors, :, : , :]
#  the images with the (wrong) predicted classification

plot_images( error_images[:36,:,:,:], valid_predicted_classes[valid_prediction_errors])
#  the images with the true classification

plot_images( error_images[:36,:,:,:], y_valid_classes[valid_prediction_errors])
# predicion on test set

predictions = model_1.predict( test_records )
print(predictions[:3])

predicted_classes = np.argmax(predictions, axis=1)

predicted_classes[:3]
predicted_classes = pd.Series(predicted_classes,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predicted_classes],axis = 1)

print(submission)
submission.to_csv("submission.csv",index=False)