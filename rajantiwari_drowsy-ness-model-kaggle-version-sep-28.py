# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import numpy as np

import cv2

import os

from scipy.spatial import distance

import tensorflow as tf

from tensorflow import keras 

import numpy as np

import PIL

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense,Activation,Flatten,Dropout

from keras.layers import Conv2D,MaxPooling2D

from keras.callbacks import ModelCheckpoint

#import imutils

#from imutils import face_utils



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_path='/kaggle/input/drowsiness-data/kaggle Upload/'

categories=os.listdir(data_path)

labels=[i for i in range(len(categories))]



label_dict=dict(zip(categories,labels)) #empty dictionary



print(label_dict)

print(categories)

print(labels)
img_size=100

data=[]

target=[]





for category in categories:

    folder_path=os.path.join(data_path,category)

    img_names=os.listdir(folder_path)

        

    for img_name in img_names:

        img_path=os.path.join(folder_path,img_name)

        img=cv2.imread(img_path)



        try:

            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           

            #Coverting the image into gray scale

            resized=cv2.resize(gray,(img_size,img_size))

            #resizing the gray scale into 50x50, since we need a fixed common size for all the images in the dataset

            data.append(resized)

            target.append(label_dict[category])

            #appending the image and the label(categorized) into the list (dataset)



        except Exception as e:

            print('Exception:',e)

            #if any exception rasied, the exception will be printed here. And pass to the next image
import numpy as np



data=np.array(data)/255.0

data=np.reshape(data,(data.shape[0],img_size,img_size,1))

target=np.array(target)



from keras.utils import np_utils



new_target=np_utils.to_categorical(target)


np.save('data',data)

np.save('target',new_target)

import numpy as np



data=np.load('data.npy')

target=np.load('target.npy')



#loading the save numpy arrays in the previous code
from keras.models import Sequential

from keras.layers import Dense,Activation,Flatten,Dropout

from keras.layers import Conv2D,MaxPooling2D

from keras.callbacks import ModelCheckpoint



model=Sequential()



model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

#The first CNN layer followed by Relu and MaxPooling layers



model.add(Conv2D(100,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

#The second convolution layer followed by Relu and MaxPooling layers



model.add(Flatten())

model.add(Dropout(0.5))

#Flatten layer to stack the output convolutions from second convolution layer

model.add(Dense(64,activation='relu'))

#Dense layer of 64 neurons

model.add(Dense(2,activation='softmax'))

#The Final layer with two outputs for two categories



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from sklearn.model_selection import train_test_split



train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)


from matplotlib import pyplot as plt



plt.plot(history.history['loss'],'r',label='training loss')

plt.plot(history.history['val_loss'],label='validation loss')

plt.xlabel('# epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
plt.plot(history.history['accuracy'],'r',label='training accuracy')

plt.plot(history.history['val_accuracy'],label='validation accuracy')

plt.xlabel('# epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
import glob

import matplotlib

import tensorflow as tf

from matplotlib import pyplot as plt

import matplotlib.image as mpimg

import numpy as np

import imageio as im

from keras.models import load_model

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint
ls /kaggle/working
model =load_model('model-020.model')

# As a reminder 

model.summary() 
from tensorflow.keras.utils import plot_model

from PIL import Image  

import PIL  

# summarize the model

x=plot_model(model, 'model.png', show_shapes=True)

im1 = model.save("/kaggle/working/") 
ls /kaggle/working
import os

os.chdir(r'../working')

from IPython.display import FileLink

FileLink(r'model.png')
from tensorflow.keras import models

layer_outputs = [layer.output for layer in model.layers[:12]] 

# Extracts the outputs of the top 12 layers

activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
img_path = '/kaggle/input/drowsiness-data/kaggle Upload/alert/alert10.jpg'

img = image.load_img(img_path, target_size=(64, 64))

img_tensor = image.img_to_array(img)

img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.0

plt.imshow(img_tensor[0])

plt.show()

print(img_tensor.shape)

activations = activation_model.predict(img_tensor) 

# Returns a list of five Numpy arrays: one array per layer activation
from keract import display_activations
# Utility functions for deep learning with Keras

# Dr. Tirthajyoti Sarkar, Fremont, CA 94536

# ==============================================



# NOTES

# Used tf.keras in general except in special functions where older/independent Keras has been used.



import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt





class myCallback(tf.keras.callbacks.Callback):

    """

  User can pass on the desired accuracy threshold while creating an instance of the class

  """



    def __init__(self, acc_threshold=0.9, print_msg=True):

        self.acc_threshold = acc_threshold

        self.print_msg = print_msg



    def on_epoch_end(self, epoch, logs={}):

        if logs.get("acc") > self.acc_threshold:

            if self.print_msg:

                print(

                    "\nReached {}% accuracy so cancelling the training!".format(

                        self.acc_threshold

                    )

                )

            self.model.stop_training = True

        else:

            if self.print_msg:

                print("\nAccuracy not high enough. Starting another epoch...\n")



    def build_classification_model(

        num_layers=1,

        architecture=[32],

        act_func="relu",

        input_shape=(28, 28),

        output_class=10,

    ):

        """

  Builds a densely connected neural network model from user input

  

  Arguments

          num_layers: Number of hidden layers

          architecture: Architecture of the hidden layers (densely connected)

          act_func: Activation function. Could be 'relu', 'sigmoid', or 'tanh'.

          input_shape: Dimension of the input vector

          output_class: Number of classes in the output vector

  Returns

          A neural net (Keras) model for classification

  """

        layers = [tf.keras.layers.Flatten(input_shape=input_shape)]

        if act_func == "relu":

            activation = tf.nn.relu

        elif act_func == "sigmoid":

            activation = tf.nn.sigmoid

        elif act_func == "tanh":

            activation = tf.nn.tanh



        for i in range(num_layers):

            layers.append(tf.keras.layers.Dense(architecture[i], activation=tf.nn.relu))

        layers.append(tf.keras.layers.Dense(output_class, activation=tf.nn.softmax))



        model = tf.keras.models.Sequential(layers)

        return model





def build_regression_model(

    input_neurons=10, input_dim=1, num_layers=1, architecture=[32], act_func="relu"

):

    """

  Builds a densely connected neural network model from user input

  

  Arguments

          num_layers: Number of hidden layers

          architecture: Architecture of the hidden layers (densely connected)

          act_func: Activation function. Could be 'relu', 'sigmoid', or 'tanh'.

          input_shape: Dimension of the input vector

  Returns

          A neural net (Keras) model for regression

  """

    if act_func == "relu":

        activation = tf.nn.relu

    elif act_func == "sigmoid":

        activation = tf.nn.sigmoid

    elif act_func == "tanh":

        activation = tf.nn.tanh



    layers = [

        tf.keras.layers.Dense(input_neurons, input_dim=input_dim, activation=activation)

    ]



    for i in range(num_layers):

        layers.append(tf.keras.layers.Dense(architecture[i], activation=activation))

    layers.append(tf.keras.layers.Dense(1))



    model = tf.keras.models.Sequential(layers)

    return model





def compile_train_classification_model(

    model,

    x_train,

    y_train,

    callbacks=None,

    learning_rate=0.001,

    batch_size=1,

    epochs=10,

    verbose=0,

):

    """

  Compiles and trains a given Keras model with the given data. 

  Assumes Adam optimizer for this implementation.

  Assumes categorical cross-entropy loss.

  

  Arguments

          learning_rate: Learning rate for the optimizer Adam

          batch_size: Batch size for the mini-batch optimization

          epochs: Number of epochs to train

          verbose: Verbosity of the training process

  

  Returns

  A copy of the model

  """



    model_copy = model

    model_copy.compile(

        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"],

    )



    if callbacks != None:

        model_copy.fit(

            x_train,

            y_train,

            epochs=epochs,

            batch_size=batch_size,

            callbacks=[callbacks],

            verbose=verbose,

        )

    else:

        model_copy.fit(

            x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose

        )

    return model_copy





def compile_train_regression_model(

    model,

    x_train,

    y_train,

    callbacks=None,

    learning_rate=0.001,

    batch_size=1,

    epochs=10,

    verbose=0,

):

    """

  Compiles and trains a given Keras model with the given data for regression. 

  Assumes Adam optimizer for this implementation.

  Assumes mean-squared-error loss

  

  Arguments

          learning_rate: Learning rate for the optimizer Adam

          batch_size: Batch size for the mini-batch operation

          epochs: Number of epochs to train

          verbose: Verbosity of the training process

  

  Returns

  A copy of the model

  """



    model_copy = model

    model_copy.compile(

        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),

        loss="mean_squared_error",

        metrics=["accuracy"],

    )

    if callbacks != None:

        model_copy.fit(

            x_train,

            y_train,

            epochs=epochs,

            batch_size=batch_size,

            callbacks=[callbacks],

            verbose=verbose,

        )

    else:

        model_copy.fit(

            x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose

        )

    return model_copy





def plot_loss_acc(model, target_acc=0.9, title=None):

    """

  Takes a Keras model and plots the loss and accuracy over epochs.

  The same plot shows loss and accuracy on two axes - left and right (with separate scales)

  Users can supply a title if desired

  Arguments:

            target_acc (optional): The desired/ target acc for the function to show a horizontal bar.

            title (optional): A Python string object to show as the plot's title

  """

    e = (

        np.array(model.history.epoch) + 1

    )  # Add one to the list of epochs which is zero-indexed

    # Check to see if loss metric is in the model history

    assert (

        "loss" in model.history.history.keys()

    ), "No loss metric found in the model history"

    l = np.array(model.history.history["loss"])

    # Check to see if loss metric is in the model history

    assert (

        "acc" in model.history.history.keys()

    ), "No accuracy metric found in the model history"

    a = np.array(model.history.history["acc"])



    fig, ax1 = plt.subplots()



    color = "tab:red"

    ax1.set_xlabel("Epochs", fontsize=15)

    ax1.set_ylabel("Loss", color=color, fontsize=15)

    ax1.plot(e, l, color=color, lw=2)

    ax1.tick_params(axis="y", labelcolor=color)

    ax1.grid(True)



    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



    color = "tab:blue"

    ax2.set_ylabel(

        "Accuracy", color=color, fontsize=15

    )  # we already handled the x-label with ax1

    ax2.plot(e, a, color=color, lw=2)

    ax2.tick_params(axis="y", labelcolor=color)



    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if title != None:

        plt.title(title)

    plt.hlines(

        y=target_acc, xmin=1, xmax=e.max(), colors="k", linestyles="dashed", lw=3

    )

    plt.show()





def plot_train_val_acc(model, target_acc=0.9, title=None):

    """

  Takes a Keras model and plots the training and validation set accuracy over epochs.

  The same plot shows both the accuracies on two axes - left and right (with separate scales)

  Users can supply a title if desired

  Arguments:

            target_acc (optional): The desired/ target acc for the function to show a horizontal bar.

            title (optional): A Python string object to show as the plot's title

  """

    e = (

        np.array(model.history.epoch) + 1

    )  # Add one to the list of epochs which is zero-indexed

    # Check to see if loss metric is in the model history

    assert (

        "acc" in model.history.history.keys()

    ), "No accuracy metric found in the model history"

    a = np.array(model.history.history["acc"])

    # Check to see if loss metric is in the model history

    assert (

        "val_acc" in model.history.history.keys()

    ), "No validation accuracy metric found in the model history"

    va = np.array(model.history.history["val_acc"])



    fig, ax1 = plt.subplots()



    color = "tab:red"

    ax1.set_xlabel("Epochs", fontsize=15)

    ax1.set_ylabel("Training accuracy", color=color, fontsize=15)

    ax1.plot(e, a, color=color, lw=2)

    ax1.tick_params(axis="y", labelcolor=color)

    ax1.grid(True)



    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



    color = "tab:blue"

    ax2.set_ylabel(

        "Validation accuracy", color=color, fontsize=15

    )  # we already handled the x-label with ax1

    ax2.plot(e, va, color=color, lw=2)

    ax2.tick_params(axis="y", labelcolor=color)



    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if title != None:

        plt.title(title)



    plt.hlines(

        y=target_acc, xmin=1, xmax=e.max(), colors="k", linestyles="dashed", lw=3

    )



    plt.show()





def train_CNN(

    train_directory,

    target_size=(256, 256),

    callbacks=None,

    classes=None,

    batch_size=128,

    num_classes=2,

    num_epochs=20,

    verbose=0,

):

    """

    Trains a conv net for a given dataset contained within a training directory.

    Users can just supply the path of the training directory and get back a fully trained, 5-layer, convolutional network.

    

    Arguments:

            train_directory: The directory where the training images are stored in separate folders.

                            These folders should be named as per the classes.

            target_size: Target size for the training images. A tuple e.g. (200,200)

            classes: A Python list with the classes 

            batch_size: Batch size for training

            num_epochs: Number of epochs for training

            num_classes: Number of output classes to consider

            verbose: Verbosity level of the training, passed on to the `fit_generator` method

    Returns:

            A trained conv net model

    

    """

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    import tensorflow as tf

    from tensorflow.keras.optimizers import RMSprop



    # ImageDataGenerator object instance with scaling

    train_datagen = ImageDataGenerator(rescale=1 / 255)



    # Flow training images in batches using the generator

    train_generator = train_datagen.flow_from_directory(

        train_directory,  # This is the source directory for training images

        target_size=target_size,  # All images will be resized to 200 x 200

        batch_size=batch_size,

        # Specify the classes explicitly

        classes=classes,

        # Since we use categorical_crossentropy loss, we need categorical labels

        class_mode="categorical",

    )



    input_shape = tuple(list(target_size) + [3])



    # Model architecture

    model = tf.keras.models.Sequential(

        [

            # Note the input shape is the desired size of the image 200x 200 with 3 bytes color

            # The first convolution

            tf.keras.layers.Conv2D(

                16, (3, 3), activation="relu", input_shape=input_shape

            ),

            tf.keras.layers.MaxPooling2D(2, 2),

            # The second convolution

            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),

            tf.keras.layers.MaxPooling2D(2, 2),

            # The third convolution

            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

            tf.keras.layers.MaxPooling2D(2, 2),

            # The fourth convolution

            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

            tf.keras.layers.MaxPooling2D(2, 2),

            # The fifth convolution

            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

            tf.keras.layers.MaxPooling2D(2, 2),

            # Flatten the results to feed into a dense layer

            tf.keras.layers.Flatten(),

            # 512 neuron in the fully-connected layer

            tf.keras.layers.Dense(512, activation="relu"),

            # Output neurons for `num_classes` classes with the softmax activation

            tf.keras.layers.Dense(num_classes, activation="softmax"),

        ]

    )



    # Optimizer and compilation

    model.compile(

        loss="categorical_crossentropy", optimizer=RMSprop(lr=0.001), metrics=["acc"]

    )



    # Total sample count

    total_sample = train_generator.n



    # Training

    model.fit_generator(

        train_generator,

        callbacks=callbacks,

        steps_per_epoch=int(total_sample / batch_size),

        epochs=num_epochs,

        verbose=verbose,

    )



    return model





def train_CNN_keras(

    train_directory,

    target_size=(256, 256),

    classes=None,

    batch_size=128,

    num_classes=2,

    num_epochs=20,

    verbose=0,

):

    """

    Trains a conv net for a given dataset contained within a training directory.

    Users can just supply the path of the training directory and get back a fully trained, 5-layer, convolutional network.

    

    Arguments:

            train_directory: The directory where the training images are stored in separate folders.

                            These folders should be named as per the classes.

            target_size: Target size for the training images. A tuple e.g. (200,200)

            classes: A Python list with the classes 

            batch_size: Batch size for training

            num_epochs: Number of epochs for training

            num_classes: Number of output classes to consider

            verbose: Verbosity level of the training, passed on to the `fit_generator` method

    Returns:

            A trained conv net model

    

    """

    from keras.layers import Conv2D, MaxPooling2D

    from keras.layers import Dense, Dropout, Flatten

    from keras.models import Sequential

    from keras.optimizers import RMSprop

    from keras.preprocessing.image import ImageDataGenerator



    # ImageDataGenerator object instance with scaling

    train_datagen = ImageDataGenerator(rescale=1 / 255)



    # Flow training images in batches using the generator

    train_generator = train_datagen.flow_from_directory(

        train_directory,  # This is the source directory for training images

        target_size=target_size,  # All images will be resized to 200 x 200

        batch_size=batch_size,

        # Specify the classes explicitly

        classes=classes,

        # Since we use categorical_crossentropy loss, we need categorical labels

        class_mode="categorical",

    )



    input_shape = tuple(list(target_size) + [3])



    # Model architecture

    model = Sequential(

        [

            # Note the input shape is the desired size of the image 200x 200 with 3 bytes color

            # The first convolution

            Conv2D(16, (3, 3), activation="relu", input_shape=input_shape),

            MaxPooling2D(2, 2),

            # The second convolution

            Conv2D(32, (3, 3), activation="relu"),

            MaxPooling2D(2, 2),

            # The third convolution

            Conv2D(64, (3, 3), activation="relu"),

            MaxPooling2D(2, 2),

            # The fourth convolution

            Conv2D(64, (3, 3), activation="relu"),

            MaxPooling2D(2, 2),

            # The fifth convolution

            Conv2D(64, (3, 3), activation="relu"),

            MaxPooling2D(2, 2),

            # Flatten the results to feed into a dense layer

            Flatten(),

            # 512 neuron in the fully-connected layer

            Dense(512, activation="relu"),

            # Output neurons for `num_classes` classes with the softmax activation

            Dense(num_classes, activation="softmax"),

        ]

    )



    # Optimizer and compilation

    model.compile(

        loss="categorical_crossentropy", optimizer=RMSprop(lr=0.001), metrics=["acc"]

    )



    # Total sample count

    total_sample = train_generator.n



    # Training

    model.fit_generator(

        train_generator,

        steps_per_epoch=int(total_sample / batch_size),

        epochs=num_epochs,

        verbose=verbose,

    )



    return model





def preprocess_image(img_path, model=None, rescale=255, resize=(256, 256)):

    """

    Preprocesses a given image for prediction with a trained model, with rescaling and resizing options

    

    Arguments:

            img_path: The path to the image file

            rescale: A float or integer indicating required rescaling. 

                    The image array will be divided (scaled) by this number.

            resize: A tuple indicating desired target size. 

                    This should match the input shape as expected by the model

    Returns:

            img: A processed image.

    """

    from keras.preprocessing.image import img_to_array, load_img

    import cv2

    import numpy as np



    assert type(img_path) == str, "Image path must be a string"

    assert (

        type(rescale) == int or type(rescale) == float

    ), "Rescale factor must be either a float or int"

    assert (

        type(resize) == tuple and len(resize) == 2

    ), "Resize target must be a tuple with two elements"



    img = load_img(img_path)

    img = img_to_array(img)

    img = img / float(rescale)

    img = cv2.resize(img, resize)

    if model != None:

        if len(model.input_shape) == 4:

            img = np.expand_dims(img, axis=0)



    return img





def pred_prob_with_model(img_path, model, rescale=255, resize=(256, 256)):

    """

    Tests a given image with a trained model, with rescaling and resizing options

    

    Arguments:

            img_path: The path to the image file

            model: The trained Keras model

            rescale: A float or integer indicating required rescaling. 

                    The image array will be divided (scaled) by this number.

            resize: A tuple indicating desired target size. 

                    This should match the input shape as expected by the model

    Returns:

            pred: A prediction vector (Numpy array).

                  Could be either classes or probabilities depending on the model.

    """

    from keras.preprocessing.image import img_to_array, load_img

    import cv2



    assert type(img_path) == str, "Image path must be a string"

    assert (

        type(rescale) == int or type(rescale) == float

    ), "Rescale factor must be either a float or int"

    assert (

        type(resize) == tuple and len(resize) == 2

    ), "Resize target must be a tuple with two elements"



    img = load_img(img_path)

    img = img_to_array(img)

    img = img / float(rescale)

    img = cv2.resize(img, resize)

    if len(model.input_shape) == 4:

        img = np.expand_dims(img, axis=0)



    pred = model.predict(img)



    return pred





def pred_class_with_model(img_path, model, rescale=255, resize=(256, 256)):

    """

    Tests a given image with a trained model, with rescaling and resizing options

    

    Arguments:

            img_path: The path to the image file

            model: The trained Keras model

            rescale: A float or integer indicating required rescaling. 

                    The image array will be divided (scaled) by this number.

            resize: A tuple indicating desired target size. 

                    This should match the input shape as expected by the model

    Returns:

            pred: A prediction vector (Numpy array).

                  Could be either classes or probabilities depending on the model.

    """

    from keras.preprocessing.image import img_to_array, load_img

    import cv2



    assert type(img_path) == str, "Image path must be a string"

    assert (

        type(rescale) == int or type(rescale) == float

    ), "Rescale factor must be either a float or int"

    assert (

        type(resize) == tuple and len(resize) == 2

    ), "Resize target must be a tuple with two elements"



    img = load_img(img_path)

    img = img_to_array(img)

    img = img / float(rescale)

    img = cv2.resize(img, resize)

    if len(model.input_shape) == 4:

        img = np.expand_dims(img, axis=0)



    pred = model.predict(img)

    pred_class = pred.argmax(axis=-1)



    return pred_class
img_path = '/kaggle/input/drowsiness-data/kaggle Upload/alert/alert376.jpg'

target_size=(64,64)

# Preprocessing the image for the model

x = preprocess_image(img_path=img_path,model=model,resize=target_size)

# Generate the activations 

activations = get_activations(model, x)
x = preprocess_image(img_path=img_path,model=model,resize=target_size)
#!pip install keract==1.1.1
from keract import get_activations

activations = get_activations(model, x)