import tensorflow as tf

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':

    raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))
# Download the dataset

print('Downloading the dataset...')

!wget -qq https://www.dropbox.com/s/fm2x6fq6xoxk5qi/YDSP_CatsDogs.zip

print('Done!')

  

# Unzip to retrieve the images

print('Unzipping the file...')

!unzip -qq YDSP_CatsDogs.zip

!rm YDSP_CatsDogs.zip

print('Done!')
!ls
# The folder structure 

!ls catsdogs/ 
!ls catsdogs/cat  # Cats Images
# General Libraries

%matplotlib inline

import os

import numpy as np

import pandas as pd

from glob import glob

from sklearn.model_selection import train_test_split



# Image Processing and CNN Libraries

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.optimizers import RMSprop, Adam

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Activation, Dropout, Flatten, Dense



# Visualisation Libraries

import seaborn as sn

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from sklearn.metrics import confusion_matrix

from IPython.display import Image
# Specify the directory of the class folders

path = "./catsdogs"
# Input Size

IMAGE_SIZE = 200

IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE



# Hyperparams

EPOCHS = 30

BATCH_SIZE = 32
# Train-Valid Data Split

def trainValidSplit(path, split_percent):

  """

  Function to perform the train-validation split by a specified validation split percentage.

  @args:

    path: The directory to the class folders.

    split_percent: Percentage of the data to split to form the validation dataset.

  """

  file_names = glob(path + '/**/*')

  for i in file_names:

    path_dir = os.path.dirname(i)

    os.rename(i, os.path.join(path_dir, os.path.splitext(os.path.basename(i))[0] + ".jpg"))

    

  all_classes = [file.split('/')[-2] for file in file_names]

  x_train, x_valid, y_train, y_valid = train_test_split(file_names, all_classes, test_size=split_percent, random_state=42)

  train_df = pd.DataFrame({'filename': x_train, 'class': y_train})

  validation_df = pd.DataFrame({'filename': x_valid, 'class': y_valid})

  return (train_df, validation_df)



# Validation Split

validation_split = 0.2

train_df, validation_df = trainValidSplit(path, validation_split)
# Training dataset

print(train_df.shape)

train_df.head()
# Validation Dataset

print(validation_df.shape)

validation_df.head()
# Training Data Augmentation

training_data_generator = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.1,

    horizontal_flip=True)
validation_data_generator = ImageDataGenerator(

    rescale=1./255)
training_generator = training_data_generator.flow_from_dataframe(train_df,

                                             target_size = (IMAGE_WIDTH, IMAGE_HEIGHT),

                                             batch_size = BATCH_SIZE,

                                             shuffle = False,

                                             class_mode = "categorical")

validation_generator = validation_data_generator.flow_from_dataframe(validation_df,

                                             target_size = (IMAGE_WIDTH, IMAGE_HEIGHT),

                                             batch_size = BATCH_SIZE,

                                             shuffle = False,

                                             class_mode = "categorical")
# Number of classes

num_classes = len(training_generator.class_indices)

classes = sorted(training_generator.class_indices, key=training_generator.class_indices.get)

print('Number of classes: ', num_classes)

print('Class Labels: ', classes)
def plotSample(datagen, image_num = 20, predict_model = None, image_per_col = 10):

    """

    Helper function to plot images along with is class labels and predictions. The class labels & predictions are placed on top

    of the individual image and its corresponding prediction in brackets.

    @args:

        datagen: Image Generator variable that contains the images to visualise.

        image_num: Number of images to visualise. Defaults to 20 images.

        predict_model: The trained model to make predictions if any. Defaults to None (No prediction to be made).

        image_per_col: Number of images to plot for each column.

    """

    from math import ceil

    # Get first batch

    images, labels = datagen.next()

    batch_img_num = images.shape[0]

  

    # Get more batch if no. of images to plot > batch_size

    while batch_img_num < image_num:

        images_a, labels_a = datagen.next()

        images, labels = np.concatenate([images, images_a]), np.concatenate([labels, labels_a])

        batch_img_num += images_a.shape[0]

    

    # Decode one-hot encoding (if any)

    if len(labels.shape) == 2:

        labels = labels.argmax(axis = 1)

  

    # Swap the class name and its numeric representation

    classes = dict((v,k) for k,v in datagen.class_indices.items())

  

    # Make prediction if this is to visualise the model predictive power

    if predict_model:

        preds_prob = predict_model.predict(images)

        preds = preds_prob.argmax(axis = 1)

        main_prob = preds_prob.max(axis=1)

    

    # Plot the images in the batch, along with the corresponding (predictions and) labels

    fig = plt.figure(figsize=(25, ceil(image_num/image_per_col)*2))

    for idx in np.arange(image_num):

        if predict_model and len(preds) <= idx:

            break

        ax = fig.add_subplot(ceil(image_num/image_per_col), image_per_col, idx+1, xticks=[], yticks=[])

        plt.imshow(images[idx])

        if not predict_model:

            ax.set_title(classes[labels[idx]])

        else:

            ax.set_title("{} ({:.2f}% {})".format(classes[labels[idx]], main_prob[idx]*100, classes[preds[idx]]),

                         color=("green" if preds[idx]==labels[idx].item() else "red"))
plotSample(training_generator, 40)
# Input Layer

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

Inp = Input(shape=input_shape, name = 'Input')



# Convolution Layer 1 with MaxPool

x = Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu', name = 'Conv_01')(Inp)

x = MaxPooling2D(pool_size=(2, 2), name = 'MaxPool_01')(x)



# Convolution Layer 2 with MaxPool

x = Conv2D(64, (3, 3), padding='same', activation='relu', name = 'Conv_02')(x)

x =  MaxPooling2D(pool_size=(2, 2), name = 'MaxPool_02')(x)



# Convolution Layer 3 with MaxPool

x = Conv2D(128, (3, 3), padding='same', activation='relu', name = 'Conv_03')(x)

x =  MaxPooling2D(pool_size=(2, 2), name = 'MaxPool_03')(x)



# Convolution Layer 4 with MaxPool

x = Conv2D(256, (3, 3), padding='same', activation='relu', name = 'Conv_04')(x)

x =  MaxPooling2D(pool_size=(2, 2), name = 'MaxPool_04')(x)



# Convolution Layer 5 with MaxPool

x = Conv2D(128, (3, 3), padding='same', activation='relu', name = 'Conv_05')(x)

x =  MaxPooling2D(pool_size=(2, 2), name = 'MaxPool_05')(x)



# Convolution Layer 6 with MaxPool

x = Conv2D(256, (3, 3), padding='same', activation='relu', name = 'Conv_06')(x)

x =  MaxPooling2D(pool_size=(2, 2), name = 'MaxPool_06')(x)



# Flatten 

x = Flatten(name = 'Flatten')(x)



# Fully Connected Layer

x = Dense(256, activation = 'relu', name = 'Dense_01')(x)



# Fully Connected Layer for classification

output = Dense(num_classes, activation='softmax', name = 'Output')(x)

    

model = Model(Inp, output)
# Print out the model structure

model.summary()
# Define the optimizer, loss and metrics

opt = Adam(lr = 0.0001)

model.compile(loss = tf.keras.losses.categorical_crossentropy,

             optimizer = opt,

             metrics = ['accuracy'])
# Specify the steps to be taken for the training and validation

training_steps = max(training_generator.samples // BATCH_SIZE, 1)

validation_steps = max(validation_generator.samples // BATCH_SIZE, 1)



training_generator.reset()  # Reset the training generator to start from the beginning



# Train the model

history = model.fit_generator(

    training_generator,

    steps_per_epoch = training_steps,

    epochs = EPOCHS,

    validation_data = validation_generator,

    validation_steps = validation_steps,

    verbose=1)
def plot_train(hist, choice = 'accuracy'):

    """

    Function to plot accuracy or loss over the training iterations.

    @args:

        hist: The history saved during the model training phase

        choice: The type of function to plot. Should only be either 'accuracy' or 'loss'. Defaults to 'accuracy'.

    """

    h = hist.history

    if choice == 'accuracy':

        meas='accuracy'

        loc='lower right'

    else:

        meas='loss'

        loc='upper right'

    plt.plot(hist.history[meas])

    plt.plot(hist.history['val_'+meas])

    plt.title('model '+meas)

    plt.ylabel(meas)

    plt.xlabel('epoch')

    plt.ylim(ymin=0)

    plt.legend(['train', 'validation'], loc=loc)

    

def view_classify(datagen, pred_model):

    """

    Function for viewing an image and it's predicted class probabilities.

    @args:

        datagen: The image generator of the test data.

        pred_model: The trained model to use to make predictions.

    """

    # Get first batch

    images, labels = datagen.next()

    

    img = images[0]

    preds = model.predict(np.expand_dims(img, axis = 0))

    ps = preds[0]

    

    # Swap the class name and its numeric representation

    classes = sorted(datagen.class_indices, key=datagen.class_indices.get)

    

    print('Probabilities:')

    print(pd.DataFrame({'Class Label': classes, 'Probabilties': ps}))

    

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    ax1.imshow(img.squeeze())

    ax1.axis('off')

    ax2.set_yticks(np.arange(len(classes)))

    ax2.barh(np.arange(len(classes)), ps)

    ax2.set_aspect(0.1)

    ax2.set_yticklabels(classes, size='small')

    ax2.set_title('Class Probability')

    ax2.set_xlim(0, 1.1)



    plt.tight_layout()
# Plot the Model accuracy on train and validation set

plot_train(history)
# Plot the Model loss on train and validation set

plot_train(history, 'loss')
# Try on one image

image_check = validation_df.filename[0] # Image Path

Image(image_check)
# Visualise how the image is seen

def display_activation(model, img_path, col_size, row_size, act_index): 

    """

    Function to plot the feature maps of the convolution networks

    @args:

      model: The trained model which contains the convolutional layers to visualise.

      img_path: The image path to the input image to the network.

      col_size: The number of columns of feature map to fill.

      row_size: The number of rows of feature map to fill.

      act_index: The desired convolution layer to visualise. The range of this index starts from the 1st convolution (0) to the last layer before Flatten/Dense

    """

    # Load and rescale the image

    img = load_img(img_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))

    x = img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x /= 255. 

    

    # Make sure that the layer index to visualise is present in the network

    assert len(model.layers) > act_index

    

    layer_outputs = [layer.output for layer in model.layers[1:]]

    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(x)

    activation = activations[act_index]

    print('Layer in Visualisation:', model.layers[act_index+1].name)

    activation_index=0

    

    # Plot the filters

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))

    for row in range(0,row_size): 

      for col in range(0,col_size):

        ax[row][col].imshow(activation[0, :, :, activation_index], cmap='magma')

        activation_index += 1

display_activation(model, image_check, 6, 5, 0)
view_classify(validation_generator, model)
plotSample(validation_generator, 40, model)
validation_generator.reset()  # Reset the validation generator to start from the beginning



# Compute the overall accuracy of the model with validation dataset

metrics = model.evaluate_generator(validation_generator)

print("model accuracy:",metrics[1])
def plotConfusionMatrix(datagen, model):

    """

    Function to compute and plot a confusion matrix of the class predictions.

    @args:

      datagen: The image generator of the test data to use to make predictions.

      model: The trained model that is used to make predictions.

    """

    # Get the class probabilities of each image

    datagen.reset()

    preds = model.predict_generator(datagen)

    

    # Compute the confusion matrix

    datagen.reset()

    true_labels = datagen.labels

    classes = sorted(datagen.class_indices, key=datagen.class_indices.get)

    cm = confusion_matrix(true_labels, preds.argmax(axis=1))

    cm_df = pd.DataFrame(cm, index = classes, columns = classes)

    

    # Compute the overall accuracy

    overall_acc = np.sum(true_labels == preds.argmax(axis = 1)) / len(true_labels)

    

    # Plot the confusion matrix

    plt.figure(figsize = (5.5, 4))

    sn.heatmap(cm_df, annot = True, cmap = 'magma_r', fmt='g')

    plt.title('Overall Accuracy: {0:.2f}%'.format(overall_acc*100))

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
plotConfusionMatrix(validation_generator, model)