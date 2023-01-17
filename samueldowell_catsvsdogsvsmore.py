import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras as ks # neural network models



# For working with images

import cv2 as cv

import matplotlib.image as mpimg

import tqdm

import matplotlib.pyplot as plt    





# Potentially useful tools - you do not have to use these

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from keras.utils import to_categorical

from keras import Sequential

from keras.layers import Activation, Convolution2D, Flatten, Dense, Dropout, MaxPooling2D

from keras.models import Model, Sequential

from keras.applications import InceptionV3, ResNet50, Xception

from keras.applications.vgg16 import preprocess_input, decode_predictions



from random import randint, shuffle

import glob





import os



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
# CONSTANTS

# You may not need all of these, and you may find it useful to set some extras



CATEGORIES = ['airplane','car','cat','dog','flower','fruit','motorbike','person']

NUM_CLASSES = 8

IMG_WIDTH = 100

IMG_HEIGHT = 100

TRAIN_PATH = '../input/natural_images/natural_images/'

TEST_PATH = '../input/evaluate/evaluate/'
# To find data:

# returns dataframe 

# data structure is three-column table

# first column is class, second column is filename, third column is image address relative to TRAIN_PATH

def load_image_locations():

    folders = os.listdir(TRAIN_PATH)



    images = []



    for folder in folders:

        files = os.listdir(TRAIN_PATH + folder)

        images += [(folder, file, folder + '/' + file) for file in files]



    image_locs = pd.DataFrame(images, columns=('class','filename','file_loc'))



    return image_locs

# data structure is three-column table

# first column is class, second column is filename, third column is image address relative to TRAIN_PATH

#Files must be unzipped

#img = cv.imread(TRAIN_PATH + "airplane/airplane_0308.jpg")

#display(img)
# Function which takes in a data frame object with columns "class" "filename" and "file_loc". 

# Returns three lists of images, file names and relative file_locations.

def extract_image_data(image_locs):

    images = []

    file_locs = []

    classes = []

    # Load all the images and append them in an array. 

    # Also append the class and id for each image in arrays.





    for _, row in tqdm.tqdm(image_locs.iterrows()):

      path = TRAIN_PATH + str(row["file_loc"])

      img = cv.imread(path)

      img = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT)) 

      images.append(img)

      file_locs.append(row["file_loc"])

      classes.append(row["class"])

        

    return images, file_locs, classes

# Helper-function for plotting images

def plot_images(images, classes):

    assert len(images) == len(classes) == NUM_CLASSES

    

    # Create figure with 4x4 sub-plots.

    fig, axes = plt.subplots(2, 4, figsize=(10,10),sharex=True)

    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    

    for i, ax in enumerate(axes.flat):

        # Plot image.

        ax.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB), cmap='hsv')    

        xlabel = "Label: {0}".format(classes[i])

    

        # Show the classes as the label on the x-axis.

        ax.set_xlabel(xlabel)

        ax.xaxis.label.set_size(10)

        

        # Remove ticks from the plot.

        ax.set_xticks([])

        ax.set_yticks([])

    

    # Ensure the plot is shown correctly with multiple plots

    # in a single Notebook cell.

    plt.show()



# Your code here

def create_model():

    # Channels first tells the pooling layer to use the (Height, Width, Depth) format instead of the (Depth, Height, Width)

    data_format="channels_first"



    # A sequential model is a basic model structure where the layers are stacked layer by layer. 

    # Another option with keras is a functional model, layers can be connected to literally any other layer within the model.

    model = Sequential()

    # A convolutional layer slides a filter over the image which is fed to the activation layer so the model can learn

    # features and activate when they see one of these visual features. Only activated features are carried over to the 

    # next layer.

    model.add(Convolution2D(32, (3, 3), input_shape=(100, 100, 3)))

    # Relu maps all negative values to 0 and keeps all positive values.

    model.add(Activation('sigmoid'))

    # A pooling layer reduces the dimensions of the image but not the depth. The computation is faster and less image data

    # means less chance of over fitting. 

    model.add(MaxPooling2D(data_format=data_format, pool_size=(2, 2)))



    model.add(Convolution2D(32, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(data_format=data_format, pool_size=(2, 2)))



    model.add(Convolution2D(64, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(data_format=data_format, pool_size=(2, 2)))



    # Squashes the output of the previous layer to an array with 1 dimension

    model.add(Flatten())

    # A dense layer's takes n num of inputs and is connected to every output by the weights it produces. 

    model.add(Dense(64))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    # Fully connected layer or Dense layer in keras, with the number of classes that the network will be able to predict.

    model.add(Dense(8))

    # Sigmoid is used because it exists between 0 and 1 and we want to give a prediction between 0 and 1.

    model.add(Activation('sigmoid'))



    model.compile(optimizer='rmsprop',              # Performs gradient descent, finding the lowest error value

                  loss='binary_crossentropy',       # Cross entropy measures the performance of each prediction made by the network

                  metrics=['accuracy'])

    

    return model







# Example values:

filenames = ['test001','test002','test003','test004']

predictions = ['car','cat','fruit','motorbike']
# Save results



# results go in dataframe: first column is image filename, second column is category name

# category names are: airplane, car, cat, dog, flower, fruit, motorbike, person

df = pd.DataFrame()

df['filename'] = filenames

df['label'] = predictions

df = df.sort_values(by='filename')



df.to_csv('results.csv', header=True, index=False)
def one_hot_to_label(prediction):

    k = 0

    for i in range(0, len(onehot_encoded)):

        if np.array_equal(prediction, onehot_encoded[i]):

            break

        k = k + 1

    return CATEGORIES[k]




if __name__ == "__main__":

    # CONSTANTS

    # You may not need all of these, and you may find it useful to set some extras



    CATEGORIES = ['airplane','car','cat','dog','flower','fruit','motorbike','person']

    NUM_CLASSES = 8

    IMG_WIDTH = 100

    IMG_HEIGHT = 100

    TRAIN_PATH = '../input/natural_images/natural_images/'

    TEST_PATH = '../input/evaluate/evaluate/'

    EPOCHS = 10

    VALIDATION_SPLIT = 0.1

    

    categories = np.asarray(CATEGORIES)

    print(CATEGORIES)

    print(categories)

    onehot_encoder = OneHotEncoder(sparse=False)

    onehot_encoded = onehot_encoder.fit_transform(categories.reshape(-1,1))

    print(onehot_encoded)

    image_locs = load_image_locations()

    display(image_locs.head())

    images, file_locs, classes = extract_image_data(image_locs)

    print("Image One:", images[0])

    print("file_loc One:", file_locs[0])

    print("class one:", classes[0])

    one_hot_labels = np.asarray(pd.get_dummies(classes, sparse=True))

    print("class one hot encoded:", one_hot_labels[0])

    x = np.asarray(images)

    y = np.asarray(one_hot_labels)

    

    model = create_model()

    model.fit(

        x=x,

        y=y,

        epochs=EPOCHS,

        validation_split=VALIDATION_SPLIT

          )

    

    test_filenames = os.listdir(TEST_PATH)

    test_images = get_test_images(test_filenames)

    predictions = model.predict(np.asarray(test_images))

    print(predictions[:10])

    print()

    maxed_predictions = (predictions == predictions.max(axis=1)[:,None]).astype(int)

    print(maxed_predictions[:10])

    print()

    class_predictions = [one_hot_to_label(prediction) for prediction in maxed_predictions]

    print(class_predictions[:10])

    

    df = pd.DataFrame()

    df['filename'] = pd.Series(test_filenames)

    df['label'] = pd.Series(class_predictions)

    df = df.sort_values(by='filename')

    display(df.head())

    df.to_csv('results.csv', header=True, index=False)

    

    
def get_test_images(filenames):

    test_images = []

    for file in filenames:

        path = TEST_PATH + file

        img = cv.imread(path)

        img = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT)) 

        test_images.append(img)



    return test_images
print(type(class_predictions))

print(type(test_images))
# Save results

print(os.listdir(TEST_PATH))



# test_images = get_test_images(test_filenames)

# print(test_images[0])

# predictions = get_predictions()

# results go in dataframe: first column is image filename, second column is category name

# category names are: airplane, car, cat, dog, flower, fruit, motorbike, person

# df = pd.DataFrame()

# df['filename'] = filenames

# df['label'] = predictions

# df = df.sort_values(by='filename')



#df.to_csv('results.csv', header=True, index=False)