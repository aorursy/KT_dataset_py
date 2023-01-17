# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_root_path = '../input/cat-and-dog/training_set/training_set'

#data_root_path = '../input/flowers-recognition/flowers/flowers'
!ls ../input/flowers-recognition/flowers/flowers
!ls ../input/cat-and-dog/training_set/training_set/cats
# If you wish to extract the code manually

#import zipfile

#with zipfile.ZipFile("file.zip","r") as zip_ref:

#    zip_ref.extractall("targetdir")
__author__ = 'sheedo10'

__desc__ = 'CNN Classification Pipeline'

__versn__ ='python_3.5.2'



import os

import random

import warnings



import matplotlib.pyplot as plt

import numpy as np

from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,

                          MaxPooling2D)

from keras.models import Sequential

from keras.optimizers import Adamax

from skimage import io

from skimage.transform import resize

from sklearn.model_selection import train_test_split



# Hide tensorflow warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class_names = []

img_height, img_width, num_channels = 64, 64, 3

num_classes = 2
def image_stat(image):

    print('image_stat: ', image.shape, image.dtype, np.max(image), np.amin(image))
def show(image):

    io.imshow(image)

    io.show()
def show_image_label(image, label):

    assert class_names != [] # class names cannot be empty

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(image)

    ax.set_title(label)  

    plt.show()
def image_preprocess(img):

    img_resized = resize(img, (img_height, img_width, num_channels), mode='reflect')

    return img_resized
'''

Create Images and labels to train and test model

'''

def create_images_and_labels(root_path, save_images):

    images, labels = [], []

    

    # load class names from directory

    classes = os.listdir(root_path)



    # if the user already read, processed and stored the images to a numpy file then just load the numpy file.

    if not save_images:

        print("loading numpy data file...")

        images = np.load('images.npy')

        labels = np.load('labels.npy')

    else:

        for idx, _class in enumerate(classes):

            print('Reading... ', _class)

            image_names = [os.path.join(root_path, _class, fname)

                        for fname in os.listdir(os.path.join(root_path, _class))

                          if fname.endswith('jpg')] #modification ggordon



            # merge processed images for all classes

            images = images + [image_preprocess(io.imread(im_name)) for im_name in image_names]



            # this will generate the one hot label, if there are 3 classes, then label = [0, 0, 0]

            label = np.zeros((len(classes)), dtype=np.int32)

            # one hot - indicates which class the image belongs to

            # e.g If there are classes a,b,c then class a = [1,0,0], class b = [0,1,0], class c = [0,0,1]

            label[idx] = 1  



            # generate labels(y) based on number of images(x)

            labels = labels + [label for _ in range(0, len(image_names))]

            print('Done... ', _class)

        

        # save images and label to binary file, so we don't have to read them again

        np.save('images.npy', images)

        np.save('labels.npy', labels)



    # link images with labels, the network expects each image X to be paired with corresponding label Y

    '''

    if x=[1,2,3] & y=[a,b,c] then zip(x,y) = [(1,a), (2,b), (3,c)]

    ''' 

    images_and_labels_pair = list(zip(images, labels))

    return images_and_labels_pair
'''

Split data into train, test, and validation set



-training size = 70%

-test size = 20%

-validation size= 10%

'''





def cross_validation(images_and_labels_pair, test_size=0.2, valid_size=0.1):

    # unzip, train_test_split expects images and labels to be seperate

    x, y = zip(*images_and_labels_pair)



    # split train(80%) and test(20%)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)



    # split train(90%) and valid(10%)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_size, random_state=0)



    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), np.array(x_valid), np.array(y_valid)
def convnet():

    model = Sequential()

    # This represents the first layer in the network

    # it has 32 filters, therefore 32 feature maps will be produced

    # each convolutonal layer is followed by an activation function then max pooling

    model.add(Conv2D(32, (3, 3), input_shape=(img_height, img_width, num_channels)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(32, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(64, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    # this converts our 2D feature maps to 1D feature vectors

    # feature maps need to be flatten, since classifcaiton requires the last layer to be fully connected

    # and we cannot connect Convolutional layers to fully connected layers because they are of a different dimension

    # Conv Layer works with a 2 dimensional data [matrix], while fully connected works with 1 dimensional data [vector]

    model.add(Flatten())

    model.add(Dense(64))

    model.add(Activation('relu'))

    # dropout - randomly enable and disbale nuerons to prevent overfitting

    model.add(Dropout(0.5))



    # The last layer is based on the number of classes

    # if we had 3 classes, then the final layer would have 3 nuerons, where each neuron would output

    # a probabilty value of the input image associtated with a given class, so eg n1=0.61, n2=0.84, n3=0.15

    model.add(Dense(num_classes))

    model.add(Activation('sigmoid'))



    # Loss function is what is to be minimized

    # Cross entropy is typically used for classification

    # Cross Entropy is used to quantify the difference between two probability distributions

    # Therefore, the network will try to match or approximate the true probability distribution

    # of the training dataset by reducing the cross entropy(differnce between predicted and true distribution)

    # Optimizers are used to optimize(Increase or Decrease) a loss function 

    model.compile(loss='categorical_crossentropy',

                  optimizer=Adamax(),

                  metrics=['accuracy'])



    return model
def train_convnet(data_path, save_images, batch_size=256, epochs=30, show=False, show_limit=2):

    # load images

    images_and_labels_pair = create_images_and_labels(data_path, save_images)



    # create train, test and validation data

    x_train, y_train, x_test, y_test, x_valid, y_valid = cross_validation(images_and_labels_pair)

    print(x_train.shape, y_train.shape)



    # Below code will iterate over each image and display it

    if show and False: # Do not show on server

        with warnings.catch_warnings():

                    warnings.simplefilter("ignore")

                    print('***** SHOWING TRAINING IMAGES *****')

                    for idx, (image, label) in enumerate(zip(x_train, y_train)):

                        print("Image# :", idx + 1)

                        image_stat(image)

                        

                        # find the index in y where 1 is, this will tell us the class name 

                        # so if y = [0, 0, 1, 0, 0], then index = 2

                        index = np.where(label==1)[0][0]

                        label_name = class_names[index]

                        show_image_label(image, label_name)

                        if idx == show_limit: break

                    print('***** SHOWING VALIDATION IMAGES *****')

                    for idx, (image, label) in enumerate(zip(x_valid, y_valid)):

                        print("Image# :", idx + 1)

                        image_stat(image)

                        index = np.where(label==1)[0][0]

                        label_name = class_names[index]

                        show_image_label(image, label_name)

                        if idx == show_limit: break

   

    # load the model

    model = convnet()



    print("training: ", x_train.shape, y_train.shape)

    print("valid: ", x_valid.shape, y_valid.shape)

    print("test: ", x_test.shape, y_test.shape)



    # fit function is used to train the network

    # epoch - An epoch is an iteration over the entire x and y

    # batch_size - Number of samples per gradient update

    model.fit(x_train, y_train, batch_size=batch_size,

                epochs=epochs,

                validation_data=(x_valid, y_valid))





    score = model.evaluate(x_test, y_test)

    print('Test loss: {0}'.format(score[0]))

    print('Test accuracy: {0}'.format(score[1]))



    model.save_weights('convnet.h5')
def evaluate_convnet(model_name, show_limit=10):

    # load the model and weights

    model = convnet()

    model.load_weights(model_name)



    # load the data

    print("loading numpy data file...")

    images = np.load('images.npy')

    labels = np.load('labels.npy')



    # Shuffle the dataset so we dont see the same predicitons 

    dataset = list(zip(images, labels))

    random.shuffle(dataset)



    for idx, data_point in enumerate(dataset):

            x, y = data_point[0], data_point[1]



            # model.predict will generate an array of probability scores

            # The input layer expects a 4d tensor, and the image is 3D, 

            # so we can reshape to create the 4th dimension needed

            probabilities = model.predict(x.reshape(-1, img_height, img_width, num_channels))[0]

 

            # if we had two classes, then model.predict output could look like this [0.81, 0.22]

            # argmax, tells us the array index of the largest probability value, 

            # so np.argmax would return 0 in this case

            index_of_max_prob = np.argmax(probabilities)



            # get class name for the index with the largest probablity value

            predicted_class = class_names[index_of_max_prob]



            # join predicted class name with the probability score

            show_image_label(x, predicted_class + "-" + str(probabilities[index_of_max_prob]))



            if idx == show_limit: break
#data_root_path = './datasets/dogvcat/'

# data_root_path = './datasets/flower_photos/'
class_names = os.listdir(data_root_path)

class_names
train_convnet(data_root_path, save_images=True, show=False)
evaluate_convnet("convnet.h5")