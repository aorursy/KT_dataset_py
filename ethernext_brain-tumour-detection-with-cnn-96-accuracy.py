!pip install imutils

import numpy as np 

import pandas as pd 

import os

from os import listdir

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

import imutils    



from tensorflow.keras.models import Model,load_model

from tensorflow.keras.layers import Conv2D,Input,ZeroPadding2D,BatchNormalization,Flatten,Activation,Dense,MaxPooling2D

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle #shuffling the data improves the model

image_dir="../input/brain-mri-images-for-brain-tumor-detection/"


os.makedirs('../output/kaggle/working/augmented-images')

os.makedirs('../output/kaggle/working/augmented-images/yes')

os.makedirs('../output/kaggle/working/augmented-images/no')
def augment_data(file_dir, n_generated_samples, save_to_dir):

    data_gen = ImageDataGenerator(rotation_range=10, 

                                  width_shift_range=0.1, 

                                  height_shift_range=0.1, 

                                  shear_range=0.1, 

                                  brightness_range=(0.3, 1.0),

                                  horizontal_flip=True, 

                                  vertical_flip=True, 

                                  fill_mode='nearest'

                                 )



    for filename in listdir(file_dir):

        image = cv2.imread(file_dir + '/' + filename)

        # reshape the image

        image = image.reshape((1,)+image.shape)

        save_prefix = 'aug_' + filename[:-4]

        i=0

        for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir,save_prefix=save_prefix, save_format='jpg'):

                i += 1

                if i > n_generated_samples:

                    break
augmented_data_path ='../output/kaggle/working/augmented-images/'

# augment data for the examples with label equal to 'yes' representing tumurous examples

augment_data(file_dir=image_dir+'yes',n_generated_samples=6, save_to_dir=augmented_data_path+'yes')

# augment data for the examples with label equal to 'no' representing non-tumurous examples

augment_data(file_dir=image_dir+'no', n_generated_samples=9, save_to_dir=augmented_data_path+'no')
def crop_brain_contour(image, plot=False):

    

    # Convert the image to grayscale, and blur it slightly

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.erode(thresh, None, iterations=2)

    thresh = cv2.dilate(thresh, None, iterations=2)



    # Find contours in thresholded image, then grab the largest one

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    c = max(cnts, key=cv2.contourArea)

    # extreme points

    extLeft = tuple(c[c[:, :, 0].argmin()][0])

    extRight = tuple(c[c[:, :, 0].argmax()][0])

    extTop = tuple(c[c[:, :, 1].argmin()][0])

    extBot = tuple(c[c[:, :, 1].argmax()][0])

    

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)

    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            



    if plot:

        plt.figure()

        plt.subplot(1, 2, 1)

        plt.imshow(image)

        plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Original Image')

        plt.subplot(1, 2, 2)

        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both',top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Cropped Image')

        plt.show()

    

    return new_image
ex_img = cv2.imread(image_dir+'yes/Y107.jpg')

ex_crop_img = crop_brain_contour(ex_img, True)
def load_data(dir_list, image_size):



    # load all images in a directory

    X = []

    y = []

    image_width, image_height = image_size

    

    for directory in dir_list:

        for filename in listdir(directory):

            image = cv2.imread(directory+'/'+filename)

            image = crop_brain_contour(image, plot=False)

            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)

            # normalize values

            image = image / 255.

            # convert image to numpy array and append it to X

            X.append(image)

            # append a value of 1 to the target array if the image

            # is in the folder named 'yes', otherwise append 0.

            if directory[-3:] == 'yes':

                y.append([1])

            else:

                y.append([0])

                

    X = np.array(X)

    y = np.array(y)

    

    # Shuffle the data

    X, y = shuffle(X, y)

    

    print(f'Number of examples is: {len(X)}')

    print(f'X shape is: {X.shape}')

    print(f'y shape is: {y.shape}')

    

    return X, y
augmented_yes =augmented_data_path+'yes'

augmented_no = augmented_data_path+'no'



IMG_WIDTH, IMG_HEIGHT = (240, 240)



X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))
def plot_sample_images(X, y, n=40):

    for label in [0,1]:

        # grab the first n images with the corresponding y values equal to label

        images = X[np.argwhere(y == label)]

        n_images = images[:n]

        

        columns_n = 10

        rows_n = int(n/ columns_n)



        plt.figure(figsize=(10, 8))

        

        i = 1 # current plot        

        for image in n_images:

            plt.subplot(rows_n, columns_n, i)

            plt.imshow(image[0])

            

            # remove ticks

            plt.tick_params(axis='both', which='both', 

                            top=False, bottom=False, left=False, right=False,

                           labelbottom=False, labeltop=False, labelleft=False, labelright=False)

            

            i += 1

        

        label_to_str = lambda label: "Yes" if label == 1 else "No"

        plt.suptitle(f"Brain Tumor: {label_to_str(label)}")

        plt.show()
plot_sample_images(X, y)
def split_data(X, y, test_size=0.2):

       

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)

    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)

    

    return X_train, y_train, X_val, y_val, X_test, y_test
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)
print ("number of training examples = " + str(X_train.shape[0]))

print ("number of validation examples = " + str(X_val.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))
def build_model(input_shape):

    X_input = Input(input_shape) 

    X = ZeroPadding2D((2, 2))(X_input) 

    

    X = Conv2D(32, (7, 7), strides = (1, 1))(X)

    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    X = Activation('relu')(X) 

    

    X = MaxPooling2D((4, 4))(X) 

    X = MaxPooling2D((4, 4))(X) 

    X = Flatten()(X) 

    X = Dense(1, activation='sigmoid')(X) 

    model = Model(inputs = X_input, outputs = X)

    

    return model
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

model=build_model(IMG_SHAPE)

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y=y_train, batch_size=32, epochs=22, validation_data=(X_val, y_val))
history = model.history.history
def plot_metrics(history):

    

    train_loss = history['loss']

    val_loss = history['val_loss']

    train_acc = history['accuracy']

    val_acc = history['val_accuracy']

    

    # Loss

    plt.figure()

    plt.plot(train_loss, label='Training Loss')

    plt.plot(val_loss, label='Validation Loss')

    plt.title('Loss')

    plt.legend()

    plt.show()

    

    # Accuracy

    plt.figure()

    plt.plot(train_acc, label='Training Accuracy')

    plt.plot(val_acc, label='Validation Accuracy')

    plt.title('Accuracy')

    plt.legend()

    plt.show()
plot_metrics(history)