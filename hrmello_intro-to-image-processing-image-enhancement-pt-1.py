import cv2

import numpy as np

import matplotlib.pyplot as plt

from skimage import io #scikit-image



# read the image from a link

# opencv doesn't directly read images from an url, so we will use

# skimage module to read it first

img = io.imread("http://www.dphclub.com/tutorials/images/war-time-1.jpg", as_gray=True)

plt.figure(figsize=(12,8))

plt.imshow(img, cmap="gray");
'''

NOTE: it is important to check the values of intensities before going on the analysis. 

Here, for instance, intensities are normalized, i.e. they range from 0 to 1 rather than 0 to 255,

so we need to convert it back to the latter so openCV works properly. That's why I multiply 

the entire image by 255 and convert it to np.uint8 (a format supported by OpenCV).

'''

# transform it to a numpy array 

img_arr = (np.round(np.array(img)*255)).astype(np.uint8)



# flatten

img_arr = img_arr.flatten()



# plot histogram

plt.hist(img_arr, bins = 256, range = [0,256])

plt.title("Number of pixels in each intensity value")

plt.xlabel("Intensity")

plt.ylabel("Number of pixels")

plt.show()
img_eq = cv2.equalizeHist((img*255).astype(np.uint8))

plt.figure(figsize=(12,8))

plt.imshow(img_eq, cmap = "gray");
#get the numpy array for the equalized image

img_eq_arr = np.array(img_eq)



# flatten

img_eq_arr = img_eq_arr.flatten()



# plot histogram

plt.hist(img_eq_arr, bins = 256, range = [0,256])

plt.title("Number of pixels in each intensity value")

plt.xlabel("Intensity")

plt.ylabel("Number of pixels")

plt.show()
# read the image

img = io.imread("https://i.ibb.co/PzYwtD9/clahe-first.jpg", as_gray=True)

# use Histogram Equalization

img_eq = cv2.equalizeHist((img*255).astype(np.uint8))

# plot

plt.figure(figsize=(10,7))

plt.imshow(img_eq, cmap = "gray");
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

clahe_img = clahe.apply((img*255).astype(np.uint8))

plt.figure(figsize=(10,7))

plt.imshow(clahe_img, cmap="gray");
from sklearn.datasets import load_files

from keras.preprocessing.image import load_img

from tqdm import tqdm # progress bar



TRAIN_DIR = '../input/fruits-360_dataset/fruits-360/Training'

TEST_DIR = '../input/fruits-360_dataset/fruits-360/Test'



def load_dataset(path):

    '''

    Gets the path to the directory where 

    the image paths and return features, 

    targets and labels

    '''

    data = load_files(path)

    files = np.array(data['filenames'])

    targets = np.array(data['target'])

    target_labels = np.array(data['target_names'])

    return files,targets,target_labels

    

def convert_image_to_array(files):

    '''

    Take a file path as input and return the image in grayscale associated with that path.

    Returns an image in grayscale of size (100,100,1)

    '''

    images_as_array=[]

    for file in tqdm(files):

        # Convert to Numpy Array

        # np.expand_dims creates the channel dimension, that is

        # original shape: (100,100) | Final shape: (100,100,1)

        images_as_array.append(np.expand_dims(np.array((load_img(file, color_mode = "grayscale"))), axis=2))

        

        # uncomment the line below if you want to load colored images

        # but be aware that we need to be more careful to use HE and CLAHE

        # in colored images. See here for more detials: 

        # https://hypjudy.github.io/2017/03/19/dip-histogram-equalization/

#         images_as_array.append(np.array(load_img(file)))

    return images_as_array



# load train and test datasets

x_train, y_train,target_labels = load_dataset(TRAIN_DIR)

x_test, y_test,_ = load_dataset(TEST_DIR)



# convert the paths to actual images to be used in training

xtrain_img = np.array(convert_image_to_array(x_train))

xtest_img = np.array(convert_image_to_array(x_test))