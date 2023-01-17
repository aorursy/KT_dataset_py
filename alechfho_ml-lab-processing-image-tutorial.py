# The original code comes from https://github.com/ardamavi/Sign-Language-Digits-Dataset

import os
from os import listdir
import glob
import numpy as np
import cv2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
dataset_path = "../input/sign_lang_dataset/Dataset"

sub_dir = glob.glob(os.path.join(dataset_path, '*'))
sub_dir

def display_img(img_path):
    img = cv2.imread(img_path)
    #TODO: colour correct the image
    color_corrected = None

    plt.imshow(color_corrected)
    plt.title(img_path)
    plt.show()
img_dir = sub_dir[0]

img_files = listdir(img_dir)
img_files[:3]

display_img(img_path=os.path.join(img_dir, img_files[0]))
img_dir = sub_dir[4]

img_files = listdir(img_dir)
img_files[:3]

#TODO: Can you display the names of the 5th to 10th image files?

display_img(img_path=os.path.join(img_dir, img_files[0]))

def get_gsimg(image_path):
    img = cv2.imread(image_path)
    #TODO: resize the image to 64 x 64 and extract the greyscale values of the given image
    resize_img = None
    gs_img = None
    return gs_img

gs_img = get_gsimg(os.path.join(img_dir, img_files[0]))
# the shape of the image array should be (64, 64)
gs_img.shape

plt.imshow(gs_img, cmap='gray')
plt.show()
def extract_array(dataset_path):
    label_dirs = glob.glob(os.path.join(dataset_path, '*'))
    num_classes = len(label_dirs)
    X = []
    Y = []
    for label_path in label_dirs:
        label = int(str.split(label_path, '/')[-1])
        imgs = glob.glob(os.path.join(label_path, '*.JPG'))
        for img in imgs:
            gs_img = get_gsimg(img)
            X.append(gs_img)
            Y.append(label)
    #TODO: normalize the values in X 
    X = np.array(X).astype('float32')
    #TODO: make the label a one-hot vector for example if the value is 3, then [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    Y = None
    return X, Y

X, Y = extract_array(dataset_path)
print(X.shape)
# X shape should be (2062, 64, 64)
print(Y.shape)
# Y shape should be (2062, 10)