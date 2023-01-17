import numpy as np

import pandas as pd

import os
china = pd.read_csv("/kaggle/input/panda-efnetb2-180-weight/china_gla.csv")

china["file"] = ["/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/Training Images/{}".format(china["filename"].values[i]) for i in range(len(china))]

china["label"] = china["Gla"]

pd.set_option('display.max_rows', 400)

china.head(400)
pd.set_option('display.max_rows', 400)

china_full = pd.read_csv("../input/ocular-disease-recognition-odir5k/full_df.csv")

china_full.head(400)
import matplotlib.pyplot as plt

%matplotlib inline

import cv2
china_1 = china[china["label"]==1]

for i in range(100):

    file_name = china_1["file"].values[i]

    img = cv2.imread(file_name)

    plt.imshow(img)

    plt.show()
china_1 = china[china["label"]==0]

for i in range(100):

    file_name = china_1["file"].values[i]

    img = cv2.imread(file_name)

    plt.imshow(img)

    plt.show()
import math

def get_pad_width(im, new_shape, is_rgb=True):

    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]

    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)

    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)

    if is_rgb:

        pad_width = ((t,b), (l,r), (0, 0))

    else:

        pad_width = ((t,b), (l,r))

    return pad_width

def crop_object(img, thresh=10, maxval=200, square=True):

    """

    Source: https://stackoverflow.com/questions/49577973/how-to-crop-the-biggest-object-in-image-with-python-opencv

    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# convert to grayscale

    #plt.imshow(gray,cmap="gray")

    #plt.show()#普通に白黒のがみえる

    # threshold to get just the signature (INVERTED)

    retval, thresh_gray = cv2.threshold(gray, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #https://qiita.com/anyamaru/items/fd3d894966a98098376c

    # Find object with the biggest bounding box

    mx = (0,0,0,0)      # biggest bounding box so far

    mx_area = 0

    for cont in contours:

        x,y,w,h = cv2.boundingRect(cont)

        area = w*h

        if area > mx_area:

            mx = x,y,w,h

            mx_area = area

    x,y,w,h = mx#(0,0,0,0)なのはcontoursに何も入ってないから

    crop = img[y:y+h, x:x+w]

    if square:

        pad_width = get_pad_width(crop, max(crop.shape))

        crop = np.pad(crop, pad_width=pad_width, mode='constant', constant_values=0)

    return crop
china_0 = china[china["label"]==0]

for i in range(100):

    file_name = china_1["file"].values[i]

    img = cv2.imread(file_name)

    img = crop_object(img, square=False)

    plt.imshow(img)

    plt.show()