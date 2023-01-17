# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib.image as mpimg





def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



img = mpimg.imread('/kaggle/input/brain-mri-images-for-brain-tumor-detection/yes/Y1.jpg')

gray_img = rgb2gray(img)



print('Img shape:', gray_img.shape)



plt.imshow(gray_img, cmap="gray")

plt.grid(False)

plt.show()
import cv2



def getInfo(imgs, img_path):

    print('Image Information')

    print('==================')

    print('Path:', img_path)



    try:

        for img in imgs:

            print('shape: {} size:{}'.format(img.shape, img.size))

    except TypeError:

        print('ERROR: TypeError, please check function input')





def getImage(img_path, display=True):

    img = cv2.imread(img_path)

    b, g, r = cv2.split(img)

    image_block = [img, b, g, r]



    if display:

        getInfo(image_block, img_path)



        cv2.imshow('Brain', img)

        cv2.imshow('B channel', b)

        cv2.imshow('G channel', g)

        cv2.imshow('R channel', r)



        cv2.waitKey()



    return image_block





def multiPyrDown(img, debug=False):

    result = [img]

    temp_img = img.copy()



    print('### Starting Multi-PyrDown ###')

    while temp_img.size > 4:

        if debug:

            print(temp_img.shape)

        temp_img = cv2.pyrDown(temp_img)

        result.append(temp_img)



    print('### Multi-PyrDown, Done with length: {} ###'.format(len(result)))



    return result

import json



# Example: data['orders']['2']

with open('/kaggle/input/hilbert-index/hc_index.json') as json_file:

    hci = json.load(json_file)
pyr_img = multiPyrDown(gray_img, debug=True)
# Q1 Q2

# d_theta = [0, 30, 90, 120, 180]



# Q2 Q3

# d_theta = [90, 120, 150, 180, 210, 240, 270]



# Q3 Q4

# d_theta = [180, 210, 240, 270, 300, 330, 360]



# Q1 Q4

# d_theta = [0, 30, 60, 90, 270, 300, 330]



# Q1 Q3

# d_theta = [0, 30, 60, 90, 180, 210, 240]
import cv2



d_scale = 31

d_sigma = 9

d_theta = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

d_lambda = 2

d_gamma = 1

gabor_store = []

invar_k = np.ndarray(shape=(d_scale, d_scale), dtype=np.float64)



for angle in d_theta:

    # Get Kernal

    gabor_k = np.asarray(cv2.getGaborKernel((d_scale, d_scale), d_sigma, angle, d_lambda, d_gamma, 0, ktype=cv2.CV_64F))

    invar_k = np.add(invar_k, gabor_k)

    gabor_store.append(gabor_k)

    



# plt.imshow(gabor_store[0], cmap='gray')

plt.imshow(invar_k, cmap='gray')



plt.grid(False)

plt.show()



# Filtering

test_img = pyr_img[0]



invar_img = cv2.filter2D(test_img, -1, invar_k, cv2.CV_64F)

plt.imshow(invar_img, cmap='gray')

plt.grid(False)

plt.show()
def transform2hc(img, hcc):

    '''

        trans(img, hcc):

            2D to 1D Transformed by Hilbert Curve



        img <-- nxn matrix

        hcc <-- Hibert curve coordinate with order k

        k <-- 4^log2(n) or nxn, length of hcc

    '''



    result = []

    k = len(hcc)

    spec_map = []



    for i in np.arange(k):

        (x, y) = hcc[i]

        try:

            val_img = img[x][y]

            result.append(val_img)

            spec_map.append([x, y])

        except IndexError:

            continue



    return result, spec_map
%matplotlib inline

import matplotlib.pyplot as plt



plt.style.use('seaborn-whitegrid')



hc_order = str(int(np.log2(invar_img.shape[0]))+1)



vhc, spec_map = transform2hc(invar_img, hci['orders'][hc_order])

plt.plot (vhc)

plt.show()



with open('/kaggle/input/mypoints/Y1.jpg.json') as json_file:

    tape = json.load(json_file)
selected_index = []

for pts in tape['tumor']:

    hc_index = spec_map.index(pts)

    selected_index.append(hc_index)



print(selected_index)
L = 2000

for idx in selected_index:

    print('index:', idx)

    print(idx-L, idx+L)

    data = vhc[idx-L:idx+L]

    plt.plot(data)

    plt.show()

    