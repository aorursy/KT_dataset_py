# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        '''



# Any results you write to the current directory are saved as output.
#may be try to classify r, u, and v

TRAIN_DIR = '/kaggle/input/asl_alphabet_train/asl_alphabet_train'

TEST_DIR = '../input/asl_alphabet_test/asl_alphabet_test'



a = os.listdir(TEST_DIR)

print(a)
b = os.listdir(os.path.join(TRAIN_DIR, 'Z'))

r_file_paths = os.path.join(TRAIN_DIR, 'R')

u_file_paths = os.path.join(TRAIN_DIR, 'U')

v_file_paths = os.path.join(TRAIN_DIR, 'V')

#b
import random

import cv2

from glob import glob

from matplotlib import pyplot as plt

from numpy import floor

def sample_image(letter):

    letter_file_paths = os.path.join(TRAIN_DIR, letter)

    list_dir = os.listdir(letter_file_paths)

    random_dir = random.sample(list_dir, 5)

    images_dir = [os.path.join(letter_file_paths, d) for d in random_dir]

    plt.figure(figsize=(16,16))

    #imgs = random.sample(path_contents, 3)

    plt.subplot(131)

    plt.imshow(cv2.imread(images_dir[0]))

    plt.subplot(132)

    plt.imshow(cv2.imread(images_dir[1]))

    plt.subplot(133)

    plt.imshow(cv2.imread(images_dir[2]))



    

    

        
#start with CNN

from scipy import misc

r_images_path = [os.path.join(r_file_paths, d) for d in os.listdir(r_file_paths)]

image = misc.imread(r_images_path[0])

print(image.shape)

M, N, D = image.shape

image = image.reshape(M * N * D)

print(image.shape)
def get_r_dataset():

    r_images_path_train = [os.path.join(r_file_paths, d) for d in os.listdir(r_file_paths)]

    

    r_matrix = np.empty([len(r_images_path_train), M * N* D])

    

    counter = 0

    for i in range(len(r_images_path_train)):

        r_image = misc.imread(r_images_path_train[i])

        

        r_image = r_image.reshape(1, M * N * D)

        

        r_matrix[counter] = r_image

        counter += 1



    return r_matrix

def get_v_dataset():

    v_images_path_train = [os.path.join(v_file_paths, d) for d in os.listdir(v_file_paths)]



    v_matrix = np.empty([len(v_images_path_train), M * N * D])



    counter = 0

    for i in range(len(v_images_path_train)):

        v_image = misc.imread(v_images_path_train[i])

        

        v_image = v_image.reshape(1, M * N * D)

        

        v_matrix[counter] = v_image

        counter += 1

        

    return v_matrix

    

def get_u_dataset():

    u_images_path_train = [os.path.join(u_file_paths, d) for d in os.listdir(u_file_paths)]

    

    u_matrix = np.empty([len(u_images_path_train), M * N * D])

    

    counter = 0

    for i in range(len(u_images_path_train)):

        u_image = misc.imread(u_images_path_train[i])

    

        u_image = u_image.reshape(1, M * N * D)

        

        u_matrix[counter] = u_image

        counter += 1

    

    return u_matrix
R = get_r_dataset()

U = get_u_dataset()

#V = get_v_dataset()
X_train = np.concatenate((R, U), axis = 0)
#prepare y_train

#R->1, U->2

y_train = np.zeros((X_train.shape[0], 1))

for i in range(len(y_train) / 2):

    y_train[i] = 1

for i in range(len(y_train) /2, len(y_train)):

    y_train[i] = 2

y_train
from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()



# 2. FIT

enc.fit(y_train)



# 3. Transform

y_train = enc.transform(y_train).toarray()

y_train.shape
print(X_train.shape)
r_images_path_train = [os.path.join(r_file_paths, d) for d in os.listdir(r_file_paths)]

len(r_images_path_train)