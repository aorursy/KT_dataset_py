# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import cv2

from glob import glob



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data_dir = r'/kaggle/input/weed-detection-in-soybean-crops/dataset/dataset/'
classes = ['broadleaf', 'grass', 'soil', 'soybean'] 

num_file = 1190 # number of items in the smallest. Choose this for balanced dataset

dataset = [] 

num_data =num_file*len(classes)

Y = np.zeros(num_data)





for i, cls in enumerate(classes):

    dataset += [f for f in glob(data_dir+cls+'/*.tif')][:num_file]

    Y[i*num_file:(i+1)*num_file] = i # label all classes with int [0.. len(classes)]

# sample the first 10 values of each class

for i in range(len(dataset)):

    if i%1190==0:

        print(str(dataset[i:i+10]).split("/")[-2])

        print("Class Values")

        print(Y[i:i+10])

#https://www.kaggle.com/datduyn/logist-regression-on-plants-weeds-discrimination

    
