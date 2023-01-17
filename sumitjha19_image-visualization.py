# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import keras 

import tensorflow as tf

import zipfile

import pdb

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

print(check_output(["ls", "../input"]).decode("utf8"))

list_camera_folder=check_output(["ls", "../input/train/"]).decode("utf8")

#print(list_camera_folder)

DEBUG=1;



data= dict();

list_camera_folder = list(list_camera_folder.split('\n')[:-1]);

for camera_model in list_camera_folder:

    #pdb.set_trace()

    if camera_model not in data.keys():

        name_of_image_files = check_output(["ls", "../input/train/"+camera_model]).decode("utf8").split('\n')[:-1]

        data[camera_model]=[];

        for files in name_of_image_files:

            file_path = "../input/train/"+camera_model+'/'+files;

            data[camera_model].append(file_path);

    else:

        name_of_image_files = check_output(["ls", "../input/train/"+camera_model]).decode("utf8").split('\n')[:-1]

        for files in name_of_image_files:

            file_path = "../input/train/"+camera_model+'/'+files;

            data[camera_model].append(file_path);

            

#print length of data

if DEBUG:

    for mykey in data.keys():

        print(mykey,len(data[mykey]))

if DEBUG:

   for index,key in enumerate(data.keys()):

        datalength = len(data[key]);

        index= np.random.randint(datalength, size=1);

        #pdb.set_trace()

        img_name= data[key][index[0]];

        image= mpimg.imread(img_name);

        plt.figure(index+1);plt.imshow(image)

plt.show()

        

        
