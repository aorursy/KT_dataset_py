# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dicom

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

data_dir = "../input/sample_images/"



patients = os.listdir(data_dir)

labels_df = pd.read_csv("../input/stage1_labels.csv", index_col=0)

#print(patients)

#print(labels_df.head())



# Any results you write to the current directory are saved as output.
for patient in patients[:1]:

    label = labels_df.get_value(patient,'cancer')

    path = data_dir + patient

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    print(len(slices), slices[0].pixel_array.shape)

    
len(patients)
import matplotlib.pyplot as plt



for patient in patients[:1]:

    label = labels_df.get_value(patient,'cancer')

    path = data_dir + patient

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    plt.imshow(slices[0].pixel_array)

    plt.show()

    
import cv2

IMG_PX_SIZE = 150



for patient in patients[:1]:

    label = labels_df.get_value(patient,'cancer')

    path = data_dir + patient

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    

    fig = plt.figure()

    for num, each_slice in enumerate(slices[:12]):

        y = fig.add_subplot(3,4,num+1)

        new_img = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE))

        y.imshow(new_img)

    plt.show()


