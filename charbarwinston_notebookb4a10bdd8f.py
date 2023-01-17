import dicom

import os

import pandas as pd



data_dir = '../input/sample_images/'

patients = os.listdir(data_dir)

labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)



for patient in patients[:1]:

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    print(len(slices),label, slices[0].pixel_array.shape)

    

    

    
len(patients)
import matplotlib.pyplot as plt

import cv2

import math

import numpy as np



IMG_PX_SIZE = 150



HM_SLICES = 20







def chunks(l,n):

    #

    for i in range(0, len(l), n):

        yield l[i:i + n]

        

def mean(l):

    return sum(l)/len(l)



for patient in patients[:1]:

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    

    new_slices = []

    

    slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

    

    chunk_size = math.ceil(len(slices)/ HM_SLICES)

    

    for slice_chunk in chunks(slices,chunk_size):

        slice_chunk = list(map(mean, zip(*slice_chunk)))

        new_slices.append(slice_chunk)

        

    print(len(new_slices))

        

    

'''fig = plt.figure()

    for num,each_slice in enumerate(slices[:12]):

        y = fig.add_subplot(3,4,num+1)

        new_image = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE))

        y.imshow(new_image)

plt.show()'''