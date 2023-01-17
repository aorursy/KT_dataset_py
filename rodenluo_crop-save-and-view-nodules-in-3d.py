import SimpleITK as sitk

import numpy as np



from glob import glob

import pandas as pd

import scipy.ndimage



import mhd_utils_3d





## Read annotation data and filter those without images

# Learned from Jonathan Mulholland and Aaron Sander, Booz Allen Hamilton

# https://www.kaggle.com/c/data-science-bowl-2017#tutorial



# Set input path

# Change to fit your environment

luna_path = './LUNA16/'

luna_subset_path = luna_path + 'subset0_samples/'

file_list = glob(luna_subset_path + "*.mhd")



df_node = pd.read_csv(luna_path+'annotations.csv')



def get_filename(file_list, case):

    for f in file_list:

        if case in f:

            return(f)



# map file full path to each record 

df_node['file'] = df_node['seriesuid'].map(lambda file_name: get_filename(file_list, file_name))

df_node = df_node.dropna()



## Define resample method to make images isomorphic, default spacing is [1, 1, 1]mm

# Learned from Guido Zuidhof

# https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

def resample(image, old_spacing, new_spacing=[1, 1, 1]):

    

    resize_factor = old_spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = old_spacing / real_resize_factor



    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode = 'nearest')

    

    return image, new_spacing

#!/usr/bin/env python

#coding=utf-8



#======================================================================

#Program:   Diffusion Weighted MRI Reconstruction

#Link:      https://code.google.com/archive/p/diffusion-mri

#Module:    $RCSfile: mhd_utils.py,v $

#Language:  Python

#Author:    $Author: bjian $

#Date:      $Date: 2008/10/27 05:55:55 $

#Version:   

#           $Revision: 1.1 by PJackson 2013/06/06 $

#               Modification: Adapted to 3D

#               Link: https://sites.google.com/site/pjmedphys/tutorials/medical-images-in-python

# 

#           $Revision: 2   by RodenLuo 2017/03/12 $

#               Modication: Adapted to LUNA2016 data set for DSB2017

#               Link: 

#======================================================================



import os

import numpy

import array



def write_meta_header(filename, meta_dict):

    header = ''

    # do not use tags = meta_dict.keys() because the order of tags matters

    tags = ['ObjectType','NDims','BinaryData',

       'BinaryDataByteOrderMSB','CompressedData','CompressedDataSize',

       'TransformMatrix','Offset','CenterOfRotation',

       'AnatomicalOrientation',

       'ElementSpacing',

       'DimSize',

       'ElementType',

       'ElementDataFile',

       'Comment','SeriesDescription','AcquisitionDate','AcquisitionTime','StudyDate','StudyTime']

    for tag in tags:

        if tag in meta_dict.keys():

            header += '%s = %s\n'%(tag,meta_dict[tag])

    f = open(filename,'w')

    f.write(header)

    f.close()

    

def dump_raw_data(filename, data):

    """ Write the data into a raw format file. Big endian is always used. """

    #Begin 3D fix

    data=data.reshape([data.shape[0],data.shape[1]*data.shape[2]])

    #End 3D fix

    rawfile = open(filename,'wb')

    a = array.array('f')

    for o in data:

        a.fromlist(list(o))

    #if is_little_endian():

    #    a.byteswap()

    a.tofile(rawfile)

    rawfile.close()

    

def write_mhd_file(mhdfile, data, dsize):

    assert(mhdfile[-4:]=='.mhd')

    meta_dict = {}

    meta_dict['ObjectType'] = 'Image'

    meta_dict['BinaryData'] = 'True'

    meta_dict['BinaryDataByteOrderMSB'] = 'False'

    meta_dict['ElementType'] = 'MET_FLOAT'

    meta_dict['NDims'] = str(len(dsize))

    meta_dict['DimSize'] = ' '.join([str(i) for i in dsize])

    meta_dict['ElementDataFile'] = os.path.split(mhdfile)[1].replace('.mhd','.raw')

    write_meta_header(mhdfile, meta_dict)



    pwd = os.path.split(mhdfile)[0]

    if pwd:

        data_file = pwd +'/' + meta_dict['ElementDataFile']

    else:

        data_file = meta_dict['ElementDataFile']



    dump_raw_data(data_file, data)

    

def save_nodule(nodule_crop, name_index):

    np.save(str(name_index) + '.npy', nodule_crop)

    write_mhd_file(str(name_index) + '.mhd', nodule_crop, nodule_crop.shape[::-1])
## Collect patients with nodule and crop the nodule

# In this code snippet, the cropped nodule is a [19, 19, 19] volume with [1, 1, 1]mm spacing.

# Learned from Jonathan Mulholland and Aaron Sander, Booz Allen Hamilton

# https://www.kaggle.com/c/data-science-bowl-2017#tutorial



# Change the number in the next line to process more

for patient in file_list[:1]:

    print(patient)

    

    # Check whether this patient has nodule or not

    if patient not in df_node.file.values:

        print('Patient ' + patient + 'Not exist!')

        continue

    patient_nodules = df_node[df_node.file == patient]

    

    full_image_info = sitk.ReadImage(patient)

    full_scan = sitk.GetArrayFromImage(full_image_info)

    

    origin = np.array(full_image_info.GetOrigin())[::-1] # get [z, y, x] origin

    old_spacing = np.array(full_image_info.GetSpacing())[::-1] # get [z, y, x] spacing

    

    image, new_spacing = resample(full_scan, old_spacing)

    

    print('Resample Done')

    



    for index, nodule in patient_nodules.iterrows():

        nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX]) 

        # Attention: Z, Y, X



        v_center = np.rint( (nodule_center - origin) / new_spacing )

        v_center = np.array(v_center, dtype=int)



#         print(v_center)

        window_size = 9 # This will give you the volume length = 9 + 1 + 9 = 19

        # Why the magic number 19, I found that in "LUNA16/annotations.csv", 

        # the 95th percentile of the nodules' diameter is about 19.

        # This is kind of a hyperparameter, will affect your final score.

        # Change it if you want.

        zyx_1 = v_center - window_size # Attention: Z, Y, X

        zyx_2 = v_center + window_size + 1



#         print('Crop range: ')

#         print(zyx_1)

#         print(zyx_2)



        # This will give you a [19, 19, 19] volume

        img_crop = image[ zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2] ]

        

        # save the nodule 

        save_nodule(img_crop, index)

    

    print('Done for this patient!\n\n')

print('Done for all!')
## Plot volume in 2D



import numpy as np

from matplotlib import pyplot as plt



def plot_nodule(nodule_crop):

    

    # Learned from ArnavJain

    # https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing

    f, plots = plt.subplots(int(nodule_crop.shape[0]/4)+1, 4, figsize=(10, 10))

    

    for z_ in range(nodule_crop.shape[0]): 

        plots[int(z_/4), z_ % 4].imshow(nodule_crop[z_,:,:])

    

    # The last subplot has no image because there are only 19 images.

    plt.show()

    

# Plot one example

img_crop = np.load('25.npy')

plot_nodule(img_crop)