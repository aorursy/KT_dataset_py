!pip install p_tqdm
# ----------------------------------------

# imports

# ----------------------------------------

import os

import pydicom

import cv2

from PIL import Image

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm

# uncomment if you want to run the dicom data collection code

from multiprocessing import pool, cpu_count

from multiprocessing.dummy import Pool as ThreadPool

from p_tqdm import p_map #pip install p_tqdm

import pandas as pd

import numpy as np
# ----------------------------------------

# constants

# ----------------------------------------

root_path = '/kaggle/input/siim-isic-melanoma-classification'

data_path = root_path

output_path = '/kaggle/working'
# ------------------------------------------------------

# get a list of all of the training dicom files

# ------------------------------------------------------

dcom_list = [f for f in os.listdir(f'{data_path}/train')]

dcom_test_list = [f for f in os.listdir(f'{data_path}/test')]
# ------------------------------------------------------

# read in a dicom image and all of its metadata

# ------------------------------------------------------

dataset = pydicom.dcmread(f'{data_path}/train/{dcom_list[0]}')
# ------------------------------------------------------

# print out a full listing

# ------------------------------------------------------

for element in dataset:

    print(element)
# -------------------------------------------------------------------------------------------

# helper function to open a dicom image, extract some meta data and return a list of values

# this works with the thread pooler below to reduce wait time

# to get both test and train you will have to switch the data_type argument to the desired

# folder

# -------------------------------------------------------------------------------------------

def collect_dcom_data(dcom_file, data_path=data_path):

    ds = pydicom.dcmread(f'{data_path}/train/{dcom_file}')

    return [ds.PatientID, ds.PatientSex, ds.PatientAge, ds.BodyPartExamined, ds.InstitutionName, 

            ds.ImageType, ds.Modality, ds.PhotometricInterpretation, ds.Rows, ds.Columns]

def collect_dcom_data_test(dcom_file, data_path=data_path):

    ds = pydicom.dcmread(f'{data_path}/test/{dcom_file}')

    return [ds.PatientID, ds.PatientSex, ds.PatientAge, ds.BodyPartExamined, ds.InstitutionName, 

            ds.ImageType, ds.Modality, ds.PhotometricInterpretation, ds.Rows, ds.Columns]
# -------------------------------------------------------------------------------------------

# Multithreaded extract of meta data

# I was using a GCP instance which only provides vCPUs (one thread per CPU)

# your mileage will vary

#

# I've saved csv's of the results so you just skip this part if you want

# -------------------------------------------------------------------------------------------

pool = ThreadPool(cpu_count())  

meta_data = p_map(collect_dcom_data, dcom_list)   

df = pd.DataFrame(meta_data, columns=['PatientID', 'PatientSex', 'PatientAge', 'BodyPartExamined', 'InstitutionName', 

                                      'ImageType', 'Modality', 'PhotometricInterpretation', 

                                      'Rows', 'Columns'])

df.to_csv(output_path + '/' + 'metadata.csv', index=False)





meta_data = p_map(collect_dcom_data_test, dcom_test_list)   

df_test = pd.DataFrame(meta_data, columns=['PatientID', 'PatientSex', 'PatientAge', 'BodyPartExamined', 'InstitutionName',

                                           'ImageType', 'Modality', 'PhotometricInterpretation', 

                                           'Rows', 'Columns'])

df_test.to_csv(output_path + '/' + 'metadata_test.csv', index=False)
df = pd.read_csv(output_path + '/' + 'metadata.csv')

df_test = pd.read_csv(output_path + '/' + 'metadata_test.csv')
# set up the fig

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ax1.set_title('Train')

ax2.set_title('Test')



# create the first chart

df_mods = df.Modality.value_counts()

ax1 = df_mods.plot.pie(ax=ax1)



# create the second chart

df_mods = df_test.Modality.value_counts()

ax2 = df_mods.plot.pie(ax=ax2)
# set up the fig

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ax1.set_title('Train')

ax2.set_title('Test')



# create the first chart

df_type = df.ImageType.value_counts()

ax1 = df_type.plot.pie(ax=ax1, labels=None)



# create the second chart

df_type = df_test.ImageType.value_counts()

ax2 = df_type.plot.pie(ax=ax2, labels=None)
# set up the fig

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ax1.set_title('Train')

ax2.set_title('Test')



# create the first chart

df_cols = df.Columns.value_counts()

df_cols[:10].plot.bar(ax=ax1)



# create the second chart

df_cols = df_test.Columns.value_counts()

df_cols[:10].plot.bar(ax=ax2)



plt.show()
# set up the fig

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ax1.set_title('Train')

ax2.set_title('Test')



# create the first chart

df_rows = df.Rows.value_counts()

df_rows[:10].plot.bar(ax=ax1)



# create the second chart

df_rows = df_test.Rows.value_counts()

df_rows[:10].plot.bar(ax=ax2)



plt.show()
# set up the fig

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ax1.set_title('Train')

ax2.set_title('Test')



# charts

ax1 = df.PatientSex.value_counts().plot.pie(ax=ax1)

ax2 = df_test.PatientSex.value_counts().plot.pie(ax=ax2)



plt.show()

# set up the fig

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ax1.set_title('Train')

ax2.set_title('Test')



# create the first chart

ages = df.PatientAge.unique()

ages.sort()

ax1 = df.PatientAge.value_counts().reindex(ages.tolist()).plot.bar(ax=ax1)



# create the second chart

ages = df_test.PatientAge.unique()

ages.sort()

ax2 = df_test.PatientAge.value_counts().reindex(ages.tolist()).plot.bar(ax=ax2)



plt.show()

# you can display a dcom image easily with matplotlib

plt.imshow(dataset.pixel_array)

plt.show()
# get a list of all of the training jpeg files

jpeg_list = [f for f in os.listdir(f'{data_path}/jpeg/train')]
# -----------------------------------------------

# Displays differences for a single image

# -----------------------------------------------

def display_dif_compare(image_name, nrows=4, ncols=3, figsize=15):

    

    # clean file name and open image in variety of styles

    file_name, _ = os.path.splitext(image_name)

    ds = pydicom.dcmread(f'{data_path}/train/{file_name}.dcm')



    dcom_image = ds.pixel_array

    jpeg = cv2.imread(f'{data_path}/jpeg/train/{file_name}.jpg', cv2.IMREAD_UNCHANGED)

    #jpeg_rgb = jpeg

    jpeg_rgb = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB)

    jpeg_ycc = cv2.cvtColor(jpeg, cv2.COLOR_BGR2YCR_CB)

    dcom_rgb = cv2.cvtColor(dcom_image, cv2.COLOR_YCR_CB2RGB)



    jpeg_ycc_pil = Image.open(f'{data_path}/jpeg/train/{file_name}.jpg')

    jpeg_ycc_pil.draft('YCbCr', None)

    jpeg_ycc_pil.load() 



    # plot the figures (apologies for the brute force inelegance of this block)

    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize, figsize))



    axarr[0, 0].set_title('Raw Dicom')

    axarr[0, 0].imshow(dcom_image)

    axarr[0, 1].set_title('JPEG Raw via cv2')

    axarr[0, 1].imshow(jpeg_rgb)

    axarr[0, 2].set_title('Dif: ' + str(np.sum(dcom_image - jpeg_rgb)//1000000) + ' x 10^5')

    axarr[0, 2].imshow(abs(dcom_image - jpeg_rgb))

    axarr[0, 0].set(ylabel = file_name)

    

    axarr[1, 0].set_title('Raw Dicom')

    axarr[1, 0].imshow(dcom_image)

    axarr[1, 1].set_title('JPEG Raw->YCrCb via cv2')

    axarr[1, 1].imshow(jpeg_ycc)

    axarr[1, 2].set_title('Dif: ' + str(np.sum(dcom_image - jpeg_ycc)//1000000) + ' x 10^5')

    axarr[1, 2].imshow(abs(dcom_image - jpeg_ycc))

    axarr[1, 0].set(ylabel = file_name)

    

    axarr[2, 0].set_title('Dicom Raw->RGB via cv2')

    axarr[2, 0].imshow(dcom_rgb)

    axarr[2, 1].set_title('JPEG Raw via cv2')

    axarr[2, 1].imshow(jpeg_rgb)

    axarr[2, 2].set_title('Dif: ' + str(np.sum(dcom_rgb - jpeg_rgb)//1000000) + ' x 10^5')

    axarr[2, 2].imshow(abs(dcom_rgb - jpeg_rgb))

    axarr[2, 0].set(ylabel = file_name)

    

    axarr[3, 0].set_title('Dicom Raw')

    axarr[3, 0].imshow(dcom_image)

    axarr[3, 1].set_title('JPEG Raw-YCrCb via PIL "draft"')

    axarr[3, 1].imshow(jpeg_ycc_pil)

    axarr[3, 2].set_title('Dif: ' + str(np.sum(dcom_image - jpeg_ycc_pil)//1000000) + ' x 10^5')

    axarr[3, 2].imshow(abs(dcom_image - jpeg_ycc_pil))

    axarr[3, 0].set(ylabel = file_name)
image_name = np.random.choice(jpeg_list)

display_dif_compare(image_name)
file_name, _ = os.path.splitext(image_name)

ds = pydicom.dcmread(f'{data_path}/train/{file_name}.dcm')

image1 = cv2.cvtColor(ds.pixel_array, cv2.COLOR_YCR_CB2RGB)

image2 = cv2.imread(f'{data_path}/jpeg/train/{image_name}', cv2.IMREAD_UNCHANGED)



# tuple to select colors of each channel line

colors = ("r", "g", "b")

channel_ids = (0, 1, 2)



plt.figure(figsize=(20,5))



# create the histogram plot, with three lines, one for

# each color for each image

for channel_id, c in zip(channel_ids, colors):

    histogram1, bin_edges = np.histogram(image1[:, :, channel_id], bins=256, range=(0, 256))



for channel_id, c in zip(channel_ids, colors):

    histogram2, bin_edges = np.histogram(image2[:, :, channel_id], bins=256, range=(0, 256))





# plot the reds for each image

ax1 = plt.subplot(1,3,1)

plt.xlim([0, 256])

plt.plot(bin_edges[0:-1], histogram1, color='r', label='Dicom-RGB via cv2')

plt.plot(bin_edges[0:-1], histogram2, color='r', linestyle='dashed', label='JPEG Raw via cv2')

plt.legend()

plt.xlabel("Color value")

plt.ylabel("Pixels")



# plot the greens for each image

ax2 = plt.subplot(1,3,2, sharey=ax1)

plt.xlim([0, 256])

plt.plot(bin_edges[0:-1], histogram1, color='g', label='Dicom-RGB via cv2')

plt.plot(bin_edges[0:-1], histogram2, color='g', linestyle='dashed', label='JPEG Raw via cv2')

plt.legend()

plt.xlabel("Color value")

plt.ylabel("Pixels")



# plot the greens for each image

ax3 = plt.subplot(1,3,3, sharey=ax1)

plt.xlim([0, 256])

plt.plot(bin_edges[0:-1], histogram1, color='b', label='Dicom-RGB via cv2')

plt.plot(bin_edges[0:-1], histogram2, color='b', linestyle='dashed', label='JPEG Raw via cv2')

plt.legend()

plt.xlabel("Color value")

plt.ylabel("Pixels")
image1 = ds.pixel_array

image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2YCR_CB)



# tuple to select colors of each channel line

colors = ("y", "Cb", "Cr")

channel_ids = (0, 1, 2)



plt.figure(figsize=(20,5))



# create the histogram plot, with three lines, one for

# each color for each image

for channel_id, c in zip(channel_ids, colors):

    histogram1, bin_edges = np.histogram(image1[:, :, channel_id], bins=256, range=(0, 256))



for channel_id, c in zip(channel_ids, colors):

    histogram2, bin_edges = np.histogram(image2[:, :, channel_id], bins=256, range=(0, 256))





# plot the reds for each image

ax1 = plt.subplot(1,3,1)

plt.xlim([0, 256])

plt.plot(bin_edges[0:-1], histogram1, color='y', label='Dicom Raw')

plt.plot(bin_edges[0:-1], histogram2, color='y', linestyle='dashed', label='JPEG Raw-YCrCb via PIL "draft"')

plt.legend()

plt.xlabel("Luminance")

plt.ylabel("Pixels")



# plot the greens for each image

ax2 = plt.subplot(1,3,2, sharey=ax1)

plt.xlim([0, 256])

plt.plot(bin_edges[0:-1], histogram1, color='b', label='Dicom Raw')

plt.plot(bin_edges[0:-1], histogram2, color='b', linestyle='dashed', label='JPEG Raw-YCrCb via PIL "draft"')

plt.legend()

plt.xlabel("Cb")

plt.ylabel("Pixels")



# plot the greens for each image

ax3 = plt.subplot(1,3,3, sharey=ax1)

plt.xlim([0, 256])

plt.plot(bin_edges[0:-1], histogram1, color='r', label='Dicom Raw')

plt.plot(bin_edges[0:-1], histogram2, color='r', linestyle='dashed', label='JPEG Raw-YCrCb via PIL "draft"')

plt.legend()

plt.xlabel("Cr")

plt.ylabel("Pixels")