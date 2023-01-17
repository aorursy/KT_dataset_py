import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt





#plotly imports

!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')
train_dir = '../input/osic-pulmonary-fibrosis-progression/train/'

test_dir = '../input/osic-pulmonary-fibrosis-progression/test/'





train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

print('Training Dataframe shape: ', train_df.shape)



train_df.head(10)
# Let's have a look at the detailed info about the dataframes

print('Training Dataframe Details: ')

print(train_df.info())



print('\n\nTest Dataframe Details: ')

print(test_df.info())





print('Number of patients in training set:',

      len(os.listdir(train_dir)))

print('Number of patients in test set:',

     len(os.listdir(test_dir)))
# Creating unique patient lists and their properties. 

patient_ids = os.listdir(train_dir)

patient_ids = sorted(patient_ids)



#Creating new rows

no_of_instances = []

age = []

sex = []

smoking_status = []



for patient_id in patient_ids:

    patient_info = train_df[train_df['Patient'] == patient_id].reset_index()

    no_of_instances.append(len(os.listdir(train_dir + patient_id)))

    age.append(patient_info['Age'][0])

    sex.append(patient_info['Sex'][0])

    smoking_status.append(patient_info['SmokingStatus'][0])



#Creating the dataframe for the patient info    

patient_df = pd.DataFrame(list(zip(patient_ids, no_of_instances, age, sex, smoking_status)), 

                                 columns =['Patient', 'no_of_instances', 'Age', 'Sex', 'SmokingStatus'])

print(patient_df.info())

patient_df.head()
patient_df['Sex'].value_counts(normalize = True).iplot(kind = 'bar', 

                                                        color = 'blue', 

                                                        yTitle = 'Unique patient count',

                                                        xTitle = 'Gender',

                                                        title = 'Gender Distribution of the unique patients')
import scipy



data = patient_df.Age.tolist()

plt.figure(figsize=(18,6))

# Creating the main histogram

_, bins, _ = plt.hist(data, 15, density=1, alpha=0.5)



# Creating the best fitting line with mean and standard deviation

mu, sigma = scipy.stats.norm.fit(data)

best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)

plt.plot(bins, best_fit_line, color = 'b', linewidth = 3, label = 'fitting curve')

plt.title(f'Age Distribution [ mean = {"{:.2f}".format(mu)}, standard_dev = {"{:.2f}".format(sigma)} ]', fontsize = 18)

plt.xlabel('Age -->')

plt.show()



patient_df['Age'].iplot(kind='hist',bins=25,color='blue',xTitle='Percent distribution',yTitle='Count')
plt.figure(figsize=(16, 6))

sns.kdeplot(patient_df.loc[patient_df['Sex'] == 'Male', 'Age'], label = 'Male',shade=True)

sns.kdeplot(patient_df.loc[patient_df['Sex'] == 'Female', 'Age'], label = 'Female',shade=True)

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
patient_df['SmokingStatus'].value_counts(normalize=True).iplot(kind='bar',

                                                      yTitle='Percentage', 

                                                      linecolor='black', 

                                                      opacity=0.8,

                                                      color='blue',

                                                      theme='pearl',

                                                      bargap=0.5,

                                                      title='SmokingStatus Distribution')
patient_df.groupby(['SmokingStatus', 'Sex']).count()['Patient'].unstack().iplot(kind='bar', 

                                                                                yTitle = 'Unique Patient Count',

                                                                                title = 'Gender vs SmokingStatus' )
plt.figure(figsize=(16, 6))

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently smokes',shade=True)

# Labeling of plot

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
from ipywidgets import interact  #, interactive, IntSlider, ToggleButtons



def patient_lookup(patient_id):

    print(train_df[train_df['Patient'] == patient_id])

    patient_info = train_df[train_df['Patient'] == patient_id].reset_index()

    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (15, 5))

    ax1.plot(patient_info['Weeks'].tolist() , patient_info['FVC'].tolist(), marker = '*', linewidth = 3,color = 'r', markeredgecolor = 'b')

    ax1.set_title('FVC Deterioriation over the Weeks')

    ax1.set_xlabel('Weeks -->')

    ax1.set_ylabel('FVC')

    ax1.grid(True)

    

    ax2.plot(patient_info['Weeks'].tolist() , patient_info['Percent'].tolist(),marker = '*', linewidth = 3,

            color = 'r', markeredgecolor = 'b' )

    ax2.set_title('Percent change over the weeks')

    ax2.set_xlabel('Weeks -->')

    ax2.set_ylabel('Percent(of adult capacity)')

    ax2.grid(True)

    fig.suptitle(f'P_ID: {patient_id}', fontsize = 20) 

    

    

    

interact(patient_lookup, patient_id = patient_ids)
import random

import pydicom

def explore_dicoms(patient_id, instance):

    RefDs = pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/' + 

                            patient_id +

                            '/' + 

                            str(instance) + '.dcm')

    plt.figure(figsize=(10, 5))

    plt.imshow(RefDs.pixel_array, cmap='gray');

    plt.title(f'P_ID: {patient_id}\nInstance: {instance}')

    plt.axis('off')





def show_ct_scans(patient_id):

    no_of_instances = int(patient_df[patient_df['Patient'] == patient_id]['no_of_instances'].values[0])

    files = sorted(random.sample(range(1, no_of_instances), 9))

    rows = 3

    cols = 3

    fig = plt.figure(figsize=(12,12))

    for idx in range(1, rows*cols+1):

        fig.add_subplot(rows, cols, idx)

        RefDs = pydicom.dcmread(train_dir + patient_id + '/' + str(files[idx-1]) + '.dcm')

        plt.imshow(RefDs.pixel_array, cmap='gray')

        plt.title(f'Instance: {files[idx-1]}')

        plt.axis(False)

        fig.add_subplot

    fig.suptitle(f'P_ID: {patient_id}') 

    plt.show()
# show_ct_scans(patient_ids[0])

interact(show_ct_scans,patient_id = patient_ids)
import warnings

warnings.filterwarnings('ignore')

import pandas_profiling as pdp

unique_patient_profile  = pdp.ProfileReport(patient_df)
patient_ids = os.listdir('../input/osic-pulmonary-fibrosis-progression/train/')

patient_id = 'ID00267637202270790561585'

dicom_filenames = os.listdir('../input/osic-pulmonary-fibrosis-progression/train/' + patient_id)

dicom_paths = ['../input/osic-pulmonary-fibrosis-progression/train/' + patient_id + '/'+ file for file in dicom_filenames]

    

def load_scan(paths):

    slices = [pydicom.read_file(path ) for path in paths]

    slices.sort(key = lambda x: int(x.InstanceNumber), reverse = True)

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness 

    return slices



def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans])

    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)   

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)
# set path and load files 

patient_dicom = load_scan(dicom_paths)

patient_pixels = get_pixels_hu(patient_dicom)

#sanity check

plt.imshow(patient_pixels[46], cmap=plt.cm.bone)

plt.axis(False)

plt.show()
# skimage image processing packages

from skimage import measure, morphology

from skimage.morphology import ball, binary_closing

from skimage.measure import label, regionprops

import copy





def largest_label_volume(im, bg=-1):

    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]

    vals = vals[vals != bg]

    if len(counts) > 0:

        return vals[np.argmax(counts)]

    else:

        return None

    

def segment_lung_mask(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2. 

    # 0 is treated as background, which we do not want

    binary_image = np.array(image >= -700, dtype=np.int8)+1

    labels = measure.label(binary_image)

 

    # Pick the pixel in the very corner to determine which label is air.

    # Improvement: Pick multiple background labels from around the  patient

    # More resistant to “trays” on which the patient lays cutting the air around the person in half

    background_label = labels[0,0,0]

 

    # Fill the air around the person

    binary_image[background_label == labels] = 2

 

    # Method of filling the lung structures (that is superior to 

    # something like morphological closing)

    if fill_lung_structures:

        # For every slice we determine the largest solid structure

        for i, axial_slice in enumerate(binary_image):

            axial_slice = axial_slice - 1

            labeling = measure.label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)

 

            if l_max is not None: #This slice contains some lung

                binary_image[i][labeling != l_max] = 1

    binary_image -= 1 #Make the image actual binary

    binary_image = 1-binary_image # Invert it, lungs are now 1

 

    # Remove other air pockets inside body

    labels = measure.label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None: # There are air pockets

        binary_image[labels != l_max] = 0

 

    return binary_image
# get masks 

segmented_lungs = segment_lung_mask(patient_pixels, fill_lung_structures=False)

segmented_lungs_fill = segment_lung_mask(patient_pixels, fill_lung_structures=True)

internal_structures = segmented_lungs_fill - segmented_lungs



# isolate lung from chest

copied_pixels = copy.deepcopy(patient_pixels)

for i, mask in enumerate(segmented_lungs_fill): 

    get_high_vals = mask == 0

    copied_pixels[i][get_high_vals] = 0

seg_lung_pixels = copied_pixels

# sanity check

f, ax = plt.subplots(1,2, figsize=(10,6))

ax[0].imshow(patient_pixels[46], cmap=plt.cm.bone)

ax[0].axis(False)

ax[0].set_title('Original')

ax[1].imshow(seg_lung_pixels[46], cmap=plt.cm.bone)

ax[1].axis(False)

ax[1].set_title('Segmented')

plt.show()
f, ax = plt.subplots(2,2, figsize = (10,10))



# pick random slice 

slice_id = 46



ax[0,0].imshow(patient_pixels[slice_id], cmap=plt.cm.bone)

ax[0,0].set_title('Original Dicom')

ax[0,0].axis(False)





ax[0,1].imshow(segmented_lungs_fill[slice_id], cmap=plt.cm.bone)

ax[0,1].set_title('Lung Mask')

ax[0,1].axis(False)



ax[1,0].imshow(seg_lung_pixels[slice_id], cmap=plt.cm.bone)

ax[1,0].set_title('Segmented Lung')

ax[1,0].axis(False)



ax[1,1].imshow(seg_lung_pixels[slice_id], cmap=plt.cm.bone)

ax[1,1].imshow(internal_structures[slice_id], cmap='jet', alpha=0.7)

ax[1,1].set_title('Segmentation with \nInternal Structure')

ax[1,1].axis(False)
# slide through dicom images using a slide bar 

plt.figure(1)

def dicom_animation(x):

    plt.imshow(patient_pixels[x], cmap = plt.cm.gray)

    return x

interact(dicom_animation, x=(0, len(patient_pixels)-1))
import imageio

from IPython import display

print('Original Image Slices before processing')

imageio.mimsave(f'./{patient_id}.gif', patient_pixels, duration=0.1)

display.Image(f'./{patient_id}.gif', format='png')
print('Lung Segmentation Mask')

imageio.mimsave(f'./{patient_id}.gif', segmented_lungs_fill, duration=0.1)

display.Image(f'./{patient_id}.gif', format='png')
print('Segmented Part of Lung Tissue')

imageio.mimsave(f'./{patient_id}.gif', seg_lung_pixels, duration=0.1)

display.Image(f'./{patient_id}.gif', format='png')
from skimage.morphology import opening, closing, binary_dilation

from skimage.morphology import disk



def plot_comparison(original, filtered, filter_name):



    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,

                                   sharey=True)

    ax1.imshow(original, cmap=plt.cm.gray)

    ax1.set_title('original')

    ax1.axis('off')

    ax2.imshow(filtered, cmap=plt.cm.gray)

    ax2.set_title(filter_name)

    ax2.axis('off')
original = segmented_lungs_fill[46]



rows = 4

cols = 4

f, ax = plt.subplots(rows, cols, figsize = (15,12))



for i in range(rows*cols):

    if i==0:

        ax[0,0].imshow(original, cmap = plt.cm.gray)

        ax[0,0].set_title('Original')

        ax[0,0].axis(False)

    else:

        closed = closing(original, disk(i))

        ax[int(i/rows),int(i % rows)].set_title(f'closed disk({i})')

        ax[int(i/rows),int(i % rows)].imshow(closed, cmap = plt.cm.gray)

        ax[int(i/rows),int(i % rows)].axis('off')

plt.show()   
original_image = patient_pixels[46]

original = segmented_lungs_fill[46]

f, ax = plt.subplots(rows, cols, figsize = (15,15))



for i in range(rows*cols):

    if i==0:

        ax[0,0].imshow(original_image, cmap = plt.cm.gray)

        ax[0,0].set_title('Original')

        ax[0,0].axis(False)

    else:

        closed = closing(original, disk(i))

        ax[int(i/rows),int(i % rows)].set_title(f'closed with disk({i})')

        ax[int(i/rows),int(i % rows)].imshow(original_image * closed, cmap = plt.cm.gray)

        ax[int(i/rows),int(i % rows)].axis('off')

plt.show()   
segmented_output = [image * binary_dilation(closing(mask, disk(20)), disk(4)) 

                    for image, mask in zip(patient_pixels, segmented_lungs_fill )]
print('Segmented Part of Lung Tissue')

imageio.mimsave(f'segmented_output_{patient_id}.gif', segmented_output, duration=0.1)

display.Image(f'segmented_output_{patient_id}.gif', format='png')
rows = 4

cols = 4

f, ax = plt.subplots(rows, cols, figsize = (15,15))



for i in range(rows*cols):

    ax[int(i/rows),int(i % rows)].set_title(f'slice({i*2+25})')

    ax[int(i/rows),int(i % rows)].imshow(segmented_output[i*2+25], cmap = plt.cm.gray)

    ax[int(i/rows),int(i % rows)].axis('off')

plt.show()
unique_patient_profile