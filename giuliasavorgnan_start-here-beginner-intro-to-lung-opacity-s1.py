%matplotlib inline
import matplotlib.pyplot as plt
import pylab
import numpy as np
import pydicom
import pandas as pd
from glob import glob
import os
from matplotlib.patches import Rectangle

datapath = '../input/'
# counting the number of files in the image folder
!ls ../input/stage_1_train_images/ | wc
df_box = pd.read_csv(datapath+'stage_1_train_labels.csv')
print('Number of rows (unique boxes per patient) in main train dataset:', df_box.shape[0])
print('Number of unique patient IDs:', df_box['patientId'].nunique())
df_box.head(6)
df_box.groupby('Target').size().plot.bar()
print(df_box.groupby('Target').size() / df_box.shape[0])
df_aux = pd.read_csv(datapath+'stage_1_detailed_class_info.csv')
print('Number of rows in auxiliary dataset:', df_aux.shape[0])
print('Number of unique patient IDs:', df_aux['patientId'].nunique())
df_aux.head(6)
df_aux.groupby('class').size().plot.bar()
print(df_aux.groupby('class').size() / df_aux.shape[0])
assert df_box.loc[df_box['Target']==0].shape[0] == df_aux.loc[df_aux['class'].isin(['Normal', \
    'No Lung Opacity / Not Normal'])].shape[0], 'Number of negative targets does not match between main and auxiliary dataset.'
assert df_box.loc[df_box['Target']==1].shape[0] == df_aux.loc[df_aux['class'] == 'Lung Opacity'].shape[0], \
    'Number of positive targets does not match between main and auxiliary dataset.'
assert df_box['patientId'].values.tolist() == df_aux['patientId'].values.tolist(), 'PatientId columns are different.'
df_train = pd.concat([df_box, df_aux.drop(labels=['patientId'], axis=1)], axis=1)
df_train.head(6)
df_train.groupby(['class', 'Target']).size()
df_train.isnull().any()
# when target==1, are any of the box coordinates null? (should all be false)
df_train.loc[df_train['Target']==1, ['x', 'y', 'width', 'height']].isnull().any()
# when target==0, are all of the box coordinates null? (should all be true)
df_train.loc[df_train['Target']==0, ['x', 'y', 'width', 'height']].isnull().all()
# sample of image filenames
!ls -U ../input/stage_1_train_images/ | head -6
# check that there is an image for each unique patient ID
assert sorted(df_train['patientId'].unique().tolist()) == sorted([f[:-4] for f in os.listdir(datapath+'stage_1_train_images/')]), \
    'Discrepancy between patient IDs and radiograph images.'
# have a look at the header meta-data of an image 
pId = df_train['patientId'].sample(1).values[0]    
dcmdata = pydicom.read_file(datapath+'stage_1_train_images/'+pId+'.dcm')
print(dcmdata)
# extract the raw pixel image and look at its properties
dcmimg = dcmdata.pixel_array
print(type(dcmimg))
print(dcmimg.dtype)
print(dcmimg.shape)
# visualize the corresponding radiograph image
plt.figure(figsize=(20,10))
plt.imshow(dcmimg, cmap=pylab.cm.binary)
plt.axis('off')
def get_boxes_per_patient(df, pId):
    '''
    Given the dataset and one patient ID, 
    return an array of all the bounding boxes and their labels associated with that patient ID.
    Example of return: 
    array([[x1, y1, width1, height1, class1, target1],
           [x2, y2, width2, height2, class2, target2]])
    '''
    
    boxes = df.loc[df['patientId']==pId][['x', 'y', 'width', 'height', 'class', 'Target']].values
    return boxes
def get_dcm_data_per_patient(pId, sample='train'):
    '''
    Given one patient ID and the sample name (train/test), 
    return the corresponding dicom data.
    '''
    return pydicom.read_file(datapath+'stage_1_'+sample+'_images/'+pId+'.dcm')
def display_image_per_patient(df, pId, angle=0.0, sample='train'):
    '''
    Given one patient ID and the dataset,
    display the corresponding dicom image with overlaying boxes and class annotation.
    To be implemented: Optionally input the image rotation angle, in case of data augmentation.
    '''
    dcmdata = get_dcm_data_per_patient(pId, sample=sample)
    dcmimg = dcmdata.pixel_array
    boxes = get_boxes_per_patient(df, pId)
    plt.figure(figsize=(20,10))
    plt.imshow(dcmimg, cmap=pylab.cm.binary)
    plt.axis('off')
    
    class_color_dict = {'Normal' : 'green',
                        'No Lung Opacity / Not Normal' : 'orange',
                        'Lung Opacity' : 'red'}

    if len(boxes)>0:
        for box in boxes:
            # extracting individual coordinates and labels
            x, y, w, h, c, t = box 
            # create a rectangle patch
            patch = Rectangle((x,y), w, h, color='red', 
                              angle=angle, fill=False, lw=4, joinstyle='round', alpha=0.6)
            # get current axis and draw rectangle
            plt.gca().add_patch(patch)
            
    # add annotation text
    plt.text(10, 50, c, color=class_color_dict[c], size=20, 
             bbox=dict(edgecolor=class_color_dict[c], facecolor='none', alpha=0.5, lw=2))
            
# run this cell multiple times to visualize different examples
pId = df_train['patientId'].sample(1).values[0]    
display_image_per_patient(df_train, pId, sample='train')
def get_metadata_per_patient(pId, attribute, sample='train'):
    '''
    Given a patient ID, return useful meta-data from the corresponding dicom image header.
    Return: 
    attribute value
    '''
    # get dicom image
    dcmdata = get_dcm_data_per_patient(pId, sample=sample)
    # extract attribute values
    attribute_value = getattr(dcmdata, attribute)
    return attribute_value
# create list of attributes that we want to extract (manually edited after checking which attributes contained valuable information)
attributes = ['PatientSex', 'PatientAge', 'ViewPosition']
for a in attributes:
    df_train[a] = df_train['patientId'].apply(lambda x: get_metadata_per_patient(x, a, sample='train'))
# convert patient age from string to numeric
df_train['PatientAge'] = df_train['PatientAge'].apply(pd.to_numeric, errors='coerce')
# remove a few outliers
df_train['PatientAge'] = df_train['PatientAge'].apply(lambda x: x if x<120 else np.nan)
df_train.head()
# look at age statistics between positive and negative target groups
df_train.drop_duplicates('patientId').groupby('Target')['PatientAge'].describe()
# look at gender statistics between positive and negative target groups
df_train.drop_duplicates('patientId').groupby(['PatientSex', 'Target']).size() / df_train.drop_duplicates('patientId').groupby(['PatientSex']).size()
# look at patient position statistics between positive and negative target groups
df_train.drop_duplicates('patientId').groupby(['ViewPosition', 'Target']).size() / df_train.drop_duplicates('patientId').groupby(['ViewPosition']).size()
# absolute split of view position
df_train.groupby('ViewPosition').size()
patientIDs_test = [f[:-4] for f in os.listdir(datapath+'stage_1_test_images/')]
df_test = pd.DataFrame(data={'patientId' : patientIDs_test})
attributes = ['PatientSex', 'PatientAge', 'ViewPosition']
for a in attributes:
    df_test[a] = df_test['patientId'].apply(lambda x: get_metadata_per_patient(x, a, sample='test'))
# convert patient age from string to numeric
df_test['PatientAge'] = df_test['PatientAge'].apply(pd.to_numeric, errors='coerce')
# remove a few outliers
df_test['PatientAge'] = df_test['PatientAge'].apply(lambda x: x if x<120 else np.nan)
df_test.head()
# save dataframes
df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)
