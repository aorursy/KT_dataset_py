import numpy as np

import pandas as pd

import os

import psutil

import glob



# https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_read_dicom.html

import pydicom



print(os.listdir("../input"))

print(os.listdir("../input/siim-acr-pneumothorax-segmentation/"))



from matplotlib import cm

from matplotlib import pyplot as plt



import tensorflow as tf



from keras.models import Model

from keras.layers import Input

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



import sys

sys.path.append('../input/siim-acr-pneumothorax-segmentation/')



from mask_functions import rle2mask

from multiprocessing.pool import Pool, ThreadPool

from joblib import Parallel, delayed



plt.style.use('ggplot')

pd.set_option("display.max_colwidth", 100)
tr_rle = pd.read_csv('../input/siim-train-test/siim/train-rle.csv')

print(tr_rle.shape)

tr_rle.head()
print('{} unique ImageId'.format(tr_rle.ImageId.nunique()))

tmp = tr_rle.groupby('ImageId').agg({'ImageId':['count']}).reset_index()

tmp.columns = ['ImageId', 'count']

id_multi = tmp.loc[tmp['count'] > 1, 'ImageId']

tr_rle[tr_rle['ImageId'].isin(id_multi)].head(10)
# Helper function

def calc_per_sur(rle, width, height):

    mask= np.zeros(width* height)

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        current_position += start

        mask[current_position:current_position+lengths[index]] = 255

        current_position += lengths[index]



    return mask.sum(), lengths.sum()
tmp = tr_rle.loc[tr_rle['ImageId'].isin(id_multi), ' EncodedPixels'].apply(

    lambda x: calc_per_sur(x, 1024, 1024))

tr_rle['Perimeter'] = np.nan

tr_rle['Surface'] = np.nan

tr_rle['Surface_Ratio(%)'] = np.nan

tr_rle.loc[tr_rle['ImageId'].isin(id_multi), 'Perimeter'] = tmp.apply(lambda x: x[0])

tr_rle.loc[tr_rle['ImageId'].isin(id_multi), 'Surface'] = tmp.apply(lambda x: x[1])

tr_rle.loc[tr_rle['ImageId'].isin(id_multi), 'Surface_Ratio(%)'] = tmp.apply(lambda x: x[1] * 100 / 1024**2)

tr_rle.loc[:, 'Perimeter':'Surface_Ratio(%)'].describe()
tr_rle[tr_rle['ImageId'].isin(id_multi)].head(10)
fig, ax = plt.subplots(figsize=(10, 5))

tr_rle['Surface_Ratio(%)'].hist(ax=ax, bins=50, color='deeppink', rwidth=0.9)

ax.set_xlabel('Surface Ratio (%)');
sub = pd.read_csv('../input/siim-acr-pneumothorax-segmentation/sample_submission.csv')

sub.head()
print('{} unique ImageId / {} rows'.format(sub.ImageId.nunique(), len(sub)))

tmp = sub.groupby('ImageId').agg({'ImageId':['count']}).reset_index()

tmp.columns = ['ImageId', 'count']

id_multi = tmp.loc[tmp['count'] > 1, 'ImageId']

print('{} ImageIds have multiple rows.'.format(len(id_multi)))

sub[sub['ImageId'].isin(id_multi)].head(10)
def show_dcm_info(fp, dataset):

    print("Filename.........:", fp.split('/')[-1])

    print("Storage type.....:", dataset.SOPClassUID)

    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print("Patient's name......:", display_name)

    print("Patient id..........:", dataset.PatientID)

    print("Patient's Age.......:", dataset.PatientAge)

    print("Patient's Sex.......:", dataset.PatientSex)

    print("Modality............:", dataset.Modality)

    print("Body Part Examined..:", dataset.BodyPartExamined)

    print("View Position.......:", dataset.ViewPosition)

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing....:", dataset.PixelSpacing)



def plot_pixel_array(dataset, figsize=(10,10)):

    plt.figure(figsize=figsize)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.grid(False)

    plt.show()
for fp in glob.glob('../input/siim-train-test/siim/dicom-images-train/*/*/*.dcm'):

    dataset = pydicom.dcmread(fp)

    show_dcm_info(fp, dataset)

    plot_pixel_array(dataset)

    break
dicom_attr = ['AccessionNumber', 

              'BitsAllocated', 

              'BitsStored', 

              'BodyPartExamined', 

              'Columns', 

              'ConversionType', 

              'HighBit',

              'InstanceNumber',

              'LossyImageCompression',

              'LossyImageCompressionMethod',

              'Modality',

              'PatientAge',

              'PatientBirthDate',

              'PatientID',

              'PatientName',

              'PatientOrientation',

              'PatientSex',

              'PhotometricInterpretation',

#               'PixelData',

              'PixelRepresentation',

              'PixelSpacing',

              'ReferringPhysicianName',

              'Rows',

              'SOPClassUID',

              'SOPInstanceUID',

              'SamplesPerPixel',

              'SeriesDescription',

              'SeriesInstanceUID',

              'SeriesNumber',

              'SpecificCharacterSet',

              'StudyDate',

              'StudyID',

              'StudyInstanceUID',

              'StudyTime',

              'ViewPosition']
def create_features(fp):

    ret = []

    ret.append(fp.split('/')[-1][:-4])

    dataset = pydicom.dcmread(fp)

    for da in dicom_attr:

        ret.append(dataset.__getattr__(da))

    return np.array(ret).T



dicom_df = Parallel(n_jobs=psutil.cpu_count(), verbose=1)(

    (delayed(create_features)(fp) for fp in glob.glob('../input/siim-train-test/siim/dicom-images-train/*/*/*.dcm'))

)

dicom_df = pd.DataFrame(np.array(dicom_df), columns=['ImageId'] + dicom_attr)

dicom_df.head()
dicom_df['PixelSpacing'] = dicom_df['PixelSpacing'].apply(lambda x: x[0])

dicom_df.nunique()
useful_col = ['ImageId',

              'PatientAge', 

              'PatientSex', 

#               'ReferringPhysicianName',

              'ViewPosition', 

              'PixelSpacing']

tr_rle = tr_rle.merge(dicom_df.loc[:, useful_col], how='left', on='ImageId')

tr_rle.loc[:, 'IsDisease'] = (tr_rle.loc[:, ' EncodedPixels'] != ' -1')

tr_rle.head()
tmp = tr_rle.groupby('PatientAge').agg({

    'IsDisease':['count', 'sum']

}).reset_index()

tmp.columns = ['PatientAge', 'count', 'sum']

tmp['PatientAge'] = tmp['PatientAge'].astype('int')

tmp.sort_values(by='PatientAge', inplace=True)

tmp['Ratio'] = tmp['sum'] / tmp['count']



fig, ax = plt.subplots(figsize=(7.5, 15))

ax.barh(range(len(tmp)), tmp['Ratio'], 

        color='deeppink', align='center', height=0.5)

ax.set_yticks(range(len(tmp)))

ax.set_yticklabels(tmp['PatientAge'].astype('str') + ' / (' + tmp['count'].astype('str') + ')', 

                   fontsize=7, color='dimgray')

ax.set_xlabel('Incidence Ratio', color='dimgray')

fig.tight_layout();
tmp = tr_rle.groupby('PatientSex').agg({

    'IsDisease':['count', 'sum']

}).reset_index()

tmp.columns = ['PatientSex', 'count', 'sum']

tmp['Ratio'] = tmp['sum'] / tmp['count']



fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(range(len(tmp)), tmp['Ratio'], 

        color='deeppink', align='center', width=0.5)

ax.set_xticks(range(len(tmp)))

ax.set_xticklabels(tmp['PatientSex'].astype('str') + ' / (' + tmp['count'].astype('str') + ')', 

                   fontsize=10, color='dimgray')

ax.set_ylabel('Incidence Ratio', color='dimgray')

fig.tight_layout();
tmp = tr_rle.groupby('ViewPosition').agg({

    'IsDisease':['count', 'sum']

}).reset_index()

tmp.columns = ['ViewPosition', 'count', 'sum']

tmp['Ratio'] = tmp['sum'] / tmp['count']



fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(range(len(tmp)), tmp['Ratio'], 

        color='deeppink', align='center', width=0.5)

ax.set_xticks(range(len(tmp)))

ax.set_xticklabels(tmp['ViewPosition'].astype('str') + ' / (' + tmp['count'].astype('str') + ')', 

                   fontsize=10, color='dimgray')

ax.set_ylabel('Incidence Ratio', color='dimgray')

fig.tight_layout();
tr_rle['PixelSpacing'] = tr_rle['PixelSpacing'].round(6)

tmp = tr_rle.groupby('PixelSpacing').agg({

    'IsDisease':['count', 'sum']

}).reset_index()

tmp.columns = ['PixelSpacing', 'count', 'sum']

tmp['Ratio'] = tmp['sum'] / tmp['count']



fig, ax = plt.subplots(figsize=(20, 5))

ax.bar(range(len(tmp)), tmp['Ratio'], 

        color='deeppink', align='center', width=0.5)

ax.set_xticks(range(len(tmp)))

ax.set_xticklabels(tmp['PixelSpacing'].astype('str') + ' / (' + tmp['count'].astype('str') + ')', 

                   fontsize=13, color='dimgray')

ax.set_ylabel('Incidence Ratio', color='dimgray')

fig.tight_layout();
num_img = 5 * 3

fig, ax = plt.subplots(nrows=num_img // 5, ncols=5, sharey=True, figsize=(20, num_img // 5 * 4))

axes = ax.ravel()

for q, fp in enumerate(glob.glob('../input/siim-train-test/siim/dicom-images-train/*/*/*.dcm')[:num_img]):

    dataset = pydicom.dcmread(fp)

    axes[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)

    axes[q].grid(False)
num_img = 5 * 3

fig, ax = plt.subplots(nrows=num_img // 5, ncols=5, sharey=True, figsize=(20, num_img // 5 * 4))

axes = ax.ravel()

for q, fp in enumerate(glob.glob('../input/siim-train-test/siim/dicom-images-train/*/*/*.dcm')[:num_img]):

    dataset = pydicom.dcmread(fp)

    axes[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)

    axes[q].grid(False)

    flag = (tr_rle.loc[:, 'ImageId'] == fp.split('/')[-1][:-4])

    if tr_rle.loc[flag, ' EncodedPixels'].values[0] != ' -1':

        mask = rle2mask(tr_rle.loc[flag, ' EncodedPixels'].values[0], 1024, 1024).T

        axes[q].set_title('Pneumothorax', fontsize=10)

        mask[mask == 0] = np.nan

        axes[q].imshow(mask, alpha = 0.2, vmin = 0, vmax = 1)

    else:

        axes[q].set_title('No Diseases', fontsize=10)
samp_id = tr_rle.loc[tr_rle.IsDisease == True, "ImageId"].sample(1).values[0]

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

axes = ax.ravel()

fp = glob.glob('../input/siim-train-test/siim/dicom-images-train/*/*/{}.dcm'.format(samp_id))[0]

dataset = pydicom.dcmread(fp)

axes[0].imshow(dataset.pixel_array, cmap=plt.cm.bone)

axes[0].grid(False)

axes[1].imshow(dataset.pixel_array, cmap=plt.cm.bone)

axes[1].grid(False)

mask = rle2mask(tr_rle.loc[tr_rle.ImageId == samp_id, ' EncodedPixels'].values[0], 1024, 1024).T

mask[mask == 0] = np.nan

axes[1].imshow(mask, alpha = 0.3, vmin = 0, vmax = 1);