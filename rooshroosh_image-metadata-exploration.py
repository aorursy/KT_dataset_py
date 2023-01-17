import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import glob

import pydicom



from tqdm import tqdm_notebook
PATH = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/'

field_list = ['PatientAge', 'PatientSex', 'ViewPosition']

attr_list = []



for fp in tqdm_notebook(glob.glob(os.path.join(PATH,'dicom-images-train/*/*/*.dcm'))):

    dataset = pydicom.dcmread(fp)

    obj = {'fp': fp}

    for f in field_list:

        obj[f] = getattr(dataset,f)    

    attr_list.append(obj)

    

frame = pd.DataFrame(attr_list)
frame.nunique()
frame['PatientAge'] = frame['PatientAge'].astype(np.int32)

frame['PatientAge'].sort_values(ascending=False).head()
frame['PatientAge'][frame['PatientAge']>90] = 90
frame['PatientAge'].hist(bins=90)
frame['PatientSex'].value_counts()
frame['ViewPosition'].value_counts()
pd.DataFrame(frame.groupby(['PatientSex','ViewPosition'])['PatientAge'].count())
sns.distplot(frame['PatientAge'][frame['PatientSex']=='F'], bins=45)

sns.distplot(frame['PatientAge'][frame['PatientSex']=='M'], bins=45)
sns.distplot(frame['PatientAge'][frame['ViewPosition']=='AP'], bins=45)

sns.distplot(frame['PatientAge'][frame['ViewPosition']=='PA'], bins=45)
train_mask = pd.read_csv(os.path.join(PATH,'train-rle.csv'))
# This fucntion based on the same name function from kernel:

# https://www.kaggle.com/abhishek/image-mask-augmentations

def rle2mask(rle, width=1024, height=1024):

    mask= np.zeros(width* height)

    array = np.asarray([int(x) for x in rle.split()])

    if array.shape == (1,):

        return mask.reshape(width, height)

    starts = array[0::2]

    lengths = array[1::2]

    current_position = 0

    for index, start in enumerate(starts):

        current_position += start

        mask[current_position:current_position+lengths[index]] = 1

        current_position += lengths[index]

    return mask.reshape(width, height)
MASKS = train_mask[' EncodedPixels'].apply(rle2mask)

MASKS = np.array(MASKS)
fraction = [i.sum()/(1024**2) for i in tqdm_notebook(MASKS)]
train_mask['fraction'] = fraction
mask_stats = train_mask.groupby('ImageId')['fraction'].agg(['sum','count'])
frame.shape, mask_stats.shape
frame['ImageId'] = frame['fp'].apply(lambda x :os.path.split(x)[-1][:-4])
frame = frame.merge(mask_stats.reset_index())
frame['sum'].hist(bins=100, log=True)
frame['log_sum'] = np.log(frame['sum']+0.01)
plt.figure(figsize=(10,10))

sns.scatterplot(y='log_sum', x='PatientAge', data=frame, hue='PatientSex', alpha=0.5)

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(y='log_sum', x='PatientAge', data=frame, hue='ViewPosition', alpha=0.5)

plt.show()
frame['count'].value_counts()
plt.figure(figsize=(10,5))

sns.distplot(frame['PatientAge'][(frame['count']==1) & (frame['sum']>0)], bins=50)

sns.distplot(frame['PatientAge'][frame['count']==2], bins=50)

sns.distplot(frame['PatientAge'][frame['count']==3], bins=50)

plt.show()
plt.figure(figsize=(10,5))

sns.distplot(frame['sum'][(frame['count']==1) & (frame['sum']>0)], bins=50, norm_hist=True)

sns.distplot(frame['sum'][frame['count']==2], bins=50, norm_hist=True)

sns.distplot(frame['sum'][frame['count']==3], bins=50, norm_hist=True)

plt.show()
plt.figure(figsize=(10,5))

sns.distplot(frame['sum'][(frame['ViewPosition']=='AP') & (frame['sum']>0)], bins=50, norm_hist=True)

sns.distplot(frame['sum'][(frame['ViewPosition']=='PA') & (frame['sum']>0)], bins=50, norm_hist=True)

plt.show()