import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import imageio

from matplotlib import pyplot as plt

import scipy.ndimage as ndi

import pydicom
os.listdir('../input/rsna-intracranial-hemorrhage-detection')
train = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')

test = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')
train['patient_id'] = train.ID.str.split('_', expand=True)[1]

train['h_type'] =  train.ID.str.split('_', expand=True)[2]
train.tail()
train_dir = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'

im = imageio.imread(train_dir + 'ID_5c8b5d701.dcm')
pdim = pydicom.dcmread(train_dir + 'ID_000039fa0.dcm')
print(pdim.pixel_array.min(),pdim.pixel_array.max())

print(im.min(),im.max())
fig, axis = plt.subplots(1, 2, figsize=(15,5))

#plt.figure(figsize=(10,10))

axis[0].imshow(pdim.pixel_array, cmap='gray')

axis[1].imshow(im, cmap='gray')
plt.figure(figsize=(10,10))

plt.imshow(im, cmap='gray')
im.flatten.max()-im.flatten.min()+1
hist = ndi.histogram(im, min=im.flatten().min(),

                         max=im.flatten().max(),

                         bins=im.flatten().max()-im.flatten().min()+1)

cdf = hist.cumsum() / hist.sum()

cdf.shape
im_equalized = cdf[im] * im.flatten().max()

fig, axes = plt.subplots(1, 2, figsize=(15,5))

axes[0].imshow(im, cmap='gray')

axes[1].imshow(im_equalized, cmap='gray')

plt.show()
im_blood = np.where((im>=30) & (im<=45), im*50, 0)

plt.imshow(im_blood, cmap=plt.cm.bone)
im_bone = np.where(im>700, im, 0)

plt.imshow(im_bone, cmap='gray')
plt.hist(im.flatten(),bins=30)
pdim.pixel_array
pdim.pixel_array.flatten()
plt.hist(im.flatten())
plt.imshow(pdim.pixel_array)
plt.imshow(im)
im.meta


plt.imshow(im)
plt.imshow(pdim.pixel_array)
im.meta
im.meta['Modality']
im.meta.keys()
plt.figure(figsize=(10,10))

plt.imshow(im, cmap=plt.cm.bone)

plt.axis('off')
im.dtype
im
hist = ndi.histogram(im, min=-2000,

                     max=2500,

                     bins=4500)

hist.shape
plt.plot(hist)
plt.figure(figsize=(15,5))

plt.hist(im)

plt.xticks(np.arange(-1200, 1500, 100.0))

plt.show()
np.where(im>750, im, 0).shape


im[im<-750].shape
fig, axes = plt.subplots(1, 8,figsize=(15,15))

axes[0].imshow(im, cmap=plt.cm.bone)

#axes[1].imshow(np.where(im<=-750, im, 0),cmap=plt.cm.bone)

axes[1].imshow(np.where((im>-1000) & (im<-800), im, 0),cmap=plt.cm.bone)

axes[2].imshow(np.where((im>-775) & (im<-550), im, 0),cmap=plt.cm.bone)

axes[3].imshow(np.where((im>-500) & (im<-300), im, 0),cmap=plt.cm.bone)

axes[4].imshow(np.where((im>-250) & (im<-50), im, 0),cmap=plt.cm.bone)

axes[5].imshow(np.where((im>-5) & (im<200), im, 0),cmap=plt.cm.bone)

#axes[2].imshow(np.where(im>-750, im, 0),cmap=plt.cm.bone)

axes[6].imshow(np.where((im>225) & (im<450), im, 0),cmap=plt.cm.bone)

axes[7].imshow(np.where((im>1000) & (im<2500), im, 0),cmap=plt.cm.bone)
plt.figure(figsize=(10,10))

plt.imshow(np.where((im>-5) & (im<200), im, 0), cmap=plt.cm.bone)

plt.axis('off')
nlabels
filt=ndi.gaussian_filter(im,sigma=1)

mask = filt < -750

labels, nlabels = ndi.label(mask)
plt.imshow(labels, cmap='rainbow')

plt.axis('off')

plt.show()
im = imageio.imread(train_dir + 'ID_5fc337950.dcm')

plt.figure(figsize=(10,10))

plt.imshow(im, cmap=plt.cm.bone)

plt.axis('off')
im.meta
plt.figure(figsize=(15,5))

plt.hist(im)

plt.xticks(np.arange(-1050, 1700, 100.0))

plt.show()
plt.figure(figsize=(10,10))

plt.imshow(np.where((im>-5) & (im<200), im, 0), cmap=plt.cm.bone)

plt.axis('off')
fig, axes = plt.subplots(1, 4,figsize=(15,15))

axes[0].imshow(im, cmap=plt.cm.bone)

#axes[1].imshow(np.where(im<=-750, im, 0),cmap=plt.cm.bone)

axes[1].imshow(np.where((im>40) & (im<600), im, 0),cmap=plt.cm.bone)

axes[2].imshow(np.where((im>80) & (im<200), im, 0),cmap=plt.cm.bone)

axes[3].imshow(np.where((im>600) & (im<2800), im, 0),cmap=plt.cm.bone)