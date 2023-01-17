import os



import numpy as np 

import pandas as pd 

import skimage

import cv2 as cv

import pydicom

import json



from scipy.spatial.distance import euclidean





from matplotlib import pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/rsna-generate-metadata-csvs/train_metadata.csv')

train['image_dir'] = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'



test = pd.read_csv('../input/rsna-generate-metadata-csvs/test_metadata.csv')

test['image_dir'] = '../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'



metadata = pd.concat([train, test])

train, test = None, None

metadata['image_orientation'] = metadata['ImageOrientationPatient'].apply(lambda x: np.array(eval(x)).astype(np.float32))
orientations = metadata['ImageOrientationPatient'].unique()

orientations = np.array([eval(o) for o in orientations]).astype(np.float32)



mean_orientation = orientations.mean(axis=0)

print('Mean Orientation: {}'.format(mean_orientation))

orientations = orientations[np.argsort([euclidean(o, mean_orientation) for o in orientations])[::-1]]

orientations = pd.DataFrame(orientations)
print("Common of Orientations:")

orientations.tail(5)
print("Orientations of Intrest:")

orientations.head(30)
# finds other orientations with similar euclidean distance

def match_orientation(target, tol=1e-6):

    metadata['dist'] = metadata['image_orientation'].apply(lambda x: euclidean(x, target))

    return metadata[metadata['dist'] < tol]



def show(dcm, ax, rot=0):

    dcm = pydicom.dcmread(dcm.image_dir + dcm.SOPInstanceUID + '.dcm')

    img = dcm.pixel_array  * dcm.RescaleSlope + dcm.RescaleIntercept

    img = np.clip(img, 0, 100)

    img = skimage.transform.rotate(img, rot)

    return ax.imshow(img, cmap='bone')
o =orientations.values[0]

prtinnp.degrees(np.arctan2(o[0], ))
N_SAMPLES = 4



for o in orientations.values[::10]:

    x = match_orientation(o)

    f, ax = plt.subplots(1, N_SAMPLES, figsize=(20, 5))

    for i in range(N_SAMPLES):

        show(x.iloc[i*3], ax[i], rot=0)



    f.suptitle('ORIENTATION: {}'.format(o))

    plt.show()