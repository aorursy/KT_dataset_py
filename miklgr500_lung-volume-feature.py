import os



import numpy as np

import pandas as pd

import pydicom



from skimage.measure import label,regionprops

from skimage.segmentation import clear_border

from tqdm.notebook import tqdm 

from multiprocessing import Pool



import matplotlib.pyplot as plt
class Detector:

    def __call__(self, x):

        raise NotImplementedError('Abstract') 



class ThrDetector(Detector):

    def __init__(self, thr=-400):

        self.thr = thr

        

    def __call__(self, x):

        x = pydicom.dcmread(x)

        img = x.pixel_array

        img = (img + x.RescaleIntercept) / x.RescaleSlope

        img = img < self.thr

        

        img = clear_border(img)

        img = label(img)

        areas = [r.area for r in regionprops(img)]

        areas.sort()

        if len(areas) > 2:

            for region in regionprops(img):

                if region.area < areas[-2]:

                    for coordinates in region.coords:                

                        img[coordinates[0], coordinates[1]] = 0

        img = img > 0

        return np.int32(img)

  



class Integral:

    def __init__(self, detector: Detector):

        self.detector = detector

    

    def __call__(self, xs):

        raise NotImplementedError('Abstract')

        



class MeanIntegral(Integral):

    def __call__(self, xs):

        with Pool(4) as p:

            masks = p.map(self.detector, xs) 

        return np.mean(masks)
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

train_data = {}

for p in train.Patient.values:

    train_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')
keys = [k for k in list(train_data.keys()) if k not in ['ID00011637202177653955184', 'ID00052637202186188008618']]
integral = MeanIntegral(ThrDetector()) 
volume = {}

for k in tqdm(keys, total=len(keys)):

    x = []

    for i in train_data[k]:

        x.append(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}') 

    volume[k] = integral(x)
for k in tqdm(train.Patient.values):

    if k in ['ID00011637202177653955184', 'ID00052637202186188008618']:

        continue

    train.loc[train.Patient == k,'v'] = volume[k]
plt.figure(figsize=(10, 10))



plt.plot(train.v, train.FVC, '.')
plt.figure(figsize=(10, 10))



plt.plot(train.v, train.Percent, '.')