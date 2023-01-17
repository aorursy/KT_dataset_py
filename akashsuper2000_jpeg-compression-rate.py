import os



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from skimage.io import imread

from tqdm.notebook import tqdm
PATH = '../input/alaska2-image-steganalysis'

sub = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))

sub
sub2 = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))

sub2
sub3 = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))

sub3
class JPEGImageCompressionRateDeterminer:

    def __call__(self, image_path):

        image = imread(image_path)

        w, h, c = image.shape

        

        # theoretical image size

        b = w*h*3

        

        # real image file size in bytes

        s = os.stat(image_path).st_size

        return (b - s) / b 
compression_rate_determiner = JPEGImageCompressionRateDeterminer()



compressions = {}



dir_path = os.path.join(PATH, 'Test')

for impath in tqdm(sub.Id.values):

    c = compression_rate_determiner(os.path.join(dir_path, impath))

    compressions[impath] = c

    sub.loc[sub.Id == impath, 'Label'] = c
compression_rate_determiner = JPEGImageCompressionRateDeterminer()



compressions = {}



dir_path = os.path.join(PATH, 'Test')

for impath in tqdm(sub.Id.values):

    c = compression_rate_determiner(os.path.join(dir_path, impath))

    compressions[impath] = c

    sub2.loc[sub.Id == impath, 'Label'] = c**2
compression_rate_determiner = JPEGImageCompressionRateDeterminer()



compressions = {}



dir_path = os.path.join(PATH, 'Test')

for impath in tqdm(sub.Id.values):

    c = compression_rate_determiner(os.path.join(dir_path, impath))

    compressions[impath] = c

    if c < 0.75:

        sub3.loc[sub.Id == impath, 'Label'] = 0.000001

    elif c >= 0.75 and c < 0.90:

        sub3.loc[sub.Id == impath, 'Label'] = 1. - 1e-3 - (c - 0.75)

    elif c >= 0.90 and c < 0.95:

        sub3.loc[sub.Id == impath, 'Label'] = 1. - 1e-3 - (c - 0.90)

    else:

        sub3.loc[sub.Id == impath, 'Label'] = 1. - 1e-3 - (c - 0.95)
plt.figure(figsize=(10,10))



plt.axvline(0.75, color='orange')

plt.axvline(0.90, color='orange')

plt.axvline(0.95, color='orange')

plt.axvspan(0., 0.95, color='green', alpha=0.25)

plt.axvspan(0.95, 1.0, color='red', alpha=0.25)

sns.distplot(list(compressions.values()));
sub.to_csv('submission.csv', index=None)

sub2.to_csv('submission2.csv', index=None)

sub3.to_csv('submission3.csv', index=None)
sub.head()