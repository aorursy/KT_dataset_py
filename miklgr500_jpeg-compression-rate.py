import os



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from skimage.io import imread

from tqdm.notebook import tqdm
PATH = '../input/alaska2-image-steganalysis'

SRNet = '../input/alaska2-srnet-baseline-inference'

sub = pd.read_csv(os.path.join(SRNet, 'submission.csv'))
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

    if c > 0.95:

        sub.loc[sub.Id == impath, 'Label'] = 1. - 1e-3
plt.figure(figsize=(10,10))



plt.axvline(0.75, color='orange')

plt.axvline(0.90, color='orange')

plt.axvline(0.95, color='orange')

plt.axvspan(0., 0.95, color='green', alpha=0.25)

plt.axvspan(0.95, 1.0, color='red', alpha=0.25)

sns.distplot(list(compressions.values()));
sub.to_csv('submission.csv', index=None)
sub.head()