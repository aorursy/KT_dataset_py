# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# glob is used to find data with expressions (regex)

from glob import glob



train_data = sorted(glob('../input/pneumothorax/dicom-images-train/*/*/*.dcm'))

test_data  = sorted(glob('../input/pneumothorax/dicom-images-test/*/*/*.dcm'))



print((len(train_data)))

print((len(test_data)))
# presenting samples of the training data

from matplotlib import pyplot as plt

import pydicom



fig = plt.figure(figsize=(20, 20))

for position in range(1,21):

    sample_dcm = np.random.choice(train_data)

    sample_img = pydicom.read_file(sample_dcm).pixel_array

    plt.subplot(5,4,position)

    plt.imshow(sample_img, cmap = 'bone')

    plt.axis('off')

plt.tight_layout()        

plt.show()
import pandas as pd

# defaultdict doesn't throw error if you tried non-existed item, instead it creates it

from collections import defaultdict



rle_mask_csv = pd.read_csv('../input/pneumothorax/train-rle.csv')

rle_mask_csv.columns = [col.replace(' ','') for col in rle_mask_csv.columns]

#images could have multiple annotation

# the dictionary was initialized with list, this means each image will have a list of RLE 

rle_masks = defaultdict(list)



for image_id, rle in zip(rle_mask_csv['ImageId'], rle_mask_csv['EncodedPixels']):

    rle_masks[image_id].append(rle)



# find the images that don't have RLE annotation

annotated = {k: v for k, v in rle_masks.items() if v[0] != ' -1'}

print("%d of %d images are annotated" % (len(annotated), len(rle_masks)))

print("Missing: ", len(train_data) - len(rle_masks))
