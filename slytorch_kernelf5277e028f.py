import numpy as np

import pandas as pd

import skimage.io as sio

import matplotlib.pyplot as plt



df = pd.read_csv('../input/train_masks.csv')



# I just put this in here because you will likely be training per image despite there are multiple instances (and masks) per image.

# This will get all the unique ImageIds.

uniques = df.groupby(['ImageId']).agg(['count'])



print(uniques)
# The ImageId actually becomes the name, so access them this way.

test = uniques.index[0]



# this is just the file directory layout assuming that your working dir includes train_images folder.

img = sio.imread('../input/train_images/' + test + '/images/' + test + ".png")



my_array = np.zeros((img.shape[0] * img.shape[1]))



# get the rows (instances) for that ImageId

test_rows = df.loc[df['ImageId'] == test]

length = len(test_rows.index)





for i in range(0, length):

  encoded = test_rows.iloc[i]['EncodedPixels']

  draw_single_instance(my_array, encoded)





tim = np.reshape(my_array, (img.shape[1], img.shape[0]))



final = np.transpose(tim) # some people are column majors

plt.show(final)