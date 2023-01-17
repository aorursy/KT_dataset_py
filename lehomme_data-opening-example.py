import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

import imageio

from tensorflow.keras.preprocessing import image







img_x=320

img_y=240

img_path = "/kaggle/input/deep-person-detection-on-non-cenital-data/DATASET_SINTETICO/TRAIN_DATA/INPUT/Image%05d.png" % (10)

input = imageio.imread(img_path)

input = image.img_to_array(input)

input = input/65535



img_path = "/kaggle/input/deep-person-detection-on-non-cenital-data/DATASET_SINTETICO/TRAIN_DATA/OUTPUT/Image%05d.png" % (10)

output = imageio.imread(img_path)

output = image.img_to_array(output)

output = output/255



plt.figure(1)

plt.imshow(input[:,:,0])

plt.figure(2)

plt.imshow(output[:,:,0])

plt.show()
