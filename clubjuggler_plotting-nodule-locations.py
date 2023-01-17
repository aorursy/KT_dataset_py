import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from skimage import io

import glob

import os
# Loading data and getting the healthy images. Plotting the locations of the nodules will be easier on a healthy chest example

data = pd.read_csv('../input/Data_Entry_2017.csv', index_col = 'Image Index')

healthy = data.loc[data['Finding Labels'] == 'No Finding']



# Reading the bounding box data for the nodules. There are 79 bounding box examples for the nodules.

# Data is read in, unnecessary cols dropped and nodule lines are extracted

bbox_data = pd.read_csv('../input/BBox_List_2017.csv', index_col = 'Image Index')

cols_to_drop = ['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8']

bbox_data.drop(cols_to_drop, axis = 1, inplace = True)

nodule_bbox_data =  bbox_data.loc[bbox_data['Finding Label'] == 'Nodule']



# Calculating the centre of the bounding box using the x, y , w, h coordinates provided in the dataset.

nodule_bbox_data = nodule_bbox_data.assign(cols=pd.Series(nodule_bbox_data['Bbox [x'] + nodule_bbox_data['w']/2))

nodule_bbox_data = nodule_bbox_data.assign(rows=pd.Series(nodule_bbox_data['y'] + nodule_bbox_data['h]']/2))
# Plotting all nodules labelled with a bounding box. Overlaid on an example radiograph selected at random

path = glob.glob('../input/images_*/images/'+str(healthy.index[6]))[0]

img = io.imread(path, as_gray = True)

fig, ax = plt.subplots(figsize = (10,10))

ax.imshow(img, cmap = 'bone')

nodule_bbox_data.plot(x = ['cols'], y = ['rows'], kind = 'scatter', marker = 'o', color = 'r', linestyle = 'None', ax = ax)

ax.axis('off')

fig.savefig('nodule_locations.png', bbox_inches='tight')

plt.close()
# Plotting radiographs that contain nodules in the lowest portion of the lungs



nodules_in_diaphragm = nodule_bbox_data.loc[nodule_bbox_data['rows'] > 700]

for n in range(0,9):

    path = glob.glob('../input/images_*/images/'+str(nodules_in_diaphragm.index[n]))[0]

    img = io.imread(path, as_gray = True)

    fig, ax = plt.subplots(figsize = (10,10))

    ax.imshow(img, cmap = 'bone')

    ax.plot(nodules_in_diaphragm.loc[nodules_in_diaphragm.index[n]]['cols'],

            nodules_in_diaphragm.loc[nodules_in_diaphragm.index[n]]['rows'],

           marker = 'o', color = 'r', linestyle = 'None')

    ax.axis('off')

    fig.savefig('nodules_'+str(nodules_in_diaphragm.index[n]))

    plt.close()