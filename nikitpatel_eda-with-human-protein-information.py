# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from collections import Counter

import os
print(os.listdir("../input"))
#import training data
train = pd.read_csv("../input/train.csv")
print(train.head())

#map of targets in a dictionary
subcell_locs = {
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles",
5:  "Nuclear bodies",
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus",
8:  "Peroxisomes",
9:  "Endosomes",
10:  "Lysosomes",
11:  "Intermediate filaments",   
12:  "Actin filaments",
13:  "Focal adhesion sites",   
14:  "Microtubules",
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle",
18:  "Microtubule organizing center",  
19:  "Centrosome",
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions", 
23:  "Mitochondria",
24:  "Aggresome",
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}
print("The image with ID == 1 has the following labels:", train.loc[1, "Target"])
print("These labels correspond to:")
for location in train.loc[1, "Target"].split():
    print("-", subcell_locs[int(location)])

#reset seaborn style
sns.reset_orig()

#get image id
im_id = train.loc[1, "Id"]

#create custom color maps
cdict1 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

cdict2 = {'red':   ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

cdict3 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0))}

cdict4 = {'red': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

plt.register_cmap(name='greens', data=cdict1)
plt.register_cmap(name='reds', data=cdict2)
plt.register_cmap(name='blues', data=cdict3)
plt.register_cmap(name='yellows', data=cdict4)

#get each image channel as a greyscale image (second argument 0 in imread)
green = cv2.imread('../input/train/{}_green.png'.format(im_id), 0)
red = cv2.imread('../input/train/{}_red.png'.format(im_id), 0)
blue = cv2.imread('../input/train/{}_blue.png'.format(im_id), 0)
yellow = cv2.imread('../input/train/{}_yellow.png'.format(im_id), 0)

#display each channel separately
fig, ax = plt.subplots(nrows = 2, ncols=2, figsize=(15, 15))
ax[0, 0].imshow(green, cmap="greens")
ax[0, 0].set_title("Protein of interest", fontsize=18)
ax[0, 1].imshow(red, cmap="reds")
ax[0, 1].set_title("Microtubules", fontsize=18)
ax[1, 0].imshow(blue, cmap="blues")
ax[1, 0].set_title("Nucleus", fontsize=18)
ax[1, 1].imshow(yellow, cmap="yellows")
ax[1, 1].set_title("Endoplasmic reticulum", fontsize=18)
for i in range(2):
    for j in range(2):
        ax[i, j].set_xticklabels([])
        ax[i, j].set_yticklabels([])
        ax[i, j].tick_params(left=False, bottom=False)
plt.show()
labels_num = [value.split() for value in train['Target']]
labels_num_flat = list(map(int, [item for sublist in labels_num for item in sublist]))
labels = ["" for _ in range(len(labels_num_flat))]
for i in range(len(labels_num_flat)):
    labels[i] = subcell_locs[labels_num_flat[i]]

fig, ax = plt.subplots(figsize=(15, 5))
pd.Series(labels).value_counts().plot('bar', fontsize=14)