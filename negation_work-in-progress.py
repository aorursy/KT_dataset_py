# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



train_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



#print(train_df.columns.values) # Get a list of all features. Too many for .describe()

X_train = train_df.drop(['species', 'id'], axis=1)

y_train = train_df.pop('species')

y_train.describe()
_max = X_train.values.max()

_min = X_train.values.min()

mean = X_train.values.mean()

print('max={} min={} mean={}'.format(_max, _min, mean))

#This tells us that values of features we get for free seem to be normalized already 
import matplotlib.image as mpimg       # reading images to numpy arrays

import matplotlib.pyplot as plt        # to plot any graph

import matplotlib.patches as mpatches  # to draw a circle at the mean contour



from skimage import measure            # to find shape contour

import scipy.ndimage as ndi            # to determine shape centrality



# matplotlib setup

%matplotlib inline

from pylab import rcParams

rcParams['figure.figsize'] = (6, 6)
import os



def load_image(number):

    return mpimg.imread('../input/images/{}.jpg'.format(str(number)))



num_images = len(os.listdir('../input/images'))

print(num_images)



img = load_image(55)



# using image processing module of scipy to find the center of the leaf

cy, cx = ndi.center_of_mass(img)



plt.imshow(img, cmap='Set3')  # show me the leaf

plt.scatter(cx, cy)           # show me its center

plt.show()
import seaborn as sns

sns.set_style('whitegrid')
