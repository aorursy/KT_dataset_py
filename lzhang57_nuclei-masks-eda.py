# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import skimage.io

import skimage.segmentation

import matplotlib.pyplot as plt

from glob import glob

%matplotlib inline

plt.rcParams['figure.figsize']=10,10

# Any results you write to the current directory are saved as output.
# Load a single image and its associated masks

id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'

file = "../input/stage1_train/{}/images/{}.png".format(id,id)

masks = "../input/stage1_train/{}/masks/*.png".format(id)

image = skimage.io.imread(file)

masks = skimage.io.imread_collection(masks).concatenate()

height, width, _ = image.shape

num_masks = masks.shape[0]



# Make a ground truth label image (pixel value is index of object label)

labels = np.zeros((height, width), np.uint16)

for index in range(0, num_masks):

    labels[masks[index] > 0] = index + 1

    

plt.imshow(image)

plt.imshow(labels,alpha=0.5)
def getContour(mask):

    ### generate bound

    mask_pad=np.pad(mask,((1,1),(1,1)),'constant')

    h,w=mask.shape

    contour=np.zeros((h,w),dtype=np.bool)

    for i in range(3):

        for j in range(3):

            if i==j==1:

                continue

            edge=(np.float32(mask)-np.float32(mask_pad[i:i+h,j:j+w]))>0

            contour=np.logical_or(contour,edge)

    return contour



def showContour(image,contours):

    vis=np.copy(image)

    for contour in contours:

        vis[:,:,0]^=np.uint8(contour*255)

    plt.imshow(vis)

    

contours = [getContour(mask) for mask in masks]

showContour(image,contours)
from scipy.ndimage.morphology import distance_transform_edt



edts=[distance_transform_edt(mask) for mask in masks]

plt.imshow(np.sum(edts,axis=0))
## contour finding with distance transform

plt.imshow(np.sum(edts,axis=0)==1)