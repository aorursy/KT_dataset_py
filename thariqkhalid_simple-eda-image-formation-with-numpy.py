# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import json

import matplotlib.pyplot as plt

from skimage import color



# Any results you write to the current directory are saved as output.
data = pd.read_json("../input/train.json")

data.head()
HH = np.asarray(np.vstack(data.band_1.values))

HV = np.asarray(np.vstack(data.band_2.values))

HB = HH/HV
def normalize(v):

    minv = np.min(v)

    maxv = np.max(v)

    res = (v-minv)/(maxv-minv)

    return res



R = normalize(HH)

G = normalize(HV)

B = normalize(HB)



print(R.shape, G.shape, B.shape)
label = np.asarray(np.vstack(data.is_iceberg.values))
ims_h = np.hstack((R,G,B))

ims = ims_h.reshape(1604,3,75,75).transpose(0,2,3,1).astype("float")

ims.shape
def displayimage(ims, id):

    plt.imshow(ims[id],cmap='inferno')

    plt.grid(False)

    plt.title(data.is_iceberg[id])

    plt.show()

    

gims = color.rgb2gray(ims)

displayimage(gims,10)