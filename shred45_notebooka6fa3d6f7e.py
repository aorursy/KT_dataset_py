# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

import scipy.ndimage

lidx = {"Type_1": 0, "Type_2": 1, "Type_3": 2}

dirs = ['train', 'additional']



labels = []

images = []



for l, i in lidx.items():

    for d in dirs:

        print(l, d)

        fnames = os.listdir("../input/%s/%s"%(d,l))

        for f in fnames:

            im = scipy.ndimage.imread("../input/%s/%s/%s"%(d,l,f))

            images.append(im)

            labels.append(i)



y_train = np.array(y)