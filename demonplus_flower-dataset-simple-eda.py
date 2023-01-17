import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import glob, os



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import scipy.io

mat = scipy.io.loadmat('../input/flower-dataset-102/imagelabels.mat')

mat.items()
mat["labels"]
import tarfile



#simple function to extract the train data

#tar_file : the path to the .tar file

#path : the path where it will be extracted

def extract(tar_file, path):

    opened_tar = tarfile.open(tar_file)

     

    if tarfile.is_tarfile(tar_file):

        opened_tar.extractall(path)

    else:

        print("The tar file you entered is not a tar file")
extract('../input/flower-dataset-102/102flowers.tgz', '../output/kaggle/working/flowers')
for dirname, _, filenames in os.walk('../output/kaggle/working/flowers'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
for i in range(20):

    file = os.path.join(dirname, filenames[i*5])

    im = plt.imread(file)

    plt.imshow(im)

    plt.show()