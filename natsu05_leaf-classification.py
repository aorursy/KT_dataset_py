# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import matplotlib

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import glob

from skimage.io import imread

from skimage.transform import resize

from tensorflow.python.framework.ops import reset_default_graph



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# name all columns in train, should be 3 different columns with 64 values each

print(train.columns[2::64])
# try and extract and plot columns

X = train.as_matrix(columns=train.columns[2:])

print( "X.shape,", X.shape)

margin = X[:, :64]

shape = X[:, 64:128]

texture = X[:, 128:]

print("margin.shape,", margin.shape)

print("shape.shape,", shape.shape)

print("texture.shape,", texture.shape) 

# let us plot some of the features

plt.figure(figsize=(8,5))

for i in range(3):

    plt.subplot(3,3,1+i*3)

    plt.plot(margin[i])

    if i == 0:

        plt.title('Margin', fontsize=20)

    plt.axis('off')

    plt.subplot(3,3,2+i*3)

    plt.plot(shape[i])

    if i == 0:

        plt.title('Shape', fontsize=20)

    plt.axis('off')

    plt.subplot(3,3,3+i*3)

    plt.plot(texture[i])

    if i == 0:

        plt.title('Texture', fontsize=20)



plt.tight_layout()

plt.show()
print (margin.mean())

print (shape.mean())

print (texture.mean())