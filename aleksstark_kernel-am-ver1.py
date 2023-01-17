# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Import libraries

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

from torch.utils import data

import torch

def loadMNIST( prefix ):

    intType = np.dtype( 'int32' ).newbyteorder( '>' )

    nMetaDataBytes = 4 * intType.itemsize



    data = np.fromfile( '/kaggle/input/the-ai-core-mnist/t10k-images.idx3-ubyte', dtype = 'ubyte' )

    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )

    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

    data = torch.stack([torch.Tensor(i) for i in data])



    labels = np.fromfile( '/kaggle/input/the-ai-core-mnist/t10k-labels.idx1-ubyte',

                      dtype = 'ubyte' )[2 * intType.itemsize:]

    labels= labels.reshape(labels.shape[0],1)

    labels = torch.stack([torch.Tensor(i) for i in labels])



    return data, labels



trainingImages, trainingLabels = loadMNIST( "train")

testImages, testLabels = loadMNIST( "t10k")



train_data = data.TensorDataset(trainingImages,trainingLabels) 

test_data = data.TensorDataset(testImages,testLabels)



plt.imshow(train_data[0][0], cmap='gray')

plt.show()
train_data.show()