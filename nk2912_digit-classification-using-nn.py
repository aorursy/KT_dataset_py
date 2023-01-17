# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir('../'))
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.)
from mlxtend.data import loadlocal_mnist
x,y = loadlocal_mnist(images_path='../input/train-images.idx3-ubyte',labels_path='../input/train-labels.idx1-ubyte')
x.shape,y.shape
weights=np.random.randn(784,10)
def stable_softmax(out):
    exps = np.exp(out - np.max(out))
    return exps / np.sum(exps)
def forward(x):
    out = np.dot(x,weights)
    prediction = stable_softmax(out)
    return prediction
    
forward(x)
