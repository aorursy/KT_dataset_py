# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
%matplotlib inline

# Any results you write to the current directory are saved as output.
meta_train = pd.read_csv('../input/training_set_metadata.csv')
meta_train.head()
def photoztodist(data) :
    return ((((((np.log((((data["hostgal_photoz"]) + (np.log((((data["hostgal_photoz"]) + (np.sqrt((np.log((np.maximum(((3.0)), ((((data["hostgal_photoz"]) * 2.0))))))))))))))))) + ((12.99870681762695312)))) + ((1.17613816261291504)))) * (3.0))
myfilter = ~meta_train.distmod.isnull()
plt.scatter(photoztodist(meta_train[myfilter]),meta_train[myfilter].distmod)
from sklearn.metrics import mean_squared_error
mean_squared_error(meta_train[myfilter].distmod,photoztodist(meta_train[myfilter]))
