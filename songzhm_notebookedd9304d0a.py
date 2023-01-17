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
%reset -f



import scipy.io

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import numpy

import pandas as pd

from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve



import numpy

import pandas

from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from itertools import combinations



from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.datasets import dump_svmlight_file



import os

import numpy as np, h5py 

from scipy.io import loadmat

#--------------------DEFINE DATA IEEG SETS-----------------------#

DATA_FOLDER= '../input/train_1/'



#SINGLE MAT FILE FOR EXPLORATION

TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE=DATA_FOLDER + "/1_1_0.mat"

#--------------------DEFINE DATA SETS-----------------------#
def ieegMatToPandasDF(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    print(ndata)

    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])   



x=ieegMatToPandasDF(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)
%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (20.0, 20.0)

n=16

for i in range(0, n):

#     print i

    plt.subplot(n, 1, i + 1)

    plt.plot(x[i +1])