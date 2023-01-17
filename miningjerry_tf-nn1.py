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
import gzip, pickle, sys

f = gzip.open('../input/mnist.pkl.gz', 'rb')

(train_X, train_y), (valid_X, valid_y), _ = pickle.load(f, encoding="bytes")
import tensorflow as tf

import numpy as np

import tensorflow as tf

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.preprocessing import OneHotEncoder



import gzip, pickle, sys

f = gzip.open('../input/mnist.pkl.gz', 'rb')

(train_X, train_y), (test_X, test_y), _ = pickle.load(f, encoding="bytes")



oneHot = OneHotEncoder(10, sparse=False)

train_y = oneHot.fit_transform(train_y[:, np.newaxis])

test_y = oneHot.fit_transform(test_y[:, np.newaxis])

test_y