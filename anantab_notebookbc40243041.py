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
import numpy as np

import pandas as pd

import sklearn.linear_model as lm

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv');

test = pd.read_csv('../input/test.csv');

print ("Dimension of train data {}".format(train.shape))

print ("Dimension of test data {}".format(test.shape))
train.describe()
train.tail()
test.head()