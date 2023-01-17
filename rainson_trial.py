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
%matplotlib inline

## modules be used

import matplotlib.pylab as plt

import sklearn
### load data for train and test

data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

print(data_train.columns, data_train.shape)

print(data_test.columns)
data_train
## processing data

pid = data_train["PassengerId"]

Y = data_train["Survived"]
## visualize data

plt.figure()

plt.plot(pid, Y)

plt.show()
list(pid)