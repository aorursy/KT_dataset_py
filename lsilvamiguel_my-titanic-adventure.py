# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk

import matplotlib.pyplot as plt

import seaborn as sns





print("Numpy version: {0}, Pandas version: {1}, Scikit-Learn: {2}".format(np.__version__,pd.__version__,sk.__version__))



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read train dataset

train = pd.read_csv("../input/train.csv")



#number of examples

train.shape[0]
#Check the first exmaples



train.head()
train.describe()
# number missing values

(891 - 714)/ 891 * 100