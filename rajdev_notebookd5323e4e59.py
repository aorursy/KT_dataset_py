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



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit



#load the data into train and test

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')



#examine the data 

train.head()

test.head()
#prepocess the data