# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Tue May 22 00:01:30 2018

@author: ist

"""



import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd 

from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier()



dataset = pd.read_csv("../input/train.csv")





x_train = dataset.iloc[0:21000,1:]

labels  =dataset.iloc[0:21000,0]

print(labels)