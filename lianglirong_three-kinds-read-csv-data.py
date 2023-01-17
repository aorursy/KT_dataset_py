# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
filename = "../input/pima_data.csv"
#load data method 1 by standard python library
with open(file=filename,mode='rt') as raw_data:
    readers = csv.reader(raw_data)
    x = list(readers)
    data = np.array(x)
    print(data.shape)
    
#load data method 2 by numpy
with open(file=filename,mode='rt') as raw_data:
    data = np.loadtxt(raw_data,delimiter=',')
    print(data.shape)
#load data method 2 by pandas
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv(filename,names=names)
print(data.shape)
