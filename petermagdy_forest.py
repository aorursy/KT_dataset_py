# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
print ("Setup Complete")
# Path of the file to read
train_filepath = "/kaggle/input/Forest/train.csv"
test_filepath = "/kaggle/input/Forest/test.csv"

# Read the file
train_data = pd.read_csv(train_filepath,index_col=0)
test_data = pd.read_csv(test_filepath,index_col=0)
print ("train data size:",train_data.shape)
print ("test data size:",test_data.shape)
train_data.info()
train_data.head()
for col in train_data.columns:
    print(train_data[col].describe(), "\n")