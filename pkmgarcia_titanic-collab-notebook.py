# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Read Data
train_data_path = "../input/train.csv"
raw_train_data = pd.read_csv(train_data_path)

# Data Summary
print("Info: \n")
raw_train_data.info()
print("\nDescription: \n{}".format(raw_train_data.describe()))
raw_train_data.describe()
print("\nHead: \n")
raw_train_data.head()
