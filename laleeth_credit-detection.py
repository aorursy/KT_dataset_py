# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Train & Test data

data_train = pd.read_csv('/kaggle/input/crm/CRM_TrainData.csv')

data_test = pd.read_csv('/kaggle/input/crm/CRM_TestData.csv')
shape_of_train_dataset = data_train.shape

print(shape_of_train_dataset)

shape_of_test_dataset = data_test.shape

print(shape_of_test_dataset)

print(shape_of_train_dataset == shape_of_test_dataset)
data_train.head(5)
data_train.isnull().sum()
data_test.isnull().sum()
data_train.describe()
import matplotlib.pyplot as plt

import seaborn as sns

data_train.head(2)
data_train.dtypes