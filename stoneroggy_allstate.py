# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# loading data
dataset = pd.read_csv('../input/allstate-claims-severity/train.csv')
dataset_test = pd.read_csv("../input/allstate-claims-severity/test.csv")
# setting pandas option to display all columns
pd.set_option('display.max_columns',None)
dataset.head(10)
# droping id column form both dataset and dataset_test
ID = dataset_test['id']
dataset_test.drop('id', inplace = True, axis=1)
dataset = dataset.iloc[:,1:]
# list of names of all categorical columns in dataset
cat_col = dataset.select_dtypes(include=['object']).columns
# one hot encoding 
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
dataset_cat = encoder.fit_transform(dataset[cat_col]).toarray()

dataset_num = np.array(dataset[dataset.select_dtypes(exclude=['object']).columns])
dataset_num
dataset = np.concatenate((dataset_cat, dataset_num), axis=1)
dataset.shape
x_train = dataset[:,0:-1]
y_train = dataset[:,-1]