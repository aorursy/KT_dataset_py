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
train_data = pd.read_csv("/kaggle/input/train.csv")

test_data = pd.read_csv("/kaggle/input/test.csv")
train_data.info()
pd.set_option('display.max_columns', None) # Ensures all columns in dataframe is shown

train_data.describe(include=[np.object]) # If Parameter include is not provided descriptive statistics of only numerical attributes are provided
train_data.describe(include=[np.number])
import matplotlib.pyplot as plt

train_data.hist(figsize=(20,20),bins=10) # The histogram function plots histogram only for numerical attributes

plt.show()
# Identifying relationship between x and y variable. As an enhancement would focus on creating a function which plots all numerical x variables against y

plt.scatter(train_data['TotRmsAbvGrd'],train_data['SalePrice'])
# Creating a validation set

def split_train_val(data,test_ratio,seed=42):

    np.random.seed(seed)

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data)*test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices],data.iloc[test_indices]

            
train_set,val_set = split_train_val(train_data,0.25)

print(len(train_set),"train+",len(val_set),"val")
import hashlib

def val_set_check(identifier,val_ratio,hash):

    return hash(np.int64(identifier)).digest()[-1]< 256*val_ratio 

# The hash value returned by a particular identifier remains constant.

# Hence if new data is appended and old data is not removed then the val data remains constant for older part of data

def split_train_val_by_id(data,val_ratio,id_column,hash=hashlib.md5):

    ids= train_data['Id']

    in_val_set = ids.apply(lambda id_:val_set_check(id_,val_ratio,hash))

    return data.loc[~in_val_set],data.loc[in_val_set]
trainset,valset=split_train_val_by_id(train_data,0.2,'Id')
# To show how stratified splitting works for BldgType

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42) # n_split 

for train_index,val_index in split.split(train_data,train_data['BldgType']):

    strat_train_set = train_data.loc[train_index]

    strat_test_set = train_data.loc[val_index]  
strat_train_set['BldgType'].value_counts()/len(strat_train_set)
train_set['BldgType'].value_counts()/len(train_set)