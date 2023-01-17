# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Loading the data

train_data = pd.read_csv('../input/train.csv')
r,c = train_data.shape
print ('Number of rows = {}\nNumber of columns = {}'.format(r,c))
# lets have a look how the data looks like 

train_data.sample(n=5)
# Lets check for nan values

train_data.isnull().sum()
# let's load the test data

test_data = pd.read_csv('../input/test.csv')
r,c = test_data.shape
print ('Number of rows = {}\nNumber of columns = {}'.format(r,c))
test_data.isnull().sum()
def fill_null(data):
    '''Fills up the null values of dataframe
    
       Input: Dataframe with null values
       Output: Dataframe with null filled up'''
    Age = data['Age'].tolist()
    Cabin = data['Cabin'].tolist()
    Embarked = data['Embarked'].tolist()
    
    Age = list(map(lambda x: np.median(Age) if x is None else x, Age))
    Cabin = list(map(lambda x: 'Unknown' if x is None else x, Cabin))
    Embarked = list(map(lambda x: max(Embarked) if x is None else x, Embarked))
    
    data['Age'] = Age; data['Cabin'] = Cabin; data['Embarked'] = Embarked
    
    return data
train_data['Cabin']
train_data = fill_null(train_data)
train_data['Cabin']