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
train = pd.read_csv('../input/train_V2.csv')
print(train.head())
print(train.columns)
train['matchType'].unique()
def dummy_func(a):
    if(a =='duo-fpp'):
        return 'duo'
    else:
        return a
    

train['matchType1'] = train['matchType'].map(lambda x:dummy_func(x))
train[['matchType1','matchType']]
