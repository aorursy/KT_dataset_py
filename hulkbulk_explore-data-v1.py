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
# Load 1000th rows of train.csv
train = pd.read_csv('../input/train.csv',nrows=1000)
# Look at the head of this sample
train.head()
#Get columns from train
columns = train.columns
# Analyze features size (if categorical or not)
for col in columns:
    print(col)
    print(len(set(train[col])))
    print(79*'*')
# 
#from pandas.tools.plotting import scatter_matrix
#scatter_matrix(train, alpha=0.2, figsize=(20, 20), diagonal='kde')
# Correlation each feature with hotel_cluster:
for col in columns:
    if (train[col].dtype)!='object':
        print(col)
        print(train[col].corr(train['hotel_cluster']))
# correlation between features:
col_used = []
for col in columns:
    if col not in col_used:
        col_used.append(col_used)
        for col2 in columns:
            if (train[col].dtype)!='object' and train[col2].dtype!='object' and col2 !=col:
                #col_used.append(col2)
                if np.abs(train[col].corr(train[col2]))>0.5:
                    print('correlation :\t'+col+' \tand '+col2+' : \t'+str(train[col].corr(train[col2])))