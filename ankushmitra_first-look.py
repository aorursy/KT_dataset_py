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
train = pd.read_csv("../input/train.csv")
train.head()
train.shape

for column in train.columns :
    print(column,"\t",np.sum(train[column] != 0),"\t",np.std(train[column]))

# Make copy of train dataframe
train_sparse = train.copy()
# drop all columns with zero STD
for column in train_sparse.columns :
    if np.std(train_sparse[column]) == 0.0 :
        train_sparse = train_sparse.drop(column, axis=1)
    
print(train.shape)
print(train_sparse.shape)
train_sparse.to_csv("../input/train_sparse.csv")
