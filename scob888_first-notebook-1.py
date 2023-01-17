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


