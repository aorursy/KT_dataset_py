# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling
from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
inputFile = '../input/KK_Premium_BASE_Kaggle.csv'

data_raw = pd.read_csv(inputFile)   #, delimiter=';')
data_raw.sort_values(by=['ID'])
data_raw.head()
DROP_COLS = ['ID', 'CAT_Insurer', 'CAT_Canton', 'CAT_Region_Num', 
             'CAT_AgeCategory', 'CAT_InsuranceType', 'CAT_InsuranceTypeDetail']

data = data_raw.drop(DROP_COLS, 1)
pandas_profiling.ProfileReport(data, correlation_threshold=0.95)
# split into x and y / test and train sets
#Premium is located at the last position!!

TRAIN_SIZE = 0.8

X=data.iloc[:, 0:-1]
y=data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
y_test.head()