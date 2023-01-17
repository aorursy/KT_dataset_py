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
#1. Imports Declaration
import pandas as pd

# Data Read Function | Definition
def get_train_data():
    file_train = '../input/train.csv'
    data_train = pd.read_csv(file_train)
    print(data_train.head())
    return(data_train)

# Data Read Function | Call
data_train = get_train_data()

