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
test_dir = "/kaggle/input/hmif-data-science-bootcamp-2019/test-data.csv"

test_data = pd.read_csv(test_dir)

test_data.info()
train = '/kaggle/input/hmif-data-science-bootcamp-2019/train-data.csv'

train_data = pd.read_csv(train, index_col='id')

train_data.head()
train_data.select_dtypes(exclude=('object', 'int64', 'float64'))
train_data[['akses_internet', 'sumber_listrik']] = train_data[['akses_internet', 'sumber_listrik']].astype('int64')

test_data[['akses_internet', 'sumber_listrik']] = test_data[['akses_internet', 'sumber_listrik']].astype('int64')