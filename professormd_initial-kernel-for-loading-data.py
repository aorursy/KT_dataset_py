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
df_test = pd.read_csv('/kaggle/input/wcobsp2020/housing_test_0126_145853.csv')

df_train = pd.read_csv('/kaggle/input/wcobsp2020/housing_train_0126_145848.csv')



print(f'Shape of the test dataset {df_test.shape}')

print(f'Shape of the training dataset {df_train.shape}')
testIds = df_test['Id']

trainIds = df_train['Id']

df_all = df_test.append(df_train, sort=False)



print(f'Shape of the combined dataset {df_all.shape}')