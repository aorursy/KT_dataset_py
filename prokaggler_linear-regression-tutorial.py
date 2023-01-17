# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.rc('figure', figsize=(8, 12))

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# loading data and listing initial 5 rows

data = pd.read_csv('../input/kc_house_data.csv')

data.head()
# visualising data types of each columns

data.dtypes
# getting the rows and columns of whole data set

data.shape
plt.matshow(data.corr())

plt.show()