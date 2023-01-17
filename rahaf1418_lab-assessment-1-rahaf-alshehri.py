# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

cars_data = pd.read_csv("../input/cars-data(1).csv")
cars_data.shape
cars_data.describe(include="all")
cars_data
list(cars_data.columns)
cars_data.dtypes
print(pd.isnull(cars_data).sum())
cars_data.info()
cars_data.describe()
dropna(axis=1, how='all')
sns.boxplot(x="symboling", y="price",data=cars_data)

cars_data.price.hist()
cars_data.corr()