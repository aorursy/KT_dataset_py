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
import matplotlib.pyplot as plt

import seaborn as sns
housing = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
housing.head()
housing.shape

housing.isnull().any()
housing.info()
plt.figure(figsize=(20,10))



sns.heatmap(data = housing.corr(), annot =True)
from sklearn.model_selection import train_test_split
y = housing.pop('price')

x = housing

print(y.head())

print(x.head())
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.70)
import statsmodels.api as sm
import sklearn as sks
sks.add_constant