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
!pip install dabl
data = pd.read_csv("/kaggle/input/melbourne-housing-market/Melbourne_housing_FULL.csv")

data.shape
data = data.dropna(subset=['Price'])

data.shape
from dabl import plot

plot(data, target_col='Price')
from dabl import SimpleRegressor

sr = SimpleRegressor().fit(data, target_col='Price')
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt

plt.figure(figsize=(25, 5))

plot_tree(sr.est_[-1], max_depth=3);