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
import seaborn
data = pd.read_csv('../input/train.csv')

data
data[data.Age.isnull() == False]

import seaborn as sns

sns.set(style='darkgrid')







# Plot the responses for different events and regions

sns.countplot(x='Survived',

             data=data)
import seaborn as sns

import matplotlib.pylab as plt



sns.set(style="darkgrid")

sns.distplot(data[(data.Age.isnull() == False) & (data.Survived == False)].Age, bins = 10, kde=True, label='Погибшие')

sns.distplot(data[(data.Age.isnull() == False) & (data.Survived == True)].Age, bins = 10, kde=True, label='Выжившие')

plt.legend()