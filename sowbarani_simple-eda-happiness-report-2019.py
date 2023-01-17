# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/world-happiness-report-2019/world-happiness-report-2019.csv')
dataset.head()
dataset = dataset.rename(columns = {'Country (region)':'Country','SD of Ladder':'SDLadder','Positive affect':'Positive','Negative affect':'Negative','Social support':'Social'})
dataset.isnull().sum()
dataset.describe()
dataset = dataset.fillna(method = 'ffill')
dataset.isnull().sum()
fig,ax = plt.subplots(figsize=(10,10))

sns.heatmap(dataset.corr(),ax=ax,annot= True,linewidth= 0.02,fmt='.2f',cmap = 'Blues')

plt.show()