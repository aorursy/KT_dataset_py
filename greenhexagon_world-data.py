# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

worlddata = pd.read_csv("../input/countries of the world.csv", decimal=',')
worlddata.head()
missing_values_count = worlddata.isnull().sum()
missing_values_count
(missing_values_count.sum()/np.product(worlddata.shape))*100
dropped_data = worlddata.dropna()
heat = sns.heatmap(dropped_data.corr(), annot = True)
heat.figure.set_size_inches(18.5, 10.5)
list(dropped_data)
p1 = sns.regplot(y = 'Infant mortality (per 1000 births)', x = 'Birthrate', data = dropped_data, fit_reg = False)
p1.figure.set_size_inches(20, 10.5)


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))
label_point(dropped_data['Birthrate'],dropped_data['Infant mortality (per 1000 births)'], dropped_data['Country'],p1)  