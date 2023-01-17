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
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
data.head()
data.shape
features = data[lambda data: data.columns[1:-1]]
features.head()
y_label = data.iloc[:,-1]
y_label.head()
#sns.pairplot(features[:100])
#sns.pairplot(data[:100])
data.hist()
fig = plt.figure(figsize = (20,20))
ax = fig.gca()
data.hist(ax = ax)
fig = plt.figure(figsize = (20,20))
ax = fig.gca()
data.plot(kind='density', subplots=True, layout=(8,4), sharex=False, ax=ax)
#data.plot(kind='density', subplots=True, layout=(8,4), sharex=False)
