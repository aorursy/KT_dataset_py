# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting the data
from mpl_toolkits.mplot3d import Axes3D #plotting in 3D

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
print(train.head())
train['X_1000'] = train.x.apply(lambda x: round(x,0))
train['Y_1000'] = train.y.apply(lambda y: round(y,0))
grouped_pid = train.groupby('place_id')
X = grouped_pid['X_1000'].agg(lambda x: x.value_counts().index[0])
Y = grouped_pid['Y_1000'].agg(lambda x: x.value_counts().index[0])
coord = pd.concat([X, Y], axis=1)
coord = coord.reset_index()
coord.columns = ['place_id', 'pid_x', 'pid_y']
print(coord.head())
train_merged = train.merge(coord, how='left')
print(train_merged.head())
train_merged['dist'] = ((train_merged.x - train_merged.pid_x)**2 + (train_merged.y - train_merged.pid_y)**2).apply(lambda x: x**0.5)
plt.scatter(train_merged.accuracy, train_merged.dist)
plt.xlabel('Accuracy')
plt.ylabel('Distance')
plt.title('Distance vs Accuracy')
plt.show()
