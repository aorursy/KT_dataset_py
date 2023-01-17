# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
%pylab inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd

df_train = pd.read_csv("../input/train.csv")
def barplots(df):
    columns = ['x', 'y', 'accuracy', 'time']
    f, axarr = plt.subplots(1, len(columns), figsize=(18, 4))
    for i, c in enumerate(columns):
        axarr[i].hist(df[c], 100)
        axarr[i].set_title(c)
    f.show()
    
barplots(df_train)
def scatterplot(df, column, lim):
    f = pylab.figure(figsize=(6, 6))
    ax = f.add_axes([0, 0, 1, 1])
    df = df[df['x'] > lim[0][0]][df['x'] < lim[0][1]][df['y'] > lim[1][0]][df['y'] < lim[1][1]]
    ax.scatter(df['x'], df['y'], c=df[column])
    ax.set_xlim(lim[0])
    ax.set_ylim(lim[1])
    ax.set_title(column)
    f.show()
    
scatterplot(df_train, 'place_id', ((2, 3), (4, 5)))
scatterplot(df_train, 'accuracy', ((2, 3), (4, 5)))
def plot(timeline):
    f = pylab.figure(figsize=(20, 4))
    ax = f.add_axes([0, 0, 1, 1])
    ax.plot(timeline)
    f.show()

def running_mean(x, N):
    a = numpy.insert(np.array(x, dtype=pd.Series), 0, 0)
    cumsum = np.cumsum(a) 
    return (cumsum[N:] - cumsum[:-N]) / N
for place_id in df_train['place_id'].value_counts().head().axes[0]:
    barplots(df_train[df_train['place_id'] == place_id])
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df_train_small = df_train[:100000]
data_small = df_train_small['x', 'y', 'accuracy', 'time']
target_small = df_train_small['place_id']

def r2_scores_of_n_trees(data, target):
    kf = KFold(len(data.index), n_folds=5, shuffle=True, random_state=1)
    data = data.values
    n_trees = 10
    for train_index, test_index in kf:
        clf = RandomForestRegressor(n_estimators=n, random_state=1)
        clf.fit(data[train_index], target[train_index])
        predictions = clf.predict(data[test_index])