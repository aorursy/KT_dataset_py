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

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head()
len(df_train['place_id'].unique())
df_train['place_id'].value_counts()
df_train.groupby('place_id').first()
df_train.groupby('place_id').mean()
time = df_train['time']
time.max() - time.min()
df_train['hour'] = (df_train['time']/60)%24
df_train['dayofweek'] = np.floor((df_train['time']/(60*24))%7)+1
df_train['monthofyear'] = np.floor((df_train['time']/(60*24*30))%12)+1
df_train['season'] = np.floor((df_train['time']/(60*24*30*4))%4)+1

df_test['hour'] = (df_test['time']/60)%24
df_test['dayofweek'] = np.floor((df_test['time']/(60*24))%7)+1
df_test['monthofyear'] = np.floor((df_test['time']/(60*24*30))%12)+1
df_placecounts = df_train["place_id"].value_counts()
df_placecounts.head()
df_topplaces = df_placecounts.iloc[0:20]
l_topplaces = list(df_topplaces.index)
import matplotlib.pyplot as plt
plt.figure(figsize=(14,10))
for i in range(len(l_topplaces)):
    place = l_topplaces[i]

    df_place = df_train[df_train["place_id"]==place]

    counts, bins = np.histogram(df_place["time"], bins=50, range=[df_train["time"].min(), df_train["time"].max()])
    binsc = bins[:-1] + np.diff(bins)/2.
    
    plt.subplot(5,4,i+1)
    plt.bar(binsc, counts/(counts.sum()*1.0), width=np.diff(bins)[0])
    plt.xlim(df_train["time"].min(), df_train["time"].max())
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.gca().get_xaxis().set_ticks([])
    plt.title("pid: " + str(place))
    
plt.tight_layout()
plt.show()
plt.figure(figsize=(14,10))
for i in range(len(l_topplaces)):
    place = l_topplaces[i]

    df_place = df_train[df_train["place_id"]==place]
    df_plot = df_place['hour']
  
    counts, bins = np.histogram(df_plot, bins=50)
    binsc = bins[:-1] + np.diff(bins)/2.
    
    plt.subplot(5,4,i+1)
    plt.bar(binsc, counts/(counts.sum()*1.0), width=np.diff(bins)[0])
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.gca().get_xaxis().set_ticks([])
    plt.title("pid: " + str(place))
    
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
plt.figure(figsize=(14,10))
for i in range(len(l_topplaces)):
    place = l_topplaces[i]

    df_place = df_train[df_train["place_id"]==place]
    df_plot = df_place['dayofweek']
  
    counts, bins = np.histogram(df_plot, bins=50)
    binsc = bins[:-1] + np.diff(bins)/2.
    
    plt.subplot(5,4,i+1)
    plt.bar(binsc, counts/(counts.sum()*1.0), width=np.diff(bins)[0])
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.gca().get_xaxis().set_ticks([])
    plt.title("pid: " + str(place))
    
plt.tight_layout()
plt.show()
plt.figure(figsize=(14,10))
for i in range(len(l_topplaces)):
    place = l_topplaces[i]

    df_place = df_train[df_train["place_id"]==place]
    df_plot = df_place['monthofyear']
  
    counts, bins = np.histogram(df_plot, bins=50)
    binsc = bins[:-1] + np.diff(bins)/2.
    
    plt.subplot(5,4,i+1)
    plt.bar(binsc, counts/(counts.sum()*1.0), width=np.diff(bins)[0])
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.gca().get_xaxis().set_ticks([])
    plt.title("pid: " + str(place))
    
plt.tight_layout()
plt.show()
plt.figure(figsize=(14,10))
for i in range(len(l_topplaces)):
    place = l_topplaces[i]

    df_place = df_train[df_train["place_id"]==place]
    df_plot = df_place['season']
  
    counts, bins = np.histogram(df_plot, bins=50)
    binsc = bins[:-1] + np.diff(bins)/2.
    
    plt.subplot(5,4,i+1)
    plt.bar(binsc, counts/(counts.sum()*1.0), width=np.diff(bins)[0])
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.gca().get_xaxis().set_ticks([])
    plt.title("pid: " + str(place))
    
plt.tight_layout()
plt.show()
df_small = df_train[(df_train['x'] > 5) & (df_train['x'] < 5.25) & (df_train['y'] > 1) & (df_train['y'] < 1.25)]
df_smallByPlaceId = df_small['place_id'].value_counts()
df_topplaces = df_smallByPlaceId.iloc[0:20]
l_topplaces = list(df_topplaces.index)
df_smallByPlaceId
df_small['place_id']
plt.figure(figsize=(10,10))
for i in range(len(l_topplaces)):
    place = l_topplaces[i]

    df_place = df_small[df_small["place_id"]==place]
    plt.scatter(df_place["x"], df_place["y"], s=10, c=plt.cm.viridis(int(i*(255/len(l_topplaces)))), lw=0, cmap=plt.cm.viridis)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
for i in range(len(l_topplaces)):
    place = l_topplaces[i]

    df_place = df_small[df_small["place_id"]==place]
    ax.scatter(df_place['x'], df_place['y'], df_place['hour'], c=plt.cm.viridis(int(i*(255/len(l_topplaces)))),lw=0, cmap=plt.cm.viridis)

