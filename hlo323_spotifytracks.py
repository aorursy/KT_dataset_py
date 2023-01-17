# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pylab import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
features = pd.read_csv("../input/featuresdf.csv")
features.plot(kind="scatter", x="danceability", y="tempo")
x=features['danceability']
y=features['tempo']
(m, b)= polyfit(x, y, 1)
yp = polyval([m, b], x)
plot(x, yp)
# as tempo goes down, danceability goes up

# Any results you write to the current directory are saved as output.
key = features['key']
dict = {}
for x in key: 
    if x not in dict: 
        dict[x] = 1
    else:
        dict[x] += 1
        
keyList = list(dict.keys())
valueList = list(dict.values())
index = np.arange(len(keyList))
plt.bar(index, valueList)
plt.xlabel('Key', fontsize=5)
plt.ylabel('Number of Songs', fontsize=5)
plt.xticks(index, keyList, fontsize=5, rotation=30)
plt.title('Number of Songs in each Key')
plt.show()
artists = features['artists']
dict = {}
for x in artists: 
    if x not in dict: 
        dict[x] = 1
    else:
        dict[x] += 1
        
keyList = list(dict.keys())
valueList = list(dict.values())
index = np.arange(len(keyList))
plt.bar(index, valueList)
plt.xlabel('Artist', fontsize=5)
plt.ylabel('Number of Songs', fontsize=5)
plt.xticks(index, keyList, fontsize=5, rotation=30)
plt.title('Number of Songs by Artist')
plt.show()
features.plot(kind="scatter", x="energy", y="valence")
x=features['energy']
y=features['valence']
(m, b)= polyfit(x, y, 1)
yp = polyval([m, b], x)
plot(x, yp)

#higher energy, higher positivity
sns.FacetGrid(features, hue="mode", size=5) \
   .map(plt.scatter, "energy", "valence") \
   .add_legend()

#not really a difference between major and minor keys in terms of positivity and energy