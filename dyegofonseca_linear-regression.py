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
data = pd.read_csv("../input/data.csv")

data.head()
data.info()
columns_filter = ['ID','Age', 'Overall', 'Potential', 'Special', 'Weak Foot', 'Height', 'Weight', 'Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle']
columns_filter = ['ID','Age', 'Overall', 'Potential', 'Special', 'Weak Foot', 'Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle']
data[columns_filter].sample(5)
import seaborn as sns

import matplotlib.pyplot as plt
data.corr()
fig = plt.subplots(figsize=(15,15))

sns.set(font_scale=1.5)

sns.heatmap(data.corr(), square=True, cbar=True, annot=True, annot_kws={'size':10})
x = data[['Strength']]

y = data[['Balance']]
plt.scatter(x.head(100), y.head(100))
x = x.head(100).values

y = y.head(100).values
xy_sum = x * y

xy_sum = xy_sum.sum()

xy_sum
x_sum = x.sum()

x_sum
y_sum = y.sum()

y_sum
x_sq_sum = x ** 2

x_sq_sum = x_sq_sum.sum()

x_sq_sum
x_sum_sq = x.sum()

x_sum_sq = x_sum_sq ** 2

x_sum_sq
n = len(x)

n
b = (n * xy_sum - x_sum * y_sum) / ((n * x_sq_sum) - x_sum_sq)

b
a = ((y_sum * x_sq_sum) - (x_sum * xy_sum)) / ((n * x_sq_sum) - x_sum_sq)

a
f_x = lambda x: a + b * x
f_x(4)
plt.plot(x, f_x(x), c = 'r') # Plota linha

plt.scatter(x, y) # Plota os pontos
from sklearn.linear_model import LinearRegression
ls = LinearRegression()
ls.fit(x, y)
ls.intercept_
a
ls.coef_
b
ls.score(x, y)
from sklearn.cluster import AgglomerativeClustering
dend = AgglomerativeClustering()
np.array([x,y])
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

from sklearn.cluster import AgglomerativeClustering
dendrograma = dendrogram(linkage(data[['Strength','Balance']].head(100) , method='ward'))
hc =  AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single')
clusteres = hc.fit_predict(data[['Strength','Balance']].head(100))

clusteres
new_data = data[['Strength','Balance']].head(100).copy()
new_data['cluster'] = clusteres
plt.scatter(x = new_data.iloc[:,0], y = new_data.iloc[:,1], c=new_data.cluster)

data_teste = new_data[new_data.cluster == 0]
ls.fit(data_teste.Strength.values.reshape(-1, 1), data_teste.Balance.values.reshape(-1, 1))
ls.score(data_teste.Strength.values.reshape(-1, 1), data_teste.Balance.values.reshape(-1, 1))
data_teste.Strength.values.shape
data_teste.Strength.values.reshape(-1, 1).shape