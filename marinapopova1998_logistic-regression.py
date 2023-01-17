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
from sklearn.datasets import load_iris

import seaborn as sns 

%matplotlib inline

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.preprocessing import StandardScaler
iris = load_iris()

data = iris["data"]

fnames = iris["feature_names"]

target  = iris["target"]
df = pd.DataFrame(data,columns=fnames)

df["target"] = target
df.head()
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

tsne = TSNE(n_components=2)

scaler = StandardScaler()

tsne_data = tsne.fit_transform(scaler.fit_transform(data))

sns.scatterplot(tsne_data[iris.target ==0, 0],tsne_data[iris.target ==0, 1],color="b");

sns.scatterplot(tsne_data[iris.target ==1, 0],tsne_data[iris.target ==1, 1],color="r");

sns.scatterplot(tsne_data[iris.target ==2, 0],tsne_data[iris.target ==2, 1],color="g")
lr = LinearRegression()
lr.fit(tsne_data[iris.target!=2], iris.target[iris.target!=2])

sns.scatterplot(tsne_data[iris.target ==0, 0],tsne_data[iris.target ==0, 1],color="b");

sns.scatterplot(tsne_data[iris.target ==1, 0],tsne_data[iris.target ==1, 1],color="r");

plt.plot(np.arange(-10,10), -np.arange(-10,10)*lr.coef_[0]/lr.coef_[1], color='g')

plt.grid()
lr1 = LogisticRegression()
lr1.fit(tsne_data[iris.target!=2], iris.target[iris.target!=2])
print(lr1.coef_[0,1])

print(lr1.intercept_)

(-np.arange(-10,10)*lr1.coef_[0,0] - lr1.intercept_)/lr1.coef_[0,1]
sns.scatterplot(tsne_data[iris.target ==0, 0],tsne_data[iris.target ==0, 1],color="b");

sns.scatterplot(tsne_data[iris.target ==1, 0],tsne_data[iris.target ==1, 1],color="r");

sns.scatterplot(tsne_data[iris.target ==2, 0],tsne_data[iris.target ==2, 1],color="y")

plt.plot(np.arange(-10,10), (-np.arange(-10,10)*lr1.coef_[0,0] - lr1.intercept_)/lr1.coef_[0,1], color='g')
lr2 = LogisticRegression()

lr2.fit(tsne_data, iris.target)
lr2.predict(tsne_data)
x_scaled = scaler.fit_transform(data)
lr3 = LogisticRegression()

lr3.fit(x_scaled, iris.target)
lr3.score(x_scaled,iris.target)