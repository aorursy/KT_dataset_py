# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("../input/mall-customers/Mall_Customers.csv")

df.head()
df.shape
df.fillna(0)
df.Genre = df.Genre.map({

    "Male": 0,

    "Female": 1

})

df.head()
x = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

sse = []

n = []

for i in range (1,10):

    n.append(i)

    model = KMeans(n_clusters = i)

    model.fit_predict(x)

    sse.append(model.inertia_)

plt.plot(n,sse)

plt.show()

model = KMeans(n_clusters = 5)

y_pred = model.fit_predict(x)
df["Group"] = y_pred
df.head()
cen = model.cluster_centers_

cen
for j in df.Group.unique():

    newdf = df[df.Group == j]

    x = newdf[["Annual Income (k$)", "Spending Score (1-100)"]].values

    plt.scatter(x[:,0],x[:,1])

    

for c in cen:

    plt.scatter(c[0],c[1], marker = "*" ,s = 100, color = "black" )

plt.show()    