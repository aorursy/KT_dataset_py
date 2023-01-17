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
import numpy as np

import pandas as pd

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = {'Lot ID' : ["pp1","pp2","pp3","pp4","pp5","pp6","pp7","pp8","pp9","pp10","pp11","pp12","pp13","pp14","pp15","pp16","pp17","pp18","pp19","pp20","pp21","pp21","pp22","pp23","pp24"],

        'Model' : ["pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp","pp"],

        'X1' : [2, 20, 28, 250, 230,250, 270, 280, 250, 230,250, 270, 280, 25, 230,250, 270, 28, 250, 230,250, 270, 280, 250, 230],

        'X2': [2, 21, 21, 206, 215,250, 270, 280, 250, 230,250, 270, 280, 25, 230,250, 270, 28, 250, 230,250, 270, 280, 250, 230],

        'X3': [25, 20, 24, 240, 220,250, 270, 280, 250, 230,250, 270, 280, 25, 230,250, 270, 28, 250, 230,250, 270, 280, 250, 230],

        'X4': [23, 20, 25, 220, 230,250, 270, 280, 250, 230,250, 270, 280, 25, 230,250, 270, 28, 250, 230,250, 270, 280, 250, 230],

        'X5': [25, 20, 20, 270, 240,250, 270, 280, 250, 230,250, 270, 280, 25, 230,250, 270, 28, 250, 230,250, 270, 280, 250, 230]   

       }

df = pd.DataFrame(data)



LotID=df.iloc[:,0]

Model=df.iloc[:,1]

X1=df.iloc[:,2]

X2=df.iloc[:,3]

X3=df.iloc[:,4]

X4=df.iloc[:,5]

X5=df.iloc[:,6]

P1=pd.concat([X1,X2,X3,X4,X5])



X=pd.concat([P1,P1],axis=1)

X.columns=['x1','x2']

X


# visualize data point

sns.lmplot('x1', 'x2', data=X, fit_reg=False, scatter_kws={"s": 200}) # x-axis, y-axis, data, no line, marker size



# title

plt.title('kmean plot')



# x-axis label

plt.xlabel('x1')



# y-axis label

plt.ylabel('x2')
data_points = X.values
#kmeans = KMeans(n_clusters=2).fit(data_points)

kmeans=KMeans(n_clusters=2, init="random", n_init=1, max_iter=4, random_state=6).fit(X)
kmeans.labels_
kmeans.cluster_centers_
X['C1'] = kmeans.labels_
X.head(12)
sns.lmplot('x1', 'x2', data=X, fit_reg=False,  # x-axis, y-axis, data, no line

           scatter_kws={"s": 150}, # marker size

           hue="C1") # color



# title

plt.title('Kmean clustering')
X
from sklearn.cluster import KMeans





def plot_KMeans(n):

    model = KMeans(n_clusters=2, init="random", n_init=1, max_iter=n, random_state=6).fit(X)

    c0, c1 = model.cluster_centers_

    plt.scatter(X[model.labels_ == 0, 0], X[model.labels_ == 0, 1], marker='v', facecolor='r', edgecolors='k')

    plt.scatter(X[model.labels_ == 1, 0], X[model.labels_ == 1, 1], marker='^', facecolor='y', edgecolors='k')

    plt.scatter(c0[0], c0[1], marker='v', c="r", s=200)

    plt.scatter(c1[0], c1[1], marker='^', c="y", s=200)

    plt.grid(False)

    plt.title("Re={}, k={:5.2f}".format(n, -model.score(X)))



plt.figure(figsize=(8, 8))

plt.subplot(321)

plot_KMeans(1)

plt.subplot(322)

plot_KMeans(2)

plt.subplot(323)

plot_KMeans(3)

plt.subplot(324)

plot_KMeans(4)

plt.tight_layout()

plt.show()

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

from matplotlib import style

import numpy as np

from sklearn.datasets import make_moons

from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import DBSCAN



C, s= make_moons(n_samples=200, noise=0.05, random_state=0)

plt.scatter(C[:,0],C[:,1])

plt.show



C
X = data_points

y = kmeans.labels_
def plotR(X, y, title='Result'):

    plt.scatter(X[y==0, 0],X[y==0, 1],c='lightblue', marker='o', s=40, label='OK')

    plt.scatter(X[y==1, 0],X[y==1, 1],c='red', marker='o', s=40, label='NG')

    

    plt.title(title)

    plt.legend()

    plt.show()



km= KMeans(n_clusters=2, random_state=0)

y_km=km.fit_predict(X)



plotR(X,y_km, title='KMeans result')



db=DBSCAN(eps=50, min_samples=5, metric='euclidean')

y_db=db.fit_predict(X)



plotR(X, y_db, title='DBscan Result')

db=DBSCAN(eps=0.5, min_samples=5, metric='euclidean')

y_db=db.fit_predict(X)



plotR(X, y_db, title='DBscan Result')