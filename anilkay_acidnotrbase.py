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
data=pd.read_csv("/kaggle/input/ph-recognition/ph-data.csv")

data.head()
def acidnotrbase(row):

    if row['label'] < 7:

        return 0 

    elif row['label']>7: 

        return 1 

    else :

        return 2

data['type'] = data.apply(lambda row: acidnotrbase(row), axis=1)

data.head()
%matplotlib inline

import seaborn as sns

sns.countplot(x=data["type"])
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection='3d')



xs = data["blue"]

ys = data['green']

zs = data['red']

import matplotlib.cm as cm

ax.scatter(xs, ys, zs,label=data["type"],cmap='viridis')



ax.set_xlabel('Blue')

ax.set_ylabel('Red')

ax.set_zlabel('Green')

plt.show()
sns.relplot(data=data,x="red",y="green",hue="type")
sns.relplot(data=data,x="red",y="blue",hue="type")
from sklearn.manifold import TSNE

tsne=TSNE(n_components=2)

x=data.iloc[:,0:3]

x_tsne=tsne.fit_transform(x)
xtes=pd.DataFrame(x_tsne)

xtes.columns = ['b', 'i']

xtes.head()
xtes["type"]=data["type"]
xtes.head()
sns.relplot(data=xtes,x="b",y="i",hue="type")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)

rgb=data[["blue","red","green"]]

x_lda=lda.fit_transform(rgb,data["type"])
xlda=pd.DataFrame(x_lda)

xlda.columns = ['ld', 'a']

xlda["type"]=data["type"]

xlda.head()
sns.relplot(data=xlda,x="ld",y="a",hue="type")
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

from sklearn.model_selection import train_test_split

X=xlda[["ld","a"]]

y=data["type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
knn.fit(X_train,y_train)

ypred=knn.predict(X_test)
import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))