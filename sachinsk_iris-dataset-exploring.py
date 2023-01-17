import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets

import matplotlib.pyplot as plt

plt.style.use('ggplot')
iris=datasets.load_iris()
type(iris)
iris.keys()
print(type(iris.data))

print(type(iris.target))

print(type(iris.DESCR))

print(type(iris.target_names))

print(type(iris.feature_names))

print(type(iris.filename))

print(iris.data.shape)

print(iris.target.shape)
print(iris.feature_names)

print(iris.target_names)
X= iris.data

Y= iris.target
df= pd.DataFrame(X, columns=iris.feature_names)

df.head()
_=pd.scatter_matrix(df,c=Y,figsize=[10,10],s=150,marker='D')