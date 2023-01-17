!wget https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('iris.csv')
df.head()
df.columns
X = df.drop('Name',axis = 1)
y = df[['Name']]
X.head()
y.head()
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
X
## it convert it to the numpy no problem
from sklearn.decomposition import PCA
## how uch component you want

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalComponents
## converting it to pandas dataframe

pdf = pd.DataFrame(data = principalComponents,columns=['pc1','pc2'])
pdf
pdf = pd.concat([pdf,df[['Name']]],axis = 1)
pdf.head()
pca.explained_variance_ratio_
d = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
target = pdf['Name'].map(d)
pdf['target']  = np.array(target)
df = pdf.drop(['Name'],axis = 1)
df.target.unique()
from sklearn.model_selection import train_test_split
fm = list(df.columns)

t = 'target'

fm.remove(t)
print(fm)

print(t)
xt,xtst,yt,ytst = train_test_split(df[fm],df[t])
xt.head()
xtst.head()
yt.head()
ytst.head()
from sklearn.ensemble import RandomForestClassifier
r = RandomForestClassifier()
r.fit(np.array(xt),yt)
r.score(np.array(xtst),ytst)
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
from sklearn.model_selection import train_test_split
# test_size: what proportion of original data is used for test set

train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target)
train_img = StandardScaler().fit_transform(train_img)

test_img = StandardScaler().fit_transform(test_img)
pca = PCA()
pca.fit(train_img)
train_img = pca.transform(train_img)

test_img = pca.transform(test_img)
train_img
test_img
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_img, train_lbl)
logisticRegr.predict(test_img[0].reshape(1,-1))
logisticRegr.predict(test_img[0:10])
logisticRegr.score(test_img, test_lbl)
plt.imshow(train_img[2].reshape(28,28))