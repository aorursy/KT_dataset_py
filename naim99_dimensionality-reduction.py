import pandas as pd

import numpy as np 

from sklearn import linear_model

from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.datasets import load_boston

boston = load_boston()

df_x = pd.DataFrame(boston.data, columns=boston.feature_names)

df_y = pd.DataFrame(boston.target)
reg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2, random_state=4)

reg.fit(x_train,y_train)
reg.score(x_test,y_test)
df_x.head()
pca = PCA(n_components=10, whiten='True')

x = pca.fit(df_x).transform(df_x)
pca.explained_variance_
reg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(x,df_y,test_size=0.2, random_state=4)

reg.fit(x_train,y_train)
reg.score(x_test,y_test)
svd = TruncatedSVD(n_components = 10)

x = svd.fit(df_x).transform(df_x)

reg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(x,df_y,test_size=0.2, random_state=4)

reg.fit(x_train,y_train)
reg.score(x_test,y_test)
df_x.corr()
data=pd.read_csv('../input/mnist-in-csv/mnist_train.csv')
data.head()
df_x = data.iloc[:,1:]

df_y = data.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2, random_state=4)

rf = RandomForestClassifier(n_estimators = 50)

rf.fit(x_train,y_train)
pred = rf.predict(x_test)
s = y_test.values

count = 0

for i in range(len(pred)):

    if pred[i]==s[i]:

        count = count + 1
count/float(len(pred))
pca = PCA(n_components=25, whiten='True')

x = pca.fit(df_x).transform(df_x)

x_train, x_test, y_train, y_test = train_test_split(x,df_y,test_size=0.2, random_state=4)

rf = RandomForestClassifier(n_estimators = 50)

rf.fit(x_train,y_train)

pred = rf.predict(x_test)

s = y_test.values

count = 0

for i in range(len(pred)):

    if pred[i]==s[i]:

        count = count + 1
count/float(len(pred))
pca.explained_variance_
pca = PCA(n_components=2, whiten='True')

x = pca.fit(df_x).transform(df_x)

x_train, x_test, y_train, y_test = train_test_split(x,df_y,test_size=0.2, random_state=4)

rf = RandomForestClassifier(n_estimators = 50)

rf.fit(x_train,y_train)

pred = rf.predict(x_test)

s = y_test.values

count = 0

for i in range(len(pred)):

    if pred[i]==s[i]:

        count = count + 1
count/float(len(pred))
y = df_y.values

for i in range(5000):

    if y[i]==0:

        plt.scatter(x[i,1],x[i,0],c='r')

    else:

        plt.scatter(x[i,1],x[i,0],c='g')

plt.show()