import pandas as pd

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/Iris.csv')
df.info()
sns.pairplot(df,hue='Species',palette="muted",size=5,vars=['SepalWidthCm','SepalLengthCm','PetalLengthCm','PetalWidthCm'],kind='scatter',markers=['X','o','+'])



plt.show()
fig, ax = plt.subplots(2,2, figsize=(12,10))

sns.violinplot(ax=ax[0,0],x=df['Species'], y=df['SepalLengthCm'])

sns.violinplot(ax=ax[0,1],x=df['Species'], y=df['SepalWidthCm'])

sns.violinplot(ax=ax[1,0],x=df['Species'], y=df['PetalLengthCm'])

sns.violinplot(ax=ax[1,1],x=df['Species'], y=df['PetalWidthCm'])
x = df.drop('Species', 1).values

x

y = df['Species'].values
from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split



x_normalize = normalize(x)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



print(x_train.shape)

print(x_test.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn import svm
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(x_train, y_train)



y_pred = knn.predict(x_test)

print("accucary using KNeighbour : ", accuracy_score(y_test,y_pred))
clf = svm.SVC()

clf.fit(x_train, y_train)



y_pred = clf.predict(x_test)

print("accucary using Classification : ", accuracy_score(y_test,y_pred))