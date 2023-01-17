# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

df = pd.read_csv('../input/Iris.csv')

df.head()
df.hist(figsize=(20,10))

df['Species'].value_counts().plot(kind='bar')
df = df.drop(['Id'],axis=1)

target = df['Species']

s = set()

for val in target:

    s.add(val)

s = list(s)

rows = list(range(100,150))

df = df.drop(df.index[rows])
x = df['SepalWidthCm']

y = df['PetalWidthCm']



setosa_x = x[:50]

setosa_y = y[:50]



versicolor_x = x[50:]

versicolor_y = y[50:]



plt.figure(figsize=(8,6))

plt.scatter(setosa_x,setosa_y,marker='o',color='green')

plt.scatter(versicolor_x,versicolor_y,marker='v',color='red')

plt.show()
x = df['SepalLengthCm']

y = df['PetalLengthCm']



setosa_x = x[:50]

setosa_y = y[:50]



versicolor_x = x[50:]

versicolor_y = y[50:]



plt.figure(figsize=(8,6))

plt.scatter(setosa_x,setosa_y,marker='o',color='green')

plt.scatter(versicolor_x,versicolor_y,marker='v',color='red')

plt.show()
le = preprocessing.LabelEncoder()

y = le.fit_transform(df['Species'])

y[y==0] = -1
x_train, x_test, y_train, y_test = train_test_split(df[['SepalWidthCm', 'PetalWidthCm']].values, y, test_size=0.2)
clf = SVC(kernel='linear')

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print('Accuracy ', accuracy_score(y_test,y_pred))

confusion_matrix(y_test, y_pred)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=40, cmap=plt.cm.Spectral)



# plot the decision function

ax = plt.gca()

xlim = ax.get_xlim()

ylim = ax.get_ylim()



# create grid to evaluate model

xx = np.linspace(xlim[0], xlim[1], 30)

yy = np.linspace(ylim[0], ylim[1], 30)

YY, XX = np.meshgrid(yy, xx)

xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = clf.decision_function(xy).reshape(XX.shape)



# plot decision boundary and margins

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,

           linestyles=['--', '-', '--'])

# plot support vectors

ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,

           linewidth=1, facecolors='none', edgecolors='k')

plt.show()
print('Support vectors\n', clf.support_vectors_)

print('Feature weights', clf.coef_)
clf.support_