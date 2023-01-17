#importing of libraries



import pandas as pd

import numpy as np

from sklearn.datasets import load_iris

from numpy.random import seed

from numpy.random import randn

from matplotlib import pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/iris/Iris.csv")

data.head()
#replacing of column names

data.rename(columns={'SepalLengthCm':'SL',

                      'SepalWidthCm':'SW',

                     'PetalLengthCm':'PL',

                      'PetalWidthCm':'PW'}, 

                 inplace=True)

data = data.drop(['Id'], axis = 1)

data.head()
#checking if the dataset is balanced

data.Species.value_counts()
#checking if there are any empty values

data.isnull().sum()
# seed the random number generator

seed(1)



x_SL = [data.SL[data.Species == 'Iris-setosa'], data.SL[data.Species == 'Iris-virginica'], data.SL[data.Species == 'Iris-versicolor']]

x_SW = [data.SW[data.Species == 'Iris-setosa'], data.SW[data.Species == 'Iris-virginica'], data.SW[data.Species == 'Iris-versicolor']]

x_PL = [data.PL[data.Species == 'Iris-setosa'], data.PL[data.Species == 'Iris-virginica'], data.PL[data.Species == 'Iris-versicolor']]

x_PW = [data.PW[data.Species == 'Iris-setosa'], data.PW[data.Species == 'Iris-virginica'], data.PW[data.Species == 'Iris-versicolor']]



fig, ax = plt.subplots(2, 2, figsize=(15, 10))





ax[0,0].boxplot(x_SL)

ax[0, 0].set_title('SL')

ax[0,1].boxplot(x_SW)

ax[0, 1].set_title('SW')

ax[1,0].boxplot(x_PL)

ax[1, 0].set_title('PL')

ax[1,1].boxplot(x_PW)

ax[1, 1].set_title('PW')



plt.show()
sns.pairplot(data, hue="Species")
#Logistic regression



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
# Label encoding

from sklearn import preprocessing 

   

label_encoder = preprocessing.LabelEncoder() 

data['Species']= label_encoder.fit_transform(data['Species']) 

data['Species'].unique() 
X = data.drop(['Species'], axis=1)

y = data.Species



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
lr = LogisticRegression(solver='lbfgs', multi_class='auto',  max_iter = 300)

lr.fit(X_train, y_train)



y_pred = lr.predict(X_test)

confusion_matrix(y_test, y_pred)
score = lr.score(X_test, y_test)

print(score)
#Principal Component Analysis (PCA)



from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)



from sklearn.decomposition import PCA

pca = PCA(n_components = 2, whiten = True).fit(X)

X_pca = pca.transform(X)
#Logistic regression with PCA



X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=1)

lr = LogisticRegression(solver='lbfgs', multi_class='auto',  max_iter = 300)

lr.fit(X_train, y_train)



y_pred = lr.predict(X_test)

confusion_matrix(y_test, y_pred)
score = lr.score(X_test, y_test)

print(score)
#SVM

from sklearn import svm, datasets





def make_meshgrid(x, y, h=.04):

    x_min, x_max = x.min() - 1, x.max() + 1

    y_min, y_max = y.min() - 1, y.max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))

    return xx, yy





def plot_contours(ax, clf, xx, yy, **params):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    out = ax.contourf(xx, yy, Z, **params)

    return out



C = 1.0  # SVM regularization parameter

models = (svm.SVC(kernel='linear', C=C),

          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))

models = (clf.fit(X_train, y_train) for clf in models)

#y_pred = (clf.predict(X_test) for clf in models)

y_pred = (clf.predict(X_test) for clf in models)

# title for the plots

titles = ('SVC with linear kernel',

          'SVC with polynomial (degree 3) kernel')



# Set-up 2x2 grid for plotting.

fig, sub = plt.subplots(1, 2, figsize=(15, 10))



X0, X1 = X_train[:, 0], X_train[:, 1]

xx, yy = make_meshgrid(X0, X1)



for clf, title, ax in zip(models, titles, sub.flatten()):

    plot_contours(ax, clf, xx, yy)

    ax.scatter(X0, X1, c=y_train, s=80, edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())

    ax.set_ylim(yy.min(), yy.max())

    ax.set_xticks(())

    ax.set_yticks(())

    ax.set_title(title)



plt.show()