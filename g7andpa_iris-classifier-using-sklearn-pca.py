# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.decomposition import PCA

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read the data using pandas

Iris_data = pd.read_csv("../input/Iris.csv")

Iris_data.head()



# partition the data using sklearn train_test_split

#X = Iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

X = Iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalLengthCm']]

# transform dataframe to ndarray so that to avoid slice unhashable problem.

X = np.array(X)

pca = PCA(n_components=2)

X = pca.fit_transform(X)

# X = pca.components_

# print(X.shape)

# print(X)

y = Iris_data[['Species']]

y = preprocessing.LabelEncoder().fit_transform(y)

# print(X)

# print(y)

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# print (X_train.shape)

# print(Y_train.shape)
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.naive_bayes import MultinomialNB

# from sklearn.linear_model import LogisticRegression

# from sklearn.ensemble import RandomForestClassifier

# from sklearn import tree

# from sklearn.ensemble import GradientBoostingClassifier

# from sklearn.svm import SVC

# from sklearn.grid_search import GridSearchCV

model = KNeighborsClassifier()

model.fit(X_train, Y_train)
# Out put accuracy using metrics built-in function

Yhat = model.predict(X_test)

#acc = np.mean(Yhat == Y_test)

from sklearn import metrics

acc = metrics.accuracy_score(Yhat, Y_test)

#rec = metrics.recall_score(Yhat, Y_test)

print(acc)

#print(rec)
def plot_decision_boundary(pred_func):

    

    #Set the boundary

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5

    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    h = 0.01

    

    # build meshgrid

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # print(xx)

    # print(yy)

    #predict

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    # print(Z.shape)

    

    # plot the contour

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plot_decision_boundary(lambda x: model.predict(x))

plt.title("decision boundary")