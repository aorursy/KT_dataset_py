import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

path = "/kaggle/input/iris-flower-dataset/IRIS.csv"
df = pd.read_csv(path).reset_index(drop=True)

df.info()
# creating a subset of feature-set

f1, f2, target = 'petal_length','petal_width', 'species'

X = df[[f1,f2]].reset_index(drop=True)

Y = df[target].reset_index(drop=True)

print("Unique classes: ",Y.unique())
# one-hot encode the target

# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder()

# ohe.fit(Y.reshape(-1, 1))

Y[Y=='Iris-setosa'] = 0

Y[Y=='Iris-versicolor'] = 1

Y[Y=='Iris-virginica'] = 2

Y=Y.astype(dtype=np.uint8)
import matplotlib.pyplot as plt

plt.figure()

plt.scatter(X[f1][Y==1], X[f2][Y==1], c='r')

plt.scatter(X[f1][Y==2], X[f2][Y==2], c='g')

plt.scatter(X[f1][Y==0], X[f2][Y==0], c='b')

plt.show()
from sklearn import svm

clf_ovo = svm.SVC(decision_function_shape='ovo')

clf_Linear = svm.LinearSVC(C=1.0, max_iter=10000)



models = [clf_ovo, clf_Linear]

models = [clf.fit(X, Y) for clf in models]
def make_meshgrid(x, y, h=.02):

    """Create a mesh of points to plot in



    Parameters

    ----------

    x: data to base x-axis meshgrid on

    y: data to base y-axis meshgrid on

    h: stepsize for meshgrid, optional



    Returns

    -------

    xx, yy : ndarray

    """

    x_min, x_max = x.min() - 1, x.max() + 1

    y_min, y_max = y.min() - 1, y.max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))

    return xx, yy





def plot_contours(ax, clf, xx, yy, **params):

    """Plot the decision boundaries for a classifier.



    Parameters

    ----------

    ax: matplotlib axes object

    clf: a classifier

    xx: meshgrid ndarray

    yy: meshgrid ndarray

    params: dictionary of params to pass to contourf, optional

    """

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    out = ax.contourf(xx, yy, Z, **params)

    return out



    X0, X1 = X[f1], X[f2]

    xx, yy = make_meshgrid(X0, X1)



    fig, sub = plt.subplots(1, 2,figsize=(20,10))





    titles = ("decision_function_shape='ovo'" , 'LinearSVC (linear kernel)')

    # models=[clf]/

    for clf, title, ax in zip(models, titles, sub.flatten()):

        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

        ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

        ax.set_xlim(xx.min(), xx.max())

        ax.set_ylim(yy.min(), yy.max())

        ax.set_xlabel('Sepal length')

        ax.set_ylabel('Sepal width')

        ax.set_xticks(())

        ax.set_yticks(())

        ax.set_title(title)

    plt.show()
from sklearn.model_selection import train_test_split

X_pool, X_test, y_pool, y_test = train_test_split(X, Y, test_size=0.6, random_state=6)

X_pool, X_test, y_pool, y_test = X_pool.reset_index(drop=True), X_test.reset_index(drop=True), y_pool.reset_index(drop=True), y_test.reset_index(drop=True)
def getdatapoint4activelearning(clf,pts):

    idxs = []

    for clf in clfs:

        decisions = (np.abs(list(clf.decision_function((X_pool.reset_index(drop=True))[min(pts):max(pts)]))))

        idx = np.argmin(np.array(decisions),axis=0)

        idxs.append(idx)

    return idxs
clf_ovo = svm.SVC(decision_function_shape='ovo')

clf_Linear = svm.LinearSVC(C=1.0, max_iter=10000)



class models():

    def __init__(self):

        self.models = [clf_ovo, clf_Linear] 



    def fit(self,x,y,idxs):

        self.models = [clf_ovo, clf_Linear] 

        models = [clf.fit(x.iloc[idxs],y.iloc[idxs]) for clf in self.models]

        return models



def plot_svm_amb(idx, models=None,ambigious=None):

    X0, X1 = X_pool[f1].iloc[idx], X_pool[f2].iloc[idx]

    xx, yy = make_meshgrid(X0, X1)



    fig, sub = plt.subplots(1, 2,figsize=(10,5))



    titles = ("decision_function_shape='ovo'" , 'LinearSVC (linear kernel)')





    

    for clf, title, ax in zip(models, titles, sub.flatten()):

        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

        ax.scatter(X0, X1, c=y_pool.iloc[idx], s=20, edgecolors='k')

        ax.set_xlim(xx.min(), xx.max())

        ax.set_ylim(yy.min(), yy.max())

        ax.set_xlabel('Sepal length')

        ax.set_ylabel('Petal width')

        ax.set_xticks(())

        ax.set_yticks(())

        ax.set_title(title)

        

    

    new_points = []

    clf1_pt = []

    clf2_pt = []

    

    if ambigious is not None:

            points = [np.squeeze(a).tolist() for a in ambigious]

            for clf_id,trio in enumerate(points):

                for pt in trio:

                    if pt not in idx:

                        new_points.append(pt)

                        if clf_id == 0 :

                            clf1_pt.append(pt)

                        else:

                            clf2_pt.append(pt)



    new_sample_data = list(random.sample(range(20, len(X_pool)), 10))

    idx.extend(new_sample_data)





        



        

    clf_pts=[clf1_pt,clf2_pt]

    for id_,ax in enumerate(sub.flatten()):

        for pt in clf_pts[id_]:

            ax.scatter(X_pool[f1][pt], X_pool[f2][pt], c='pink', marker="*", s=125)



    idx.extend(new_points)

    plt.plot()

    return list(set(idx))
import random

begining_thesh = 5#initial observation

idxs = list(random.sample(range(0, len(X_pool)), begining_thesh))

ambigious_pts = None

clfs_combo = models()

for i in range(10):

    clfs = clfs_combo.fit(X_pool,y_pool,idxs)

    unknown_idxs = [i for i in range(len(X_pool)) if i not in idxs]

    idxs = plot_svm_amb(idxs, models=clfs,ambigious=ambigious_pts)

    ambigious_pts = getdatapoint4activelearning(clfs,unknown_idxs)

    