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

# Basic Visualization

import seaborn as sn

import matplotlib.pyplot as plt 

%matplotlib inline 

sn.set_style(style="whitegrid")

import cufflinks as cf

cf.go_offline()

from sklearn import svm

from mlxtend.plotting import plot_decision_regions
df= pd.read_csv("/kaggle/input/iris/Iris.csv")
df.head(4)
df.shape
df.groupby('Species').size()
sn.countplot("Species",data=df)
label = 'Species'



f,axes = plt.subplots(2,2, figsize = (10,10) , dpi=100)

sn.violinplot(x = label , y = 'SepalLengthCm', data = df , ax= axes[0,0])

sn.violinplot(x = label , y = 'SepalWidthCm', data = df , ax= axes[0,1])

sn.violinplot(x = label , y = 'PetalLengthCm', data = df , ax= axes[1,0])

sn.violinplot(x = label  , y = 'PetalWidthCm', data = df , ax= axes[1,1])

plt.show()
# Creating a pairplot to visualize the similarities and especially difference between the species

sn.pairplot(data=df, hue='Species', palette='Set1')
Df = df.drop("Id",axis=1)

Df.iloc[:,:4].iplot(kind= 'box' , boxpoints = 'outliers')

plt.show()
df.columns
# Separating the independent variables from dependent variables

X=df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

y=df.Species
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.svm import SVC

model=SVC()
model.fit(X_train, y_train)
pred=model.predict(X_test)
# Importing the classification report and confusion matrix

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))
from sklearn.svm import SVC

model=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))
from sklearn.semi_supervised import LabelSpreading

from sklearn import datasets

from sklearn.svm import SVC

iris = datasets.load_iris()



X = iris.data[:, :2]

y = iris.target



rng = np.random.RandomState(0)

# step size in the mesh

h = .02



y_30 = np.copy(y)

y_30[rng.rand(len(y)) < 0.3] = -1

y_50 = np.copy(y)

y_50[rng.rand(len(y)) < 0.5] = -1

# we create an instance of SVM and fit out data. We do not scale our

# data since we want to plot the support vectors

ls30 = (LabelSpreading().fit(X, y_30), y_30)

ls50 = (LabelSpreading().fit(X, y_50), y_50)

ls100 = (LabelSpreading().fit(X, y), y)

rbf_svc = (svm.SVC(kernel='rbf', gamma=.5).fit(X, y), y)



# create a mesh to plot in

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                     np.arange(y_min, y_max, h))



# title for the plots

titles = ['Label Spreading 30% data',

          'Label Spreading 50% data',

          'Label Spreading 100% data',

          'SVC with rbf kernel']



color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}



for i, (clf, y_train) in enumerate((ls30, ls50, ls100, rbf_svc)):

    # Plot the decision boundary. For that, we will assign a color to each

    # point in the mesh [x_min, x_max]x[y_min, y_max].

    plt.subplot(2, 2, i + 1)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])



    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.axis('off')



    # Plot also the training points

    colors = [color_map[y] for y in y_train]

    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')



    plt.title(titles[i])



plt.suptitle("Unlabeled points are colored white", y=0.1)

plt.show()
svc = svm.SVC(kernel='rbf', C=1,gamma="auto").fit(X, y)
# create a mesh to plot in

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

h = (x_max / x_min)/100

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

 np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
svc = svm.SVC(kernel='rbf', C=10,gamma="auto").fit(X, y)
# create a mesh to plot in

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

h = (x_max / x_min)/100

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

 np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#Changing gamma to Scale

svc = svm.SVC(kernel='rbf', C=10,gamma="scale").fit(X, y)
# create a mesh to plot in

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

h = (x_max / x_min)/100

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

 np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
C = 100  # SVM regularization parameter

svc = svm.SVC(kernel='linear', C=C).fit(X, y)

rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)

poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)

lin_svc = svm.LinearSVC(C=C).fit(X, y)
# create a mesh to plot in

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                     np.arange(y_min, y_max, h))
# title for the plots

titles = ['SVC with linear kernel',

          'SVC with RBF kernel',

          'SVC with polynomial (degree 3) kernel',

          'LinearSVC (linear kernel)']
plt.set_cmap(plt.cm.Paired)



for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):

    # Plot the decision boundary. For that, we will asign a color to each

    # point in the mesh [x_min, m_max]x[y_min, y_max].

    plt.subplot(2, 2, i + 1)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])



    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    plt.set_cmap(plt.cm.Paired)

    plt.contourf(xx, yy, Z)

    plt.axis('off')



    # Plot also the training points

    plt.scatter(X[:, 0], X[:, 1], c=y)



    plt.title(titles[i])



plt.show()
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
# import some data to play with

iris = datasets.load_iris()

X = iris.data[:, :]

y = iris.target

print ("Number of data points ::", X.shape[0])

print("Number of features ::", X.shape[1])
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
fig = plt.figure(1, figsize=(14, 12))

ax = Axes3D(fig, elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(X_scaled)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,

           cmap=plt.cm.Set1, edgecolor='b', s=40)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])



plt.show()

print("The number of features in the new subspace is " ,X_reduced.shape[1])
X, y = datasets.load_iris(return_X_y=True)

X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

clf.score(X_test, y_test)

pred=clf.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))
from sklearn import svm, datasets

from sklearn.metrics import auc

from sklearn.metrics import plot_roc_curve

from sklearn.model_selection import StratifiedKFold
# Data IO and generation



# Import some data to play with

iris = datasets.load_iris()

X = iris.data

y = iris.target

X, y = X[y != 2], y[y != 2]

n_samples, n_features = X.shape



# Add noisy features

random_state = np.random.RandomState(0)

X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
# Classification and ROC analysis



# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=6)

classifier = svm.SVC(kernel='linear', probability=True,

                     random_state=random_state)
tprs = []

aucs = []

mean_fpr = np.linspace(0, 1, 100)



fig, ax = plt.subplots()

for i, (train, test) in enumerate(cv.split(X, y)):

    classifier.fit(X[train], y[train])

    viz = plot_roc_curve(classifier, X[test], y[test],

                         name='ROC fold {}'.format(i),

                         alpha=0.3, lw=1, ax=ax)

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)

    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)

    aucs.append(viz.roc_auc)



ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',

        label='Chance', alpha=.8)



mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

std_auc = np.std(aucs)

ax.plot(mean_fpr, mean_tpr, color='b',

        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

        lw=2, alpha=.8)



std_tpr = np.std(tprs, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,

                label=r'$\pm$ 1 std. dev.')



ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],

       title="Receiver operating characteristic example")

ax.legend(loc="lower right")

plt.show()