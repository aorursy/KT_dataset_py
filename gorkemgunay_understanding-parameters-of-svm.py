# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read data from csv file

df = pd.read_csv("../input/iris/Iris.csv")

df.head()
df.info()
#classes

pd.unique(df["Species"])
from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()

label_encoder.fit(df["Species"])



# show convertion result

print(dict(zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_))))



df["Species"] = label_encoder.fit_transform(df["Species"])
# first 5 row 

df.head()
from sklearn.model_selection import train_test_split



#drop ID column

df.drop(["Id"],axis=1,inplace = True)



X = df.drop(["Species"],axis=1,inplace = False)

Y = df["Species"].values.reshape(-1,1)



#split data 

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0, test_size = 0.2)



print("X_train shape:",X_train.shape)

print("X_test shape:",X_test.shape)

print("Y_train shape:",Y_train.shape)

print("Y_test shape:",Y_test.shape)
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score





#def svm_model(X_train,X_test,Y_train,Y_test,C = 1.0,kernel = 'rbf',degree = 3,gamma = 'scale'):

svc = SVC()

accuracies = cross_val_score(estimator=svc, X = X_train, y = Y_train, cv = 3)

train_score = np.mean(accuracies)

    

svc.fit(X_train,Y_train)

test_score = svc.score(X_test,Y_test)



print("Train Score of Default Parameters:",train_score)

print("Test Score of Default Parameters:",test_score)


def visualize_svm_C(C,title):

    X_petal = df[df.Species != 2].iloc[:,2:4]

    X_petal = X_petal[X_petal != 2]

    y = df.Species[df.Species != 2]





    model = SVC(kernel='linear', C=C)

    model.fit(X_petal, y)



    ax = plt.gca()



    plt.scatter(X_petal.iloc[:,0], X_petal.iloc[:,1], c=y, s=50, cmap='autumn')

    plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1])

    



    xlim = ax.get_xlim()

    ylim = ax.get_ylim()



    xx = np.linspace(xlim[0], xlim[1], 30)

    yy = np.linspace(ylim[0], ylim[1], 30)

    YY, XX = np.meshgrid(yy, xx)

    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = model.decision_function(xy).reshape(XX.shape)



    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])



    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,linewidth=1, facecolors='none', edgecolors='k')



    plt.xlabel("Petal Length",fontsize = 18)

    plt.ylabel("Petal Width",fontsize = 18)

    plt.title(title + " Score:" + str(model.score(X_petal,y)),fontsize = 18)



plt.figure(figsize=(25,15))

for i,c in enumerate([0.01,0.1,1,10]):  

    plt.subplot(2,2,i+1)

    visualize_svm_C(C = c, title = "C = " + str(c))

plt.show()
def meshgrid(x, y, h=.01):

    x_min, x_max = x.min() - 1, x.max() + 1

    y_min, y_max = y.min() - 1, y.max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    return xx, yy
def contours(ax, clf, xx, yy, **params):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    out = ax.contourf(xx, yy, Z, **params)

    return out
X = df.iloc[:, :2] # we only take the first two features.

y = df.Species



def visualize_kernels(X,y,ax, title, C  = 1.0 ,kernel='linear',degree = 3,gamma='scale'):

    # The classification SVC model

    model = SVC(C = C, kernel=kernel,gamma = gamma,degree = degree)

    clf = model.fit(X, y)

    

    # title for the plots

    # Set-up grid for plotting.

    X0, X1 = X.iloc[:, 0], X.iloc[:, 1]

    xx, yy = meshgrid(X0, X1)

    contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")

    ax.set_ylabel("Sepal Length",fontsize = 18)

    ax.set_xlabel("Sepal Width", fontsize = 18)

    ax.set_title(title + "    Score:" + str(model.score(X,y)),fontsize = 18)

fig, ax = plt.subplots(1,4,figsize=(30,5))

for i,k in enumerate(['linear','poly','rbf','sigmoid']):

    visualize_kernels(X,y,ax[i],title = ("Kernel = " + k),kernel = k)

plt.show()
#normalize input and visualize again

X_sepal = X

X_norm = (X_sepal - np.min(X_sepal)) / (np.max(X_sepal) - np.min(X_sepal))

fig, ax = plt.subplots(figsize=(10,5))

visualize_kernels(X_norm,y,ax,title = ("Kernel = sigmoid"),kernel = "sigmoid")

plt.show()
fig, ax = plt.subplots(1,4,figsize=(30,5))

for i,d in enumerate([1,3,5,7]):

    visualize_kernels(X,y,ax[i],title = ("Degree = " + str(d)),kernel = 'poly',degree = d)

plt.show()
fig, ax = plt.subplots(1,4,figsize=(30,5))

for i,g in enumerate([0.01,1,10,500]):

    visualize_kernels(X,y,ax[i],kernel = 'rbf', gamma = g,title = ("Gamma:" + str(g)))

plt.show()
from sklearn.model_selection import GridSearchCV



def calculate_best_params(grid):

    svm  = SVC ();

    svm_cv = GridSearchCV(svm, grid, cv = 3)



    svm_cv.fit(X_train,Y_train)

    print("Best Parameters:",svm_cv.best_params_)

    print("Train Score:",svm_cv.best_score_)

    print("Test Score:",svm_cv.score(X_test,Y_test))
grid = {

    'C':[0.01,0.1,1,10],

    'kernel' : ["linear","poly","rbf","sigmoid"],

    'degree' : [1,3,5,7],

    'gamma' : [0.01,1,10,500]

}



calculate_best_params(grid)