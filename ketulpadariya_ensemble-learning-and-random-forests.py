# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_moons



X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_clf = LogisticRegression(solver = 'liblinear')

rnd_clf = RandomForestClassifier(n_estimators= 10)

svm_clf = SVC(gamma = 'auto',probability=True)
voting_clf = VotingClassifier(

    

    estimators = [('lr',log_clf),('rc',rnd_clf),('svc',svm_clf)],

    voting = 'soft'

    )



voting_clf.fit(X_train,y_train)
from sklearn.metrics import accuracy_score



for clf in (log_clf,rnd_clf,svm_clf,voting_clf):

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__,accuracy_score(y_test,y_pred))
from matplotlib.colors import ListedColormap



def figure():

    plt.figure(figsize = (12,8))



figure()

def plot_decision_boundary(clf=clf,X=X,y=y,alpha = 0.5):

    X1_new = np.linspace(-2,3,200).reshape(-1,1)

    X2_new = np.linspace(-2,2,200).reshape(-1,1)

    X1,X2 = np.meshgrid(X1_new,X2_new)

    X_new = np.c_[X1.ravel(),X2.ravel()]

    y_pred_new = clf.predict(X_new).reshape(X1.shape)

    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

    plt.contour(X1,X2,y_pred_new,alpha = alpha,cmap=custom_cmap)

    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)

    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)

    plt.xlabel(r"$x_1$", fontsize=18)

    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

    plt.title(clf.__class__.__name__)

plot_decision_boundary()
from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier



bag_clf = BaggingClassifier(

    DecisionTreeClassifier(), n_estimators= 500,

    max_samples= 110, bootstrap= True, n_jobs= -1    )



bag_clf.fit(X_train,y_train)

y_pred = bag_clf.predict(X_test)

accuracy_score(y_pred,y_test)



dec_clf = DecisionTreeClassifier()

dec_clf.fit(X_train,y_train)
figure()

plt.subplot(121);plot_decision_boundary(clf = bag_clf)

plt.subplot(122);plot_decision_boundary(clf = dec_clf)
from sklearn.ensemble import BaggingClassifier



bagg_clf = BaggingClassifier(DecisionTreeClassifier(),

                        n_estimators= 500, bootstrap= True, n_jobs= -1,

                        oob_score= True)

bagg_clf.fit(X_train,y_train)
bagg_clf.oob_score_
bagg_clf.oob_decision_function_
y_pred = bagg_clf.predict(X_test)

accuracy_score(y_pred,y_test)
from sklearn.ensemble import RandomForestClassifier



rnd_clf = RandomForestClassifier(

    n_estimators= 500,

    max_leaf_nodes= 16,

    n_jobs  = -1,



    )



rnd_clf.fit(X,y)
y_pred = rnd_clf.predict(X_test)

accuracy_score(y_pred,y_test)
from sklearn.datasets import load_iris



iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators= 500,n_jobs= -1)

rnd_clf.fit(iris['data'],iris['target'])

for name, score in zip(iris['feature_names'],rnd_clf.feature_importances_):

    print(name,score)
from sklearn.ensemble import AdaBoostClassifier



ada_clf = AdaBoostClassifier(

    DecisionTreeClassifier(max_depth =1),n_estimators= 200,

    algorithm= 'SAMME.R',learning_rate= 0.5

    )

ada_clf.fit(X_train,y_train)
y_pred = ada_clf.predict(X_test)
accuracy_score(y_pred,y_test)
y_pred
figure()

plot_decision_boundary(ada_clf)
from sklearn.tree import DecisionTreeRegressor

figure()

plt.subplot(131)

tree_reg1 = DecisionTreeClassifier(max_depth= 2)

tree_reg1.fit(X,y)

plot_decision_boundary(tree_reg1)





plt.subplot(132)

y2 = y - tree_reg1.predict(X)

tree_reg2 = DecisionTreeClassifier(max_depth= 2)

tree_reg2.fit(X,y2)

plot_decision_boundary(tree_reg2)



y_pred = sum(tree.predict(X_new) for tree in (tree_reg1,tree_reg2,tree_reg3))
y_pred
from sklearn.ensemble import GradientBoostingRegressor



gbrt = GradientBoostingRegressor(max_depth = 2 , n_estimators= 3, learning_rate= 1.0)

gbrt.fit(X,y)

figure()

plot_decision_boundary(gbrt)
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



X_train,X_val, y_train,y_val = train_test_split(X,y)
gbrt = GradientBoostingRegressor(max_depth = 2, n_estimators= 120)

gbrt.fit(X_train,y_train)
errors = [

    mean_squared_error(y_val,y_pred) for y_pred in gbrt.staged_predict(X_val)

]

bst_n_estimators = np.argmin(errors)
bst_n_estimators
gbrt_best = GradientBoostingRegressor(max_depth =2 , n_estimators=bst_n_estimators)

gbrt_best.fit(X_train,y_train)
figure()

plot_decision_boundary(gbrt_best)
gbrt = GradientBoostingRegressor(max_depth= 2, warm_start= True)



min_val_error = float('inf')

error_going_up =  0



for n_estimators in range(1,120):

    gbrt.n_estimators = n_estimators

    

    gbrt.fit(X_train,y_train)

    y_pred = gbrt.predict(X_val)

    mse = mean_squared_error(y_pred,y_test)

    

    if min_val_error > mse:

        min_val_error = mse

        error_going_up = 0

    else:

        error_going_up += 1

        if error_going_up == 5:

            break

            