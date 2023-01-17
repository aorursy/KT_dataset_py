# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_pos=pd.read_table("../input/Pos_DPC.tsv")

df_pos['#']=1

df_pos.head()
df_neg=pd.read_table("../input/Neg_DPC.tsv")

df_neg['#']=0

df_neg.head()
frames = [df_pos, df_neg]

result = pd.concat(frames)

result.head()
y=result['#']

X= result.drop(columns="#")
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(

         X, y, test_size=0.3, random_state=0)

from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.svm import SVC

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

model =RandomForestClassifier()

model.fit(X_train, y_train)

NB_pred = model.predict(X_test)

print("Random Forest Accuracy : ", accuracy_score(NB_pred, y_test))
model =MLPClassifier(solver='lbfgs', alpha=1e-5,

                   hidden_layer_sizes=(30, 25), random_state=1)



model.fit(X_train, y_train)

NB_pred = model.predict(X_test)

print("MLP Accuracy : ",accuracy_score(NB_pred, y_test))

from sklearn.naive_bayes import MultinomialNB

model = GaussianNB()

model.fit(X_train, y_train)

NB_pred = model.predict(X_test)

print("Multi-Nominal Naive Bayes Accuracy : ",accuracy_score(NB_pred, y_test))
from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.svm import SVC

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    MLPClassifier(solver='lbfgs', alpha=1e-5,

                   hidden_layer_sizes=(30, 25), max_iter=10000,random_state=1),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



size=classifiers.count



for i in classifiers:

    clf=i

    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    print("Accuracy of : ",i,"\n",metrics.accuracy_score(y_test, y_pred))

    print("\n\n")
X_train.shape