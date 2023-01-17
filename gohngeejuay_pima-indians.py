# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.simplefilter("ignore")
pima = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
pima.head()
pima.shape
pima.describe()
pima[pima.isna().any(axis=1)]
pima.isna().sum()
import seaborn as sns

import matplotlib.pyplot as plt



#separating the independent variables and dependent

x = pima.iloc[:,0:8]

y = pima.iloc[:,8]

#sns.boxplot(pima["Pregnancies"])

for col in x.columns:

    plt.figure()

    sns.boxplot(x[col])
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

logReg = LogisticRegression()

logReg.fit(x_train,y_train)
#from sklearn.metrics import confusion_matrix

y_pred = logReg.predict(x_test)

#cm = confusion_matrix(y_test,y_pred)

accLogReg = metrics.accuracy_score(y_pred,y_test)

print("Accuracy using lbfgs solver = " + str(accLogReg))

print("Coefficients of features = " + str(logReg.coef_))
logReg1 = LogisticRegression(penalty = 'l2',solver = 'liblinear',C = 1)

logReg1.fit(x_train,y_train)

y_pred1 = logReg1.predict(x_test)

accLogReg1 = metrics.accuracy_score(y_pred1,y_test)

print("Accuracy using liblinear solver = " + str(accLogReg1))

print("Coefficients of features = " + str(logReg1.coef_))
logReg2 = LogisticRegression(penalty = 'l2',solver = 'newton-cg',C = 1)

logReg2.fit(x_train,y_train)

y_pred2 = logReg2.predict(x_test)

accLogReg2 = metrics.accuracy_score(y_pred2,y_test)

print("Accuracy using newton-cg solver = " + str(accLogReg2))

print("Coefficients of features = " + str(logReg2.coef_))
logReg3 = LogisticRegression(penalty = 'l2',solver = 'sag',C = 1)

logReg3.fit(x_train,y_train)

y_pred3 = logReg3.predict(x_test)

accLogReg3 = metrics.accuracy_score(y_pred3,y_test)

print("Accuracy using sag solver = " + str(accLogReg3))

print("Coefficients of features = " + str(logReg3.coef_))
logReg4 = LogisticRegression(penalty = 'l2',solver = 'saga',C = 1)

logReg4.fit(x_train,y_train)

y_pred4 = logReg4.predict(x_test)

accLogReg4 = metrics.accuracy_score(y_pred4,y_test)

print("Accuracy using saga solver = " + str(accLogReg4))

print("Coefficients of features = " + str(logReg4.coef_))
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(x_train,y_train)
from sklearn import tree

tree.plot_tree(classifier,fontsize = 7)
y_pred = classifier.predict(x_test)

accDecTree = metrics.accuracy_score(y_pred,y_test)

print("Accuracy using criterion entropy = " + str(accDecTree))
classifier1 = DecisionTreeClassifier(random_state = 0)

classifier1.fit(x_train,y_train)

y_pred1 = classifier1.predict(x_test)

accDecTree1 = metrics.accuracy_score(y_pred1,y_test)

print("Accuracy using criterion gini  = " + str(accDecTree1))
classifier2 = DecisionTreeClassifier(max_depth = 5, random_state = 0)

classifier2.fit(x_train,y_train)

y_pred2 = classifier2.predict(x_test)

accDecTree2 = metrics.accuracy_score(y_pred2,y_test)

print("Accuracy using max_depth 5 = " + str(accDecTree2))
classifier2 = DecisionTreeClassifier(max_depth = 5, max_leaf_nodes = 10, random_state = 0)

classifier2.fit(x_train,y_train)

y_pred2 = classifier2.predict(x_test)

accDecTree2 = metrics.accuracy_score(y_pred2,y_test)

print("Accuracy using max_depth 5, max_leaf_nodes 10= " + str(accDecTree2))

#KFold

from sklearn import model_selection

from sklearn.model_selection import KFold

kfold = model_selection.KFold(n_splits = 10, random_state = 0)

kfold_model = LogisticRegression()

result_kfold = model_selection.cross_val_score(kfold_model,x,y,cv = kfold)

print(sum(result_kfold)/10*100)

#76.95146958304854

#Leave one out

from sklearn.model_selection import LeaveOneOut

loocv = model_selection.LeaveOneOut()

loocv_model = LogisticRegression()

result_loocv = model_selection.cross_val_score(loocv_model,x,y,cv = loocv)

print(sum(result_loocv)/768*100)

#76.82291666666666
