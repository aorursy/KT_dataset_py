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
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/predict-west-nile-virus/train.csv")

df
X = df.iloc[:,7:11]

Y = df.iloc[:,-1]
X
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.25, random_state = 0)
from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier(criterion='entropy', random_state = 0)

dtree.fit(XTrain, YTrain)

y_pred = dtree.predict(XTest)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(YTest, y_pred)

plt.clf()

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)

classNames = ['Negative','Positive']

#plt.title('Decision Tree Confusion Matrix')

plt.ylabel('True label')

plt.xlabel('Predicted label')

tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['TN','FP'], ['FN', 'TP']]

 

for i in range(2):

    for j in range(2):

        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

classifier = LogisticRegression(random_state = 0,solver='lbfgs')

classifier.fit(XTrain, YTrain)

y_pred = classifier.predict(XTest)

cm = confusion_matrix(YTest, y_pred)

plt.clf()

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)

classNames = ['Negative','Positive']

#plt.title('Logistic Regression Confusion Matrix')

plt.ylabel('True label')

plt.xlabel('Predicted label')

tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['TN','FP'], ['FN', 'TP']]

 

for i in range(2):

    for j in range(2):

        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

plt.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

#from sklearn.datasets import make_classification

clf = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=0)

clf.fit(XTrain, YTrain)

# Faire de nouvelles prédictions

y_pred = clf.predict(XTest)

cm = confusion_matrix(YTest, y_pred)

plt.clf()

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)

classNames = ['Negative','Positive']

#plt.title('Logistic Regression Confusion Matrix')

plt.ylabel('True label')

plt.xlabel('Predicted label')

tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['TN','FP'], ['FN', 'TP']]

 

for i in range(2):

    for j in range(2):

        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

plt.show()