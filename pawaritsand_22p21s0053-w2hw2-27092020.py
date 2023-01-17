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
XY_train = pd.read_csv('/kaggle/input/titanic/train.csv').set_index('PassengerId')

x_test= pd.read_csv('/kaggle/input/titanic/test.csv').set_index('PassengerId')

y_test = pd.read_csv('/kaggle/input/titanic/gender_submission.csv').set_index('PassengerId')

XY_test=pd.concat([x_test, y_test], axis=1)

df=pd.concat([XY_train, XY_test])
XY_train
XY_test
df
df.count()
df.describe()
#replace lable to value

df['Sex'] = df['Sex'].replace(['male', 'female'], [0,1])



# repair data

df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Fare'] = df['Fare'].fillna(df['Fare'].mean())



# set input ouput

X = pd.get_dummies(df[['Pclass','Sex','Age','Fare']])

Y = df['Survived']
df.count()
X
Y
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn import metrics



print("_____________________DecisionTreeClassifier_____________________")

decision_tree = DecisionTreeClassifier() 

predictions_decision_tree = cross_val_predict(decision_tree, X, Y, cv=5)

print(metrics.classification_report(Y, predictions_decision_tree, digits=4))



print("_____________________Gaussian Naive Bayes_____________________")

gaussian = GaussianNB() 

predictions_gaussian = cross_val_predict(gaussian, X, Y, cv=5)

print(metrics.classification_report(Y, predictions_gaussian, digits=4))



print("_____________________Neural Network_____________________")

nn = MLPClassifier(alpha=1, max_iter=1000)

predictions_nn = cross_val_predict(nn, X, Y, cv=5)

print(metrics.classification_report(Y, predictions_nn, digits=4))