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
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
women = train.loc[train.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train.loc[train.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier



y = train["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)
model2 = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1)

model2.fit(X, y)

predictions2 = model2.predict(X_test)



output2 = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions2})

output2.to_csv('2nd_try.csv', index=False)
##Output Finder Function

def output(model):

    model.fit(X, y)

    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

    return output
model3 = RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=1)



output3 = output(model3)

output3.to_csv('3rd_try.csv', index=False)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

model4 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')



output4 = output(model4)

output4.to_csv('4th_try.csv', index=False)
#Neural Network

from sklearn.neural_network import MLPClassifier

model5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)



output5 = output(model5)

output5.to_csv('5th_try.csv', index=False)
#Naive Bayes

from sklearn.naive_bayes import GaussianNB

model6 = GaussianNB()



output6 = output(model6)

output6.to_csv('6th_try.csv', index=False)
#Gradient Descent

from sklearn.linear_model import SGDClassifier

model7 = SGDClassifier(loss='modified_huber', shuffle=True,random_state=101)



output(model7).to_csv('7th_try.csv', index=False)
#K Nearest Neighbours

from sklearn.neighbors import KNeighborsClassifier

model8 = KNeighborsClassifier(n_neighbors = 15)



output(model8).to_csv('8th_try.csv', index=False)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

model9 = DecisionTreeClassifier(max_depth=10, random_state=101, max_features=None, min_samples_leaf = 15)



output(model9).to_csv('9th_try.csv', index=False)
#SVM

from sklearn.svm import SVC

model10 = SVC(kernel="linear", C=0.025, random_state=101)



output(model10).to_csv('10th_try.csv', index=False)
model11 = SVC()



output(model11).to_csv('11th_try.csv', index=False)