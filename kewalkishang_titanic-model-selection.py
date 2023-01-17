# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
trainset = pd.read_csv("../input/train.csv")

testset = pd.read_csv("../input/test.csv")
#Perfomaring exploratory data analysis

trainset.shape
testset.columns
trainset.columns
testset['PassengerId']
x_train=trainset.drop(['Name','Cabin','Ticket','PassengerId'],axis=1)

x_test=testset.drop(['Name','Cabin','Ticket','PassengerId'],axis=1)



#y_train.shape

for colName in x_test.columns:

    print(colName ,":" ,x_test[colName].isnull().values.any())
for colName in x_train.columns:

    print(colName ,":" ,x_train[colName].isnull().values.any())
x_test.columns
x_train.columns
#dealing with NAN values

x_train['Embarked'].isnull().value_counts()
x_train['Age'].isnull().value_counts()
# Taking care of missing data

x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)

x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mode()[0], inplace=True)

x_train=x_train.dropna(axis=0,how='any')
x_train
x_test.columns

y_train=x_train['Survived']

x_train=x_train.drop(['Survived'],axis=1)
x_test.columns


# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder

sex_encoder = LabelEncoder()

x_train['Sex'] = sex_encoder.fit_transform(x_train['Sex'])



x_test['Sex'] = sex_encoder.fit_transform(x_test['Sex'])
embarked_encoder = LabelEncoder()

x_train['Embarked'] = embarked_encoder.fit_transform(x_train['Embarked'])



x_test['Embarked'] = embarked_encoder.fit_transform(x_test['Embarked'])
x_train
x_train=x_train.drop(['Survived'],axis=1)
x_train
x_train.corr()
plt.matshow(x_train.corr())

plt.xticks(range(len(x_train.columns)),x_train.columns)

plt.yticks(range(len(x_train.columns)),x_train.columns)

plt.colorbar()

plt.show()
#Fitting on different models

from sklearn.metrics import accuracy_score

# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(x_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(x_train)

accuracy_score(y_train, y_pred)
#Multi-Layer Perceptron

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(6,6,6),max_iter=1000 )

clf.fit(x_train, y_train) 

y_predmlp = clf.predict(x_train)

accuracy_score(y_train,y_predmlp)
#Gradient Boosting

from xgboost.sklearn import XGBClassifier

clfX=XGBClassifier()

clfX.fit(x_train, y_train) 

y_predX = clfX.predict(x_train)

accuracy_score(y_train,y_predX)
from sklearn.ensemble import RandomForestClassifier

regressor = RandomForestClassifier(n_estimators = 10, random_state = 0)

regressor.fit(x_train, y_train)

y_predrf = regressor.predict(x_train)

accuracy_score(y_train,y_predrf)
from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier() 

model.fit(x_train,y_train)

prediction=model.predict(x_train)

accuracy_score(prediction,y_train)
from sklearn.naive_bayes import GaussianNB

model=GaussianNB()

model.fit(x_train,y_train)

prediction=model.predict(x_train)

accuracy_score(prediction,y_train)
from sklearn import svm

model=svm.SVC(kernel='rbf',C=1,gamma=0.1)

model.fit(x_train,y_train)

prediction=model.predict(x_train)

accuracy_score(prediction,y_train)
df = pd.DataFrame({"PassengerId": testset['PassengerId'], "Survived":regressor.predict(x_test) })

df.to_csv('submission.csv', index=False)

df