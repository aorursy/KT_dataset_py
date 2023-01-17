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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings=all

import numpy as np

import os
df = pd.read_csv('../input/titanic/train.csv')
df.head(10)
df.describe(include="all")
dataframe = df.dropna(how='all',axis=1)
dataframe
dataframe.describe(include = 'all' )


dataframe.groupby(['Survived']).count()
dataframe.groupby(['Sex']).count()
dataframe.groupby(['Pclass']).count()
dataframe.groupby(['SibSp']).count()
dataframe.groupby(['Parch']).count()
plt.title("Titanic Survied")

plt.xlabel("Pclass")

plt.ylabel("Survived")



x = df["Pclass"]

y = df["Survived"]

sns.barplot(x,y,data=dataframe)

print("Percentage of Class1 who survived:", dataframe["Survived"][dataframe["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Class2 who survived:", dataframe["Survived"][dataframe["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Class3 who survived:", dataframe["Survived"][dataframe["Pclass"] == 3].value_counts(normalize = True)[1]*100)
sns.barplot('Sex','Survived',data=dataframe)

print("Percentage of female who survived:", dataframe["Survived"][dataframe["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of Class2 who survived:", dataframe["Survived"][dataframe["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
sns.barplot('SibSp','Survived',data=dataframe)

print("Percentage of single survived:", dataframe["Survived"][dataframe["SibSp"]== 0].value_counts(normalize = True)[1] * 100)

print("Percentage of Sibling Spous 1 who survived:", dataframe["Survived"][dataframe["SibSp"]== 1].value_counts(normalize = True)[1] * 100)

print("Percentage of Sibling Spous 2 who survived:", dataframe["Survived"][dataframe["SibSp"]== 2].value_counts(normalize = True)[1] * 100)

print("Percentage of Sibling Spous 3 who survived:", dataframe["Survived"][dataframe["SibSp"]== 3].value_counts(normalize = True)[1] * 100)

sns.barplot('Parch','Survived',data=dataframe)

print("Percentage of Single survived:", dataframe["Survived"][dataframe["SibSp"]== 0].value_counts(normalize = True)[1] * 100)

print("Percentage of Parents & Childresns 1 who survived:", dataframe["Survived"][dataframe["SibSp"]== 1].value_counts(normalize = True)[1] * 100)

print("Percentage of Parents & Childresns 2 who survived:", dataframe["Survived"][dataframe["SibSp"]== 2].value_counts(normalize = True)[1] * 100)



a =dataframe["Age"]//10



#print(a)

sns.barplot(a,'Survived',data=dataframe)

#print("Percentage of Age 0's  survived:", dataframe["Survived"][dataframe["Age"]//10 == 0].value_counts(normalize = True)[1] * 100)

#print("Percentage of Parents & Childresns 1 who survived:", dataframe["Survived"][dataframe["SibSp"]== 1].value_counts(normalize = True)[1] * 100)

#print("Percentage of Parents & Childresns 2 who survived:", dataframe["Survived"][dataframe["SibSp"]== 2].value_counts(normalize = True)[1] * 100)

dataframe.head()
dataframe.drop(["Name"],axis =1)
dataframe = dataframe.dropna(axis=1)

dataframe = dataframe.drop(['Name'], axis = 1)


dataframe = dataframe.drop(['Ticket'], axis = 1)

dataframe.describe()
dataframe.head(5)

#dataframe = dataframe.values
#from sklearn.preprocessing import LabelEncoder

#labelencoder_x = LabelEncoder()

#dataframe[:,3] = labelencoder_x.fit_transform(dataframe[:,3])

sex_mapping = {"male": 0, "female": 1 ,"Nan":3}

dataframe['Sex'] = dataframe['Sex'].map(sex_mapping)

#dataframe['Sex'] = dataframe['Sex'].map(sex_mapping)

dataframe
#df = df.drop(['Cabin'], axis = 1)

#dataframe = dataframe.drop(['Ticket'], axis = 1)

#dataframe = dataframe.drop(['Age'], axis = 1)

#df = df.drop(['Name'], axis = 1)



dataframe.head()

dataframe = dataframe.drop(['Fare'],axis=1)

#dataframe = dataframe.drop['Fare']



dataframe.head()
!pip install sklearn
!pip install --upgrade scikit-learn
from sklearn.model_selection import train_test_split



predictors = dataframe.drop(['Survived'], axis=1)
target = dataframe["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gaussian)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
# Linear SVC

from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_val)

acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_svc)

#Decision Tree

from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
# KNN or k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)

acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_val)

acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_sgd)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes',  'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbk]})

models
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head(5)

#test = test.drop(['Fare'], axis = 1)

test = test.drop(['Name'], axis = 1)

test = test.drop(['Cabin'], axis = 1)

test = test.drop(['Embarked'], axis = 1)

test = test.drop(['Ticket'], axis = 1)

#test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

#map each Sex value to a numerical value

sex_mapping = {"male": 0, "female": 1}



test['Sex'] = test['Sex'].map(sex_mapping)

test.head(5)

test = test.dropna(how='all',axis=1)
dataframe.head(5)
test.head(5)

test.dropna(axis=1)

#test["Age"] = test["Age""].astype(int)

#test["Fare""] = test["Fare"].astype(int)
from sklearn.ensemble import RandomForestClassifier



y = dataframe["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(dataframe[features])

X_test = pd.get_dummies(test[features])


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)


output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output