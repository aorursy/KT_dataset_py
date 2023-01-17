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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
test.drop(['Name', 'Cabin', 'Ticket'], axis = 1, inplace = True)
train.drop(['Name', 'PassengerId', 'Cabin'], axis = 1, inplace = True)
train.head()

missing = train.isnull()
missing
missingt = test.isnull()
for column in missing.columns.values.tolist():
    print (column)
    print (missing[column].value_counts())
    print ('')
for column in missingt.columns.values.tolist():
    print (column)
    print (missingt[column].value_counts())
    print ('')
avg_age = train['Age'].mean(axis=0)
print ("Average age is: ", avg_age)
train['Age'].replace(np.nan, avg_age, inplace = True)
avg_aget = test['Age'].mean(axis=0)
print ("Average age is: ", avg_aget)
test['Age'].replace(np.nan, avg_aget, inplace = True)
avg_fare = test['Fare'].mean(axis=0)
print ("Average fare is: ", avg_fare)
test['Fare'].replace(np.nan, avg_fare, inplace = True)
train['Embarked'].value_counts().idxmax()
train["Embarked"].fillna("S", inplace = True)
import matplotlib as plt
import seaborn as sns
%matplotlib inline
sns.regplot(x = 'SibSp', y = 'Survived', data= train)
train[["SibSp", "Survived"]].corr()
#No correlation between them
sns.regplot(x = 'Parch', y = 'Survived', data = train)
train[["Parch", "Survived"]].corr()
sns.regplot(x = 'Pclass', y = 'Survived', data = train)
train[["Pclass", "Survived"]].corr()
sns.regplot(x = 'Age', y = 'Survived', data = train)
train[["Age", "Survived"]].corr()
train.head()
missing = train.isnull()
for column in missing.columns.values.tolist():
    print(column)
    print (missing[column].value_counts())
    print("")
train.head()
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(train["Fare"])

# set x/y labels and plot title
plt.pyplot.xlabel("Fare")
plt.pyplot.ylabel("count")
plt.pyplot.title("fare")
bins = np.linspace(min(train['Fare']), max(train['Fare']), 4)
bins
group_names = ['Low', 'Medium', 'High']
train['Fare-bin'] = pd.cut(train['Fare'], bins, labels=group_names, include_lowest=True )
train[['Fare','Fare-bin']].head(20)
test['Fare-bin'] = pd.cut(test['Fare'], bins, labels=group_names, include_lowest=True )
test[['Fare','Fare-bin']].head(20)
train['Fare-bin'].value_counts()
pyplot.bar(group_names, train['Fare-bin'].value_counts())
train.head()
dummy1 = pd.get_dummies(train['Sex'])
dummy2 = pd.get_dummies(train['Fare-bin'])
dummy1.rename(columns={'Sex-male':'male', 'Sex-male':'female'}, inplace=True)
dummy1.head()
dummy2.rename(columns={'Fare-bin-Low':'Low', 'Fare-bin-Medium':'Medium', 'Fare-bin-High':'High'}, inplace=True)
dummy2.head()
train = pd.concat([train, dummy1], axis = 1)
train = pd.concat([train, dummy2], axis = 1)
train.head()
train.drop('Ticket', axis = 1, inplace = True)
train.head()
from scipy import stats
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
train["sex_code"] = lb_make.fit_transform(train["Sex"])
train["fare_code"] = lb_make.fit_transform(train["Fare-bin"])
train["embark_code"] = lb_make.fit_transform(train["Embarked"])
train.head()
lb_make = LabelEncoder()
test["sex_code"] = lb_make.fit_transform(test["Sex"])
test["fare_code"] = lb_make.fit_transform(test["Fare-bin"])
test["embark_code"] = lb_make.fit_transform(test["Embarked"])
test.head()
train.drop(['Sex','Fare','Fare-bin','Embarked','female','male','Low','Medium','High'], axis = 1, inplace = True)
test.drop(['Sex','Fare','Fare-bin','Embarked'], axis = 1, inplace = True)
test.head()
train.head()
pearson_coef, p_value = stats.pearsonr(train['Pclass'], train['Survived'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(train['Age'], train['Survived'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(train['SibSp'], train['Survived'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(train['Parch'], train['Survived'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(train['sex_code'], train['Survived'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(train['fare_code'], train['Survived'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(train['embark_code'], train['Survived'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
train.head()
test.head()
from sklearn.metrics import accuracy_score

X_train = train[['Age', 'Parch', 'fare_code']]
Y_train = train['Survived']

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

#y_pred= classifier.predict(X_train)

#print (accuracy_score(Y_train, y_pred))
X_train = train[['Age', 'Parch']]
Y_train = train['Survived']

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

#y_pred= classifier.predict(X_train)

#print (accuracy_score(Y_train, y_pred))
X_train = train[['Age', 'Parch', 'SibSp']]
Y_train = train['Survived']

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

#y_pred= classifier.predict(X_train)

#print (accuracy_score(Y_train, y_pred))
X_train = train[['Age', 'Parch', 'fare_code', 'Pclass', 'SibSp', 'sex_code', 'embark_code']]
Y_train = train['Survived']

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

#y_pred= classifier.predict(X_train)

#print ('Logistic Regression: ',accuracy_score(Y_train, y_pred))
X_train = train[['Age', 'Parch', 'fare_code', 'Pclass', 'SibSp', 'sex_code', 'embark_code']]
Y_train = train['Survived']

from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train,Y_train)

y_pred= classifier.predict(test[['Age', 'Parch', 'fare_code', 'Pclass', 'SibSp', 'sex_code', 'embark_code']])

#print ('KNN: ',accuracy_score(Y_train, y_pred))
X_train = train[['Age', 'Parch', 'fare_code', 'Pclass', 'SibSp', 'sex_code', 'embark_code']]
Y_train = train['Survived']

from sklearn.svm import SVC
classifier= SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,Y_train)

#y_pred= classifier.predict(X_train)

#print ('SVM: ',accuracy_score(Y_train, y_pred))
X_train = train[['Age', 'Parch', 'fare_code', 'Pclass', 'SibSp', 'sex_code', 'embark_code']]
Y_train = train['Survived']

from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(X_train,Y_train)

#y_pred= classifier.predict(X_train)

#print ('Naive Bayes: ',accuracy_score(Y_train, y_pred))
X_train = train[['Age', 'Parch', 'fare_code', 'Pclass', 'SibSp', 'sex_code', 'embark_code']]
Y_train = train['Survived']

from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

#y_pred= classifier.predict(X_train)

#print ('Decision Tree: ',accuracy_score(Y_train, y_pred))
X_train = train[['Age', 'Parch', 'fare_code', 'Pclass', 'SibSp', 'sex_code', 'embark_code']]
Y_train = train['Survived']

from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

#y_pred= classifier.predict(X_train)

#print ('Decision Tree: ',accuracy_score(Y_train, y_pred))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)
