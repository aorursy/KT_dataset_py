# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')
titanic_train.head()


titanic_train.info()

#Using groupby, we can make tables computing survival rates of different group
by_sex = titanic_train.groupby('Sex')['Survived'].mean()
pd.DataFrame(by_sex)

#Female group's survival rate is 4 times larger than the male group's survival rate. 
by_Pclass = titanic_train.groupby('Pclass')['Survived'].agg(['sum','count'])     
by_Pclass['survival_rate'] = by_Pclass['sum'].divide(by_Pclass['count'])
by_Pclass

#The survival rate of Pclass 1 is 0.63, almost as three times that for Pclass 3.
by_sex_and_Pclass = titanic_train.groupby(['Sex', 'Pclass'])['Survived'].mean()
pd.DataFrame(by_sex_and_Pclass)

#Female group's survival rate is 4 times larger than the male group's survival rate. 
#We can plot the tables above

fig, (ax1, ax2) = plt.subplots(1, 2)
sns.barplot(x='Sex', y='Survived', data=titanic_train, ax=ax1)
sns.barplot(x ='Sex', y ='Survived', hue='Pclass', data=titanic_train, ax=ax2)

# The survival rate of female is much higher than of male.
# The survival rate of passengers in class 3 is almost half of those in class 1, independent of sex. 
# For the female group, the survival rate of passengers in class 2 is similar to those in class 1; 
#   while for the male group, it is much lower.

fig, (ax1, ax2) = plt.subplots(1, 2)
sns.barplot(x='Pclass', y='Survived', data=titanic_train, ax=ax1)
sns.barplot(x ='Pclass', y='Survived', hue='Sex', data = titanic_train, ax=ax2)

#The survival rate of passengers in class 1 is much higher than those in class 3.
#Females who stay in class 1 and 2 have very high rates of survival.
fig, (ax1, ax2) = plt.subplots(1, 2)
sns.boxplot(x='Survived', y='Age', data=titanic_train, ax=ax1)
sns.barplot(x='Survived', y ='Age', hue='Sex', data=titanic_train, ax=ax2)

#Looks like inside the survived group, the age quantiles of the male group and female group are very similar. 
#For the dead group, the median age of the male group is a bit larger than the median age of the female group.
#It may be better to create age_range and compare survival rate among different age_range groups.

bins = [0, 5, 16, 30, 45, 60, 75, 80]
age_range = ['0-5', '5-16', '16-30','30-45', '45-60', '60-75', '75-']
titanic_train['age_range'] = pd.cut(titanic_train['Age'], bins, labels=age_range)

#Groupby age_range
by_age_range = titanic_train.groupby('age_range')['Survived'].mean()
print(by_age_range)

#Among the groups, the infant group has the highest survival rate at 70%. The 16-30 group has the lowest rate at 36%.

sns.barplot(x='age_range', y='Survived', data=titanic_train)
by_embarked = titanic_train.groupby(['Embarked', 'Sex'])['Survived'].mean()
by_embarked
#It is very interesting that males who embarked from C=Charbough has much higher rate of survival.
titanic_train.groupby(['Embarked', 'Sex'])['Name'].count()

#Nearly 2/3 of the sampled passengers embarked at S=Southampton.
titanic_train.groupby(['Embarked','Pclass'])['Name'].count()

#We can see that there is a correlation between Pclass and Embarked. For example, %50 of people who embarked at C=Cherbough
#are in class 1 while most of people who embarked at Q=Queenstown are in class 3. 
#We now modify our dataset to make it more suitable for implementing some machine learning techniques.
#First, we need to fill in the missing age. There are only 714 available observations out of 891 datas. 
#We will fill these NA (of both training and test datasets) by the age median groupby Pclass and Sex.

def fill_na_age(df):
    age_median_by_group = pd.DataFrame(df.groupby(['Sex', 'Pclass'])['Age'].median().reset_index())
    age_median_by_group.columns = ['Sex', 'Pclass', 'Median_age']
    df = pd.merge(df, age_median_by_group, on = ['Sex', 'Pclass'])
    df['Age'] = df['Age'].fillna(df['Median_age'])
    df.drop('Median_age', axis=1)
    return df

titanic_train = fill_na_age(titanic_train)
titanic_test = fill_na_age(titanic_test)

#Convert the Sex column into numerical variable, Embarked column into numerical variable

titanic_train['Sex'] = titanic_train['Sex'].map({'female': 10, 'male': 1}).astype(int)
titanic_test['Sex'] = titanic_test['Sex'].map({'female': 10, 'male': 1}).astype(int)


#Make age_range column as a new categorical variable

def create_age_range(df):
    bins = [0, 5, 16, 30, 45, 60, 75, 80]
    age_range = ['0-5', '5-16', '16-30','30-45', '45-60', '60-75', '75-']
    df['age_range'] = pd.cut(df['Age'], bins, labels=age_range)
    return df

titanic_train = create_age_range(titanic_train)
titanic_test = create_age_range(titanic_test)

#Change the age_range column as a new categorical variable

titanic_train['age_range'] = titanic_train['age_range'].map({'0-5':0, '5-16':1, '16-30':2,'30-45':3, '45-60':4, '60-75':5, '75-':6}).astype(int)
titanic_test['age_range'] = titanic_test['age_range'].map({'0-5':0, '5-16':1, '16-30':2,'30-45':3, '45-60':4, '60-75':5, '75-':6}).astype(int)


#Change the Age column to int type
titanic_train['Age'] = titanic_train['Age'].astype(int)
titanic_test['Age'] = titanic_test['Age'].astype(int)
#Let's analyze SibSp and Parch by creating a new column Companion: counting the number of family members who were with the passenger.
titanic_train['Companion'] = titanic_train['SibSp'] + titanic_train['Parch']
titanic_test['Companion'] = titanic_train['SibSp'] + titanic_train['Parch']
titanic_train.groupby('Companion')[['Survived', 'Sex']].agg(['mean', 'count'])

#There is some correlation between Companion and Sex. Quite the majority of passengers do not have companion.
#Passengers without companion are 75% male, while passengers with 1 or 2 companions are only 50% male. 
#The survival rate of each group (grouped by the number of companions) are also very different. 
# We eliminate Passenger ID because they are probably not correlated with being survived or not. The Cabin column
# is very incomplete, only 204 data points. We finally also drop SibSp, Parch while keep Companion

train_var = ['Sex', 'Age', 'Pclass', 'Fare', 'Embarked', 'Companion', 'age_range', 'Survived']
titanic_train = titanic_train[train_var]
test_var = ['PassengerId', 'Sex', 'Age', 'Pclass', 'Fare', 'Embarked', 'age_range', 'Companion']
titanic_test = titanic_test[test_var]
titanic_train.head()

#In this training sample, the survived rate is 38.3%
#The mean age is 29.69.
#At least 75% of the sampled passengers are in class 2 or 3, with more than 50% are in class 3.
#Checking the test file, we see that there is a missing entry in Fare and two missing entries in Embarked.
#We will fill the NA in Fare with the median, while NA in Embarked in the most frequent entry in test.
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
titanic_train['Embarked'] = titanic_train['Embarked'].fillna(titanic_train['Embarked'].dropna().mode()[0])
titanic_test['Embarked'] = titanic_test['Embarked'].fillna(titanic_test['Embarked'].mode()[0])


#Change the Embarked from a categorical variable to a numerical variable
titanic_train['Embarked'] = titanic_train['Embarked'].map({'C': 3,'Q': 2, 'S':1}).astype(int)
titanic_test['Embarked'] = titanic_test['Embarked'].map({'C': 3,'Q':2, 'S': 1}).astype(int)
# Add extra variable for the Titanic:
titanic_train['Emb_Sex'] = titanic_train['Embarked']*titanic_train['Sex']
titanic_test['Emb_Sex'] = titanic_test['Embarked']*titanic_train['Sex']
titanic_train['P_sex'] = titanic_train['Pclass']*titanic_train['Sex']
titanic_test['P_sex'] = titanic_test['Pclass']*titanic_train['Sex']
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

X_train = titanic_train.drop('Survived', axis=1)
Y_train = titanic_train['Survived']
X_test = titanic_test.drop('PassengerId', axis=1)
titanic_train.head()
score = {}
X_tr, X_te, Y_tr, Y_te = train_test_split(X_train, Y_train , test_size = 0.2, random_state=1)
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_tr, Y_tr)
Y_pred = logreg.predict(X_te)

Y_pred_prob = logreg.predict_proba(X_te)[:,1]

fpr, tpr, thresholds = roc_curve(Y_te, Y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


print(confusion_matrix(Y_pred, Y_te))
print(roc_auc_score(Y_te, Y_pred_prob))

log_regression = LogisticRegression()
log_regression.fit(X_train, Y_train)
Y_pred = log_regression.predict(X_test)
score['Logistic Regression'] = log_regression.score(X_train, Y_train)

my_submission = pd.DataFrame({'PassengerId': titanic_test['PassengerId'], 'Survived': Y_pred})
my_submission.to_csv('submission.csv', index=False)
#When I tried these models on the test set, the results were only 75%. Maybe we have overfitting problem. Let's take away a few columns 
#that have strong correlation with other variables.

train_var = ['Sex', 'Age', 'Pclass', 'age_range', 'Survived']
titanic_train = titanic_train[train_var]
test_var = ['PassengerId', 'Sex', 'Age', 'Pclass', 'age_range']
titanic_test = titanic_test[test_var]
X_train = titanic_train.drop('Survived', axis=1)
Y_train = titanic_train['Survived']
X_test = titanic_test.drop('PassengerId', axis=1)

score = {}
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
score['LogisticRegression'] = logreg.score(X_train, Y_train)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
result = pd.DataFrame([score]).T
result.columns = ['Score']
result
#SVC
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
score['SVC'] = svc.score(X_train, Y_train)

#Linear SVC
Linear_svc = LinearSVC()
Linear_svc.fit(X_train, Y_train)
Y_pred = Linear_svc.predict(X_test)
score['LinearSVC'] = Linear_svc.score(X_train, Y_train)

#DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
score['DecisionTreeClassifier'] = decision_tree.score(X_train, Y_train)

#RandomForest
random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
score['RandomForestClassifier'] = random_forest.score(X_train, Y_train)

#KNeighborsClassifier
knbor = KNeighborsClassifier()
knbor.fit(X_train, Y_train)
Y_pred = knbor.predict(X_test)
score['KneighborsClassifier'] = knbor.score(X_train, Y_train)
result = pd.DataFrame([score]).T
result.columns = ['Score']
result

