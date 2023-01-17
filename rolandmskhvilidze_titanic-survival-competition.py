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
#data analysis libraries 

import numpy as np

import pandas as pd



#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df= pd.read_csv('../input/titanic/train.csv')

df.head()
df.describe(include='all')
# extracting the usefull variablesfrom the original dataframe

train=df.loc[:, ['PassengerId', 'Survived', 'Pclass', 'Sex',

                 'Age', 'SibSp','Parch', 'Fare', 'Embarked' ]]

train.head()
# plot both together to compare

fig, ax=plt.subplots(2,2)

sns.barplot(x='Sex', y=('Survived'),  data=train, ax=ax[0, 0])

ax[0,0].set_title("SibSp")

sns.barplot(x='Pclass', y=('Survived'),  data=train, ax=ax[0, 1])

ax[0,1].set_title("Pclass")

sns.scatterplot(x='Age', y=('Survived'), data=train, ax=ax[1, 0]);





#print percentages of females vs. males that survive

print("Percentage of females who survived:", 

      train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", 

      train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)





#print percentages of first, second and third class passengers that survive

print("Percentage of first class who survived:", 

      train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of second class who survived:", 

      train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of second class who survived:", 

      train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)

# plot both together to compare

fig, ax=plt.subplots(2, 2)

sns.barplot(x='SibSp', y=('Survived'),  data=train, ax=ax[0, 0])

ax[0,0].set_title("SibSp")

sns.barplot(x='Parch', y=('Survived'),  data=train,  ax=ax[0, 1])

ax[0,1].set_title("Parch")

sns.barplot(x='Embarked', y=('Survived'), data=train, ax=ax[1, 0])

#ax[1, 0].set_title("Embarked")

sns.scatterplot(x='Fare', y=('Survived'), data=train, ax=ax[1, 1]);

#ax[1, 1].set_title("Fare")



#print percentages of passengers with none, one, two, three or four siblings or a spouse that survive

print("Percentage of none sibling or spouse who survived:", 

      train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)



print("Percentage of one sibling or spouse who survived:", 

      train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of two sibling or spouse who survived:", 

      train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of three sibling or spouse who survived:", 

      train["Survived"][train["SibSp"] == 3].value_counts(normalize = True)[1]*100)



print("Percentage of four sibling or spouse who survived:", 

      train["Survived"][train["SibSp"] == 4].value_counts(normalize = True)[1]*100)







#print percentages of passengers with none, one, two, three or five children or parents that survive

print("Percentage of none children or parents who survived:", 

      train["Survived"][train["Parch"] == 0].value_counts(normalize = True)[1]*100)



print("Percentage of 1 children or parents  who survived:", 

      train["Survived"][train["Parch"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of 2 children or parents  who survived:", 

      train["Survived"][train["Parch"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of 3 children or parents  who survived:", 

      train["Survived"][train["Parch"] == 3].value_counts(normalize = True)[1]*100)



print("Percentage of 5 children or parents  who survived:", 

      train["Survived"][train["Parch"] == 5].value_counts(normalize = True)[1]*100)



#print percentages of passengers with different ports of embarkation that survive

print("Percentage of passengers who embarked in Cherbourg who survived:", 

      train["Survived"][train["Embarked"] == 'C'].value_counts(normalize = True)[1]*100)



print("Percentage of passengers who embarked in Queenstown who survived:", 

      train["Survived"][train["Embarked"] == 'Q'].value_counts(normalize = True)[1]*100)



print("Percentage of passengers who embarked in Southampton who survived:", 

      train["Survived"][train["Embarked"] == 'S'].value_counts(normalize = True)[1]*100)
# filling the missings with median

median= train['Age'].median()

train['Age'].fillna(median, inplace=True)



# Check if it worked

train['Age'].describe()
groups = [train['Age'].between(0, 3), 

          train['Age'].between(4, 9), 

          train['Age'].between(10, 18),

          train['Age'].between(19, 59),

          train['Age'].between(60, 80)]

 



values = [1, 2, 3, 4, 5]



train['AgeGroup'] = np.select(groups, values, 0)



# check

train.head()
groups = [train['Age'].between(0, 3), 

          train['Age'].between(4, 9), 

          train['Age'].between(10, 18),

          train['Age'].between(19, 59),

          train['Age'].between(60, 80)]

 



values = ['Babies', 'Children', 'Teenagers', 'Adults', 'Seniors']



train['AgeGroup_names'] = np.select(groups, values, 0)
train.head()
train['Survived'].count()
# check



plt.figure(figsize=(10,6))

sns.barplot(x='AgeGroup_names',

            y= 'Survived',  

            data=train, 

            order=['Babies', 'Children', 'Teenagers', 'Adults', 'Seniors']);
train= train.drop(['Age'], axis=1)

train= train.drop(['AgeGroup_names'], axis=1)

#train.head()
# converting the values from sex variable into 1 an 0

d = {'male': 1, 'female': 0}

train['Sex'] = train['Sex'].map(d)



#check

train.head()
train['Embarked'].value_counts()
train['Embarked'].fillna(value='S', inplace=True)

# Check

train['Embarked'].describe()
# converting the values from Embarked variable into 1, 2 and 3

d = {'Q': 1, 'C': 2, 'S': 3}

train['Embarked'] = train['Embarked'].map(d)



#check

train.head()
from sklearn.model_selection import train_test_split



predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
test= pd.read_csv('../input/titanic/test.csv')

test.head()
# extracting the usefull variablesfrom the original dataframe

test=test.loc[:, ['PassengerId', 'Pclass', 'Sex',

                  'Age', 'SibSp','Parch', 'Fare',

                  'Embarked' ]]

test.head(5)
test.describe(include='all')
age_median=test['Age'].median()

test['Age'].fillna(value= age_median, inplace=True)
groups = [test['Age'].between(0, 3), 

          test['Age'].between(4, 9), 

          test['Age'].between(10, 18),

          test['Age'].between(19, 59),

          test['Age'].between(60, 76)]

 



values = [1, 2, 3, 4, 5]



test['AgeGroup'] = np.select(groups, values, 0)



# check

test.head()
test=test.drop(['Age'], axis=1)
fare_median=test['Fare'].median()

test['Fare'].fillna(value= fare_median, inplace=True)
# converting the values from sex variable into 1 an 0

d = {'male': 1, 'female': 0}

test['Sex'] = test['Sex'].map(d)



#check

test.head()
d={'Q':1, 'C':2, 'S':3}

test['Embarked']= test['Embarked'].map(d)
#Check

test.head()
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gaussian)
# Logistic Regression

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
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes',  

              'Decision Tree',  'Gradient Boosting Classifier'],

    'Score': [acc_svc,  acc_logreg, 

              acc_randomforest, acc_gaussian,  acc_decisiontree,

               acc_gbk]})

models.sort_values(by='Score', ascending=False)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = decisiontree.predict(test.drop('PassengerId', axis=1))
#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)
#Check

print(predictions, predictions.size)