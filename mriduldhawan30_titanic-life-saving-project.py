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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, auc, f1_score, precision_recall_curve, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.preprocessing import normalize

sns.set_style('darkgrid')

from sklearn.metrics import roc_auc_score, roc_curve
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
train.shape
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
test.shape
train.describe()
test.describe()
train.isnull().sum()
test.isnull().sum()
train.Age = train.Age.fillna(train['Age'].mean())

train.Embarked = train.Embarked.fillna(np.random.choice(['C','Q','S']))

test.Age = test.Age.fillna(test['Age'].mean())

test.Fare = test.Fare.fillna(test['Fare'].mean())
train.isnull().sum()
test.isnull().sum()
train.corr()
plt.figure(figsize = (8,6))

sns.heatmap(train.corr())
plt.figure(figsize = (8,6))

sns.heatmap(test.corr())
fig, axes = plt.subplots(2,4,figsize=(18,12))

sns.countplot(x = 'Sex', data = train, ax = axes[0,0], edgecolor = 'blue')

sns.countplot(x = 'Survived', data = train, ax = axes[0,1], edgecolor = 'blue')

sns.countplot(x = 'Parch', data = train, ax = axes[0,2], edgecolor = 'blue')

sns.countplot(x = 'SibSp', data = train, ax = axes[0,3], edgecolor = 'blue')

sns.countplot(x = 'Embarked', data = train, ax = axes[1,0], edgecolor = 'blue')

sns.countplot(x = 'Pclass', data = train, ax = axes[1,1], edgecolor = 'blue')

sns.distplot(train['Age'], ax = axes[1,2] )

sns.distplot(train['Fare'], ax = axes[1,3])

plt.show()
# pie chart for Male-Female %age

plt.figure(figsize = (8,6))

plt.title('Male-Female percent', fontsize = 20)

x_sex, y_sex = train.Sex.value_counts(normalize=True)*100

wedges = plt.pie([x_sex, y_sex], labels = ['Male','Female'], colors = ['green','blue'], autopct = '%.2f%%', 

                 textprops = {'color':'Black','size':20})

plt.show()
# pie chart for survived-not survived

plt.figure(figsize=(8,6))

plt.title('Survived-Not Srvived percent', fontsize=20)

x_surv, y_surv = train.Survived.value_counts(normalize=True)*100

plt.pie([x_surv,y_surv],labels = ['Survived','Not Survived'], colors = ['green','blue'], autopct= '%.2f%%',

       textprops = {'color':'black', 'size': 20})

plt.show()
def get_title(string):

    if '.' in string:

        return string.split(',')[1].split('.')[0].strip()

    else:

        return 'N.F'



def replace_titles(x):

    title = x['Title']

    if title in ['Capt','Col','Dona','Don','Jonkheer','Major','Rev','Sir']:

        return 'Mr'

    elif title in ['the Countess','Mme','Lady']:

        return 'Mrs'

    elif title in ['Mlle','Ms']:

        return 'Miss'

    elif title == 'Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

            return title

        

train['Title'] = train['Name'].apply(get_title)

temp_title = train.apply(replace_titles, axis = 1)

temp_title.value_counts()

sur = train[train['Survived']==1]

survived_title = sur.apply(replace_titles, axis=1)

survived_title.value_counts()
temp_title.value_counts()
# Train Set Creation

train['Ticket'] = train['Ticket'].apply(lambda x: str(x))

train['Ticket'] = train['Ticket'].apply(lambda x: len(x))

train['Title'] = train['Name'].apply(get_title)

train['Title'] = train.apply(replace_titles, axis = 1)

drop_cols = ['PassengerId','Name','Cabin','Title']

encode_cols = ['Sex','Embarked','Title']

encode_after = pd.get_dummies(train[encode_cols])

final_data = train.copy()

final_data = final_data.drop(drop_cols, axis=1)

final_data = pd.concat([final_data, encode_after], axis = 1)

print(final_data.columns)

final_data.drop(['Sex','Embarked'],axis = 1, inplace = True)

final_data.head()
# Test Set Creation

test['Ticket'] = test['Ticket'].apply(lambda x: str(x))

test['Ticket'] = test['Ticket'].apply(lambda x: len(x))

test['Title'] = test['Name'].apply(get_title)

test['Title'] = test.apply(replace_titles, axis = 1)

encode_cols_test = pd.get_dummies(test[encode_cols])

final_test = test.copy()

final_test = final_test.drop(['PassengerId','Name','Cabin','Sex','Embarked','Title'], axis = 1)

final_test = pd.concat([final_test, encode_cols_test], axis=1)

print(final_test.columns)

final_test.head()
gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

gender.head()
X_train = final_data.iloc[:,1:]

y_train = final_data.iloc[:,0]

X_test = final_test.iloc[:,:]

y_test = gender.iloc[:,1:]
model = LogisticRegression()

model.fit(X_train,y_train)

y_predict = model.predict(X_test)

print(classification_report(y_test, y_predict))

print(confusion_matrix(y_test, y_predict))

print(accuracy_score(y_test,y_predict))
auc_score_model = roc_auc_score(y_test,y_predict)

print('AUROC Score : {0}'.format(round(auc_score_model, 4)))

fpr, tpr, threshold = roc_curve(y_test, y_predict)

plt.figure(figsize = (12,7))

plt.plot([0,1],[0,1],'--')

plt.plot(fpr,tpr,'-*',color = 'Red', label = 'ROC Curve Area : {0}'.format(round(auc_score_model, 4)))

plt.xlabel('False Positive rate', fontdict = {'fontsize':15})

plt.ylabel('True Positive rate', fontdict = {'fontsize': 15})

plt.title('ROC-AUC Curve (Logistic Regression)', fontdict = {'fontsize':20})

plt.legend()

plt.show()
model1 = RandomForestClassifier()

model1.fit(X_train, y_train)

y1_predict = model1.predict(X_test)

print(classification_report(y_test, y1_predict))

print(confusion_matrix(y_test, y1_predict))

print(accuracy_score(y_test,y1_predict))
auc_score_model1 = roc_auc_score(y_test,y1_predict)

print('AUROC Score : {0}'.format(round(auc_score_model1, 4)))

fpr, tpr, threshold = roc_curve(y_test, y1_predict)

plt.figure(figsize = (12,7))

plt.plot([0,1],[0,1],'--')

plt.plot(fpr,tpr,'-*',color = 'Red', label = 'ROC Curve Area : {0}'.format(round(auc_score_model1, 4)))

plt.xlabel('False Positive rate', fontdict = {'fontsize':15})

plt.ylabel('True Positive rate', fontdict = {'fontsize': 15})

plt.title('ROC-AUC Curve (Logistic Regression)', fontdict = {'fontsize':20})

plt.legend()

plt.show()
# As accuracy is more with Logistic regression, so it is taken as finalized model.
submission = pd.DataFrame({

       "PassengerId": gender['PassengerId'],

       "Survived": y1_predict

})



submission.PassengerId = submission.PassengerId.astype(int)

submission.Survived = submission.Survived.astype(int)



submission.to_csv('titanic_submission.csv', index = False)