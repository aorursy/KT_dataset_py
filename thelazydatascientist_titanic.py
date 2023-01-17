# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import data visualization library

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.preprocessing import LabelEncoder

# import sklearn model class

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier



# import sklearn model selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



# import sklearn model evaluation classification metrics

from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,

f1_score, fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

dataset = train.append(test,sort = False)
pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 50)
dataset.describe()
dataset.shape
dataset.isna().sum()
dataset.Embarked.value_counts()
dataset.Embarked = dataset.Embarked.fillna('S')
dataset.Fare = dataset.Fare.fillna(dataset.Fare.median())
print('No. of Unique Values Pclass = ',dataset.Pclass.nunique())

print('No. of Unique Values Sex = ',dataset.Sex.nunique())

print('No. of Unique Values Age = ',dataset.Age.nunique())

print('No. of Unique Values SibSp = ',dataset.SibSp.nunique())

print('No. of Unique Values Parch = ',dataset.Parch.nunique())

print('No. of Unique Values Embarked = ',dataset.Embarked.nunique())
dataset.head()
plt.rcParams['figure.figsize'] = (18,5)

sns.distplot((dataset['Fare']), color = 'red')

plt.title('Fare')
plt.rcParams['figure.figsize'] = (18,5)

sns.distplot(np.log1p(dataset['Fare']), color = 'red')

plt.title('Fare')
sns.countplot(dataset['Pclass'], palette = 'muted')

plt.title('P Classes',  fontsize = 30)
sns.countplot(dataset['Survived'], palette = 'muted')

plt.title('Survived',  fontsize = 30)
sns.countplot(dataset['SibSp'], palette = 'muted')

plt.title('Siblings/Spouse',  fontsize = 30)
sns.countplot(dataset['Sex'], palette = 'muted')

plt.title('Sex',  fontsize = 30)
sns.countplot(dataset['Parch'], palette = 'muted')

plt.title('Parents/Children',  fontsize = 30)
sns.countplot(dataset['Embarked'], palette = 'muted')

plt.title('Embarked',  fontsize = 30)
dataset.head()
dataset.groupby(['Sex','Pclass', 'Survived']).size()
def Pclass_sex(x):

    if x['Pclass'] == 1 and x['Sex'] == 'Female' : return 20

    elif x['Pclass'] == 2 and x['Sex'] == 'Female' : return 20

    elif x['Pclass'] == 3 and x['Sex'] == 'Female' : return 10

    else : return -2





dataset['Pclass_and_Sex'] = dataset.apply(Pclass_sex, axis = 1)
dataset['Title'] = dataset.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

dataset['LastName'] = dataset.Name.str.split(',').str[0]

dataset['IsWomanOrChild'] = ((dataset.Title == 'Master') | (dataset.Sex == 'female'))
#dataset['LastName'].value_counts(sort = False,dropna = False)

dataset['family_orno']=dataset.duplicated(subset='LastName', keep=False)
train.groupby(['SibSp', 'Survived']).size()
def Pclass_sex(x):

    if x['Pclass'] == 1 and x['Embarked'] == 'C' : return 'C1'

    elif x['Pclass'] == 1 and x['Embarked'] == 'Q' : return 'Q1'

    elif x['Pclass'] == 1 and x['Embarked'] == 'S' : return 'S1'

    elif x['Pclass'] == 2 and x['Embarked'] == 'C' : return 'C2'

    elif x['Pclass'] == 2 and x['Embarked'] == 'Q' : return 'Q2'

    elif x['Pclass'] == 2 and x['Embarked'] == 'S' : return 'S2'

    elif x['Pclass'] == 3 and x['Embarked'] == 'C' : return 'C3'

    elif x['Pclass'] == 3 and x['Embarked'] == 'Q' : return 'Q3'

    else : return 'S3'





dataset['Pclass_and_Embarked'] = dataset.apply(Pclass_sex, axis = 1)
lb = LabelEncoder()

dataset['Pclass_and_Embarked'] = lb.fit_transform(dataset['Pclass_and_Embarked'])
dataset['Family_size'] = dataset['SibSp'] + dataset['Parch']
def family_check(x):

    if x['Family_size'] > 0 : return 1

    else : return 0





dataset['Family_bool'] = dataset.apply(family_check, axis = 1)
dataset.Fare.value_counts(bins = (0,8.0600,14,27,32,60,

                                600))
def age_binned(x):

    if x['Age'] <= 10 : return 1

    elif x['Age'] > 10 and x['Age']<= 18 : return 2

    elif x['Age'] > 18 and x['Age'] <= 25 : return 3

    elif x['Age'] > 25 and x['Age'] <= 32 : return 4

    elif x['Age'] > 32 and x['Age'] <= 40 : return 5

    elif x['Age'] > 40 and x['Age'] <= 50 : return 6

    else : return 7





dataset['Age_binned'] = dataset.apply(age_binned, axis = 1)
def fare_binned(x):

    if x['Fare'] <= 8.0500 : return 1

    elif x['Fare'] > 8.0500 and x['Fare']<= 14 : return 2

    elif x['Fare'] > 14 and x['Fare'] <= 27 : return 3

    elif x['Fare'] > 27 and x['Fare'] <= 32 : return 4

    elif x['Fare'] > 32 and x['Fare'] <= 60 : return 5

    else : return 7





dataset['fare_binned'] = dataset.apply(fare_binned, axis = 1)
dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



for i in dataset:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Col','Don', 'Dr', 'Major', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Rev'], 'Crew')

    dataset['Title'] = dataset['Title'].replace('Capt', 'Crew')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
title_weights = {"Mr": 2, "Miss": 50, "Mrs": 50, "Master": 50, "Rare": 6, "Crew": 1}

dataset['Title'] = dataset['Title'].map(title_weights)
def length_name(x):

    if len(x['Name']) <= 27 : return 1

    elif len(x['Name']) > 27  and len(x['Name']) < 37: return 2

    elif len(x['Name']) > 37  and len(x['Name']) < 50: return 3

    else: return 4

    

dataset['Length_of_name'] = dataset.apply(length_name,axis = 1)



dataset.drop('Name',axis = 1,inplace = True)
dataset.info()
dataset.corr()
dataset['Deck'] = dataset.Cabin.str.get(0)

dataset['Deck'] = dataset['Deck'].fillna('NOTAVL')

dataset.Deck.replace('T' , 'G' , inplace = True)

dataset.drop('Cabin' , axis = 1 , inplace =True)
age_to_fill = train.groupby(['Pclass' , 'Sex' , 'Embarked'])[['Age']].median()

age_to_fill
for cl in range(1,4):

    for sex in ['male' , 'female']:

        for E in ['C' , 'Q' , 'S']:

            filll = pd.to_numeric(age_to_fill.xs(cl).xs(sex).xs(E).Age)

            dataset.loc[(dataset.Age.isna() & (dataset.Pclass == cl) & (dataset.Sex == sex) 

                    &(dataset.Embarked == E)) , 'Age'] =filll
dataset.info()
dataset.Ticket = pd.to_numeric(train.Ticket.str.split().str[-1] , errors='coerce')
Ticket_median = dataset[:len(train)].Ticket.median()

dataset.Ticket.fillna(Ticket_median , inplace =True)
dataset = pd.get_dummies(data=dataset, columns=['Deck', 'SibSp','Parch','Age_binned','Title','Pclass','Embarked','family_orno',

                                               'IsWomanOrChild','LastName','Sex'])
new_train,new_test = dataset[:len(train)],dataset[len(train):]

X,y = new_train.drop(['Survived','PassengerId'],axis = 1), new_train.Survived
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
from xgboost import XGBClassifier

alg = XGBClassifier(scale_pos_weight = 1,depth = 0,eta = 0.1,max_delta_step = 0.8,

                    min_child_weight = 2.1,

                    objective ='binary:logistic',

                    n_estimators = 400

                   )

alg.fit(X_train,y_train)

preds = alg.predict(X_val)

print (accuracy_score(y_val,preds))

print (confusion_matrix(y_val,preds))
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier,RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(objective = 'binary',num_iter = 120,learning_rate = 0.1,

                        scale_pos_weight = 2.9)

lgb.fit(new_train.drop(['Survived','PassengerId'],axis = 1), new_train.Survived)

preds = lgb.predict(new_test.drop(['Survived','PassengerId'],axis = 1))

#print (accuracy_score(y_val,preds))

#print (confusion_matrix(y_val,preds))
subbmissions = pd.read_csv('../input/gender_submission.csv')
subbmissions.Survived = preds
subbmissions.Survived = subbmissions.Survived.astype(int)
subbmissions.to_csv('gender_submission.csv',index = 0)
subbmissions.Survived.value_counts()