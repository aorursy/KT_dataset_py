# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

train.head()
test = pd.read_csv("../input/test.csv")

test.head()
all = pd.concat([train, test], sort = False)

all.info()
all.describe()
all.isnull().sum()
#Fill Missing numbers with median

all['Age'] = all['Age'].fillna(value=all['Age'].median())

all['Fare'] = all['Fare'].fillna(value=all['Fare'].median())

all['Cabin'] = all['Cabin'].fillna('Missing')

all['Cabin'] = all['Cabin'].str[0]

all['Embarked'] = all['Embarked'].fillna(value=all['Embarked'].mode().values[0])
#checking label in balance for target variable

sns.catplot(x = 'Survived', kind = 'count', data = all) #or all['Survived'].value_counts()
sns.catplot(x = 'Embarked', kind = 'count', data = all) #or all['Embarked'].value_counts()
#Age

all.loc[ all['Age'] <= 16, 'Age'] = 0

all.loc[(all['Age'] > 16) & (all['Age'] <= 32), 'Age'] = 1

all.loc[(all['Age'] > 32) & (all['Age'] <= 48), 'Age'] = 2

all.loc[(all['Age'] > 48) & (all['Age'] <= 64), 'Age'] = 3

all.loc[ all['Age'] > 64, 'Age'] = 4 
#Title

import re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+\.)', name)

    

    if title_search:

        return title_search.group(1)

    return ""
all['Title'] = all['Name'].apply(get_title)

all['Title'].value_counts()
all['Title'] = all['Title'].replace(['Capt.', 'Dr.', 'Major.', 'Rev.'], 'Officer.')

all['Title'] = all['Title'].replace(['Lady.', 'Countess.', 'Don.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Royal.')

all['Title'] = all['Title'].replace(['Mlle.', 'Ms.'], 'Miss.')

all['Title'] = all['Title'].replace(['Mme.'], 'Mrs.')

all['Title'].value_counts()
all['Cabin'].value_counts()
#Family Size & Alone 

all['Family_Size'] = all['SibSp'] + all['Parch'] + 1

all['IsAlone'] = 0

all.loc[all['Family_Size']==1, 'IsAlone'] = 1

all.head()
#Drop unwanted variables

all_1 = all.drop(['Name', 'Ticket'], axis = 1)

all_1.head()
from scipy.stats import chi2_contingency

def print_chi2(df,name):

    print(name)

    chi2 = chi2_contingency(df)

    #print("Expected:")

    #print(chi2[3])

    chisquare = chi2[0].round(4)

    pvalue = chi2[1].round(4)

    print("chisquare",chisquare," pvalue:",pvalue)

    print("")
cat_attributes = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Title']

cat_col = 'Survived'

for cat in cat_attributes:

    data = pd.crosstab(all_1[cat], 

                            all_1[cat_col],  

                               margins = False)

    print_chi2(data,cat)
all_dummies = pd.get_dummies(all_1, drop_first = True)

all_dummies.head()
all_dummies.columns
total_columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',

       'Family_Size', 'IsAlone', 'Sex_male', 'Cabin_B', 'Cabin_C', 'Cabin_D',

       'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_M', 'Cabin_T', 'Embarked_Q',

       'Embarked_S', 'Title_Master.', 'Title_Miss.', 'Title_Mr.', 'Title_Mrs.',

       'Title_Officer.', 'Title_Royal.']

f,ax = plt.subplots(figsize=(26, 26))

sns.heatmap(all_dummies[total_columns].corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax);
#filtered column with high corelation > .50

filtered_columns=['Pclass', 'Age', 'SibSp', 'Parch',

       'Sex_male', 'Cabin_B', 'Cabin_C', 'Cabin_D',

       'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T',

       'Embarked_S', 'Title_Master.', 'Title_Officer.', 'Title_Royal.']

f,ax = plt.subplots(figsize=(26, 26))

sns.heatmap(all_dummies[filtered_columns].corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax);
all_train = all_dummies[all_dummies['Survived'].notna()]
all_test = all_dummies[all_dummies['Survived'].isna()]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(all_train[filtered_columns], 

                                                    all_train['Survived'], test_size=0.30, 

                                                    random_state=0)
from sklearn.metrics import classification_report, confusion_matrix

def check_accuracy(model, y_test, X_test):

    predictions = model.predict(X_test)

    print(model.score(X_test, y_test))

    print(classification_report(y_test,predictions))

    print(confusion_matrix(y_test,predictions))
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

logmodel = LogisticRegression(solver = 'liblinear', class_weight='balanced')

X_train_standard = scaler.fit_transform(X_train, y_train)

X_test_standard = scaler.transform(X_test)

logmodel.fit(X_train_standard,y_train)

check_accuracy(logmodel, y_test, X_test_standard)
from sklearn.ensemble import GradientBoostingClassifier

n_estimator=[130]

for estimator in n_estimator:

    gbmodel = GradientBoostingClassifier(random_state=0,n_estimators=estimator,max_features='sqrt')

    gbmodel.fit(X_train, y_train)

    print(estimator,gbmodel.score(X_test, y_test))

check_accuracy(gbmodel, y_test, X_test)
from sklearn.model_selection import GridSearchCV

parameters ={

    'criterion': ['friedman_mse'], 

     'learning_rate': [0.075], 

     'loss': ['deviance'], 

     'max_depth': [3], 

     'max_features': ['log2','sqrt'], 

     'min_samples_leaf': [0.1], 

     'min_samples_split': [0.3545454545454546], 

     'n_estimators': [160, 170, 180, 190, 200], 

     'subsample': [1.0]}



gbcvmodel = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)



gbcvmodel.fit(X_train, y_train)

print(gbcvmodel.score(X_train, y_train))

print(gbcvmodel.best_params_)

check_accuracy(gbcvmodel, y_test, X_test)
best_model = gbmodel
all_test.head()
TestForPred = all_test.drop(['PassengerId', 'Survived'], axis = 1)
t_pred = best_model.predict(TestForPred[filtered_columns]).astype(int)
PassengerId = all_test['PassengerId']
logSub = pd.DataFrame({'PassengerId': PassengerId, 'Survived':t_pred })

logSub.head()
logSub.to_csv("1_Logistics_Regression_Submission.csv", index = False)