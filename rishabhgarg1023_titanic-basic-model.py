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
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.naive_bayes import GaussianNB

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn import model_selection

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
train_df=pd.read_csv('../input/titanic/train.csv')

test_df=pd.read_csv('../input/titanic/test.csv')

submission_df=pd.read_csv('../input/titanic/gender_submission.csv')
train_df.head()
train_df.info()

#age, cabin, and embarked have null values
test_df.info()

# Age cabin and fare have null values
#merging data for easy processing

all_df=pd.concat([train_df, test_df], ignore_index= True)

all_df.tail()
all_df.info()
all_df.shape
#Filling missing values

all_df
# Ticket no. 3701 has null fare in test table. He embarked from station S in pclass 3. Since fare prices are different for different pclass. For now i will fill this value with median of fare values for pclass 3

Med_val=all_df[all_df['Pclass']==3]['Fare'].median()

all_df.loc[all_df['Ticket']=='3701',['Fare']]=Med_val
#Embarked has two missing values, both are females who are travelling on same ticket no. 113572 in pclass 1, sharing same cabin.

# It means both of them embarked from same port. Out of those in pclass 1, 141 embarked from C and 177 embarked from S. Only 3 passengers embarked from Q.

# So i'll just fill both values with S, it will not effect the outcome much though

all_df.loc[all_df['Ticket']=='113572',['Embarked']]='S'
all_df.head()
all_df['Age'].isnull().sum()
#Age has 263 null values. Salutation for missing values is Dr, Master, Miss, Mr, Mrs, Ms

#Lets first spilt the name variable in three columns each for salutation, family name and first name

all_df['Family_Name']=all_df['Name'].apply(lambda x: x.split(',')[0])

all_df['Name']=all_df['Name'].apply(lambda x: x.split(',')[1])
all_df['Salut']=all_df['Name'].apply(lambda x: x.split('.')[0])

all_df['Name']=all_df['Name'].apply(lambda x: x.split('.')[1])
all_df['Salut']=all_df['Salut'].apply(lambda x: x.strip())
#Age can be an important factor for determining outcome so for filling null values. I'll categorise them on the basis of age and sex

for i in range(len(all_df)):

    if ((all_df.iloc[i,all_df.columns.get_loc('Sex')]=='male') and (all_df.iloc[i,all_df.columns.get_loc('Age')]<=14)):

        all_df.iloc[i, all_df.columns.get_loc('Salut')] = 1

    if ((all_df.iloc[i,all_df.columns.get_loc('Sex')]=='female') and (all_df.iloc[i,all_df.columns.get_loc('Age')]<=14)):

        all_df.iloc[i, all_df.columns.get_loc('Salut')] = 2

    if ((all_df.iloc[i,all_df.columns.get_loc('Sex')]=='male') and (all_df.iloc[i,all_df.columns.get_loc('Age')]>14)):

        all_df.iloc[i, all_df.columns.get_loc('Salut')] = 3

    if ((all_df.iloc[i,all_df.columns.get_loc('Sex')]=='female') and (all_df.iloc[i,all_df.columns.get_loc('Age')]>14)):

        all_df.iloc[i, all_df.columns.get_loc('Salut')] = 4
all_df[all_df['Age'].isnull()]
all_df['Age']=all_df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

all_df.head()
# Master=all_df[all_df['Salut']==1]['Age'].mean()

# Miss=all_df[all_df['Salut']==2]['Age'].mean()

# Mr=all_df[all_df['Salut']==3]['Age'].mean()

# Mrs=all_df[all_df['Salut']==4]['Age'].mean()

# print(Master)

# print(Miss)

# print(Mr)

# print(Mrs)

# # There is not much difference in values of male and female, so for simplicity i'll fill 6 and 32.





for i in range(len(all_df)):

#     if all_df.iloc[i,all_df.columns.get_loc('Salut')] in ['Dr','Capt','Col','Don','Dona','Jonkheer','Lady','Major','Miss','Mlle','Mme','Mr','Mrs','Ms','Rev','Sir','the Countess']:

#         all_df.iloc[i,all_df.columns.get_loc('Age')]=32

#     if all_df.iloc[i,all_df.columns.get_loc('Salut')] in ['Master']:

#         all_df.iloc[i,all_df.columns.get_loc('Age')]=6

    if all_df.iloc[i,all_df.columns.get_loc('Salut')] in ['Dr','Capt','Col','Don','Jonkheer','Major','Mr','Rev','Sir']:

        all_df.iloc[i,all_df.columns.get_loc('Salut')]=3

    if all_df.iloc[i,all_df.columns.get_loc('Salut')] in ['Dona','Lady','Miss','Mlle','Mme','Mrs','Ms']:

        all_df.iloc[i,all_df.columns.get_loc('Salut')]=4

    if all_df.iloc[i,all_df.columns.get_loc('Salut')] in ['Master']:

        all_df.iloc[i,all_df.columns.get_loc('Salut')]=1

all_df.info()

#only cabin is left with null values but it has more than 1000 null values so we will remove this column

# we well also remove the columns which are not helpful in predicting the survival such as ticket no., Name, Family_name.
# all_df['Cabin']=all_df['Cabin'].fillna('U')

# all_df['Cabin']
# deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

# all_df['Deck'] = all_df['Cabin'].astype(str).str[0] 

# all_df['Deck'] = all_df['Deck'].str.capitalize()

# all_df['Deck'] = all_df['Deck'].map(deck)

# all_df['Deck'] = all_df['Deck'].fillna(0)

# all_df['Deck'] = all_df['Deck'].astype(int) 

# all_df['Deck']
all_df.drop(columns=['Cabin','Name','Family_Name','Ticket','Pclass'], inplace=True)

all_df.info()
#now we will map the data to make it categorical

cleanup_maps = {"Sex":     {"male": 1, "female": 2},

                "Embarked": {"S": 1, "Q": 2, "C": 3}}

all_df.replace(cleanup_maps, inplace=True)

all_df.head()
corr=all_df.corr()

sns.heatmap(corr)
# all_df['family_size']=all_df['SibSp']+all_df['Parch']+1

# all_df.head()
# all_df['is_alone']=0

# for i in range(len(all_df)):

#     if all_df.iloc[i,all_df.columns.get_loc('family_size')]>1:

#         all_df.iloc[i,all_df.columns.get_loc('is_alone')]=1

        

# all_df.head()
# all_df.drop(columns=['SibSp','Parch'], inplace=True)

# all_df.head()
all_df['Age'] = pd.qcut(all_df['Age'], 10)

all_df.head()
all_df['Fare'] = pd.qcut(all_df['Fare'], 13)

all_df.head()
non_numeric_features=['Age', 'Fare']

for feature in non_numeric_features:        

        all_df[feature] = LabelEncoder().fit_transform(all_df[feature])

        

all_df.head()
train_df=all_df.iloc[0:891,:]

test_df=all_df.iloc[891: ,:]

print(train_df.shape)

print(test_df.shape)
train_passengerId=train_df['PassengerId']

test_passengerId=test_df['PassengerId']

target=train_df['Survived']

train_df.drop(columns=['PassengerId','Survived'],inplace=True)

test_df.drop(columns=['PassengerId','Survived'],inplace=True)
test_df.info()
decision_tree=DecisionTreeClassifier(random_state=0, max_depth=6)

accuracy_decision_tree = cross_val_score(decision_tree, train_df, target, scoring='accuracy', cv = 10)

print("The accuracy from decision tree model is {0}".format(accuracy_decision_tree.mean()))



# # logistic_regression= LogisticRegression()

# # accuracy_logistic_regression = cross_val_score(logistic_regression, train_df, target, scoring='accuracy', cv = 10)

# # print("The accuracy from logistic regression model is {0}".format(accuracy_logistic_regression.mean()))



# # clf_svm= svm.SVC()

# # accuracy_svm = cross_val_score(clf_svm, train_df, target, scoring='accuracy', cv = 10)

# # print("The accuracy from svm model is {0}".format(accuracy_svm.mean()))



# # clf_rf = RandomForestClassifier(max_depth=6, random_state=0)

# # accuracy_random_forest = cross_val_score(clf_rf, train_df, target, scoring='accuracy', cv = 10)

# # print("The accuracy from random forest model is {0}".format(accuracy_random_forest.mean()))



# clf_gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

# accuracy_gradiet_boost = cross_val_score(clf_gbc, train_df, target, scoring='accuracy', cv = 10)

# print("The accuracy from gradient boost model is {0}".format(accuracy_gradiet_boost.mean()))



# clf_bagging = BaggingClassifier()

# accuracy_bagging = cross_val_score(clf_bagging, train_df, target, scoring='accuracy', cv = 10)

# print("The accuracy from bagging model is {0}".format(accuracy_bagging.mean()))



# clf_gnb = GaussianNB()

# accuracy_gaussian = cross_val_score(clf_gnb, train_df, target, scoring='accuracy', cv = 10)

# print("The accuracy from Gaussian model is {0}".format(accuracy_gaussian.mean()))



# clf_xgb = XGBClassifier()

# accuracy_xgb = cross_val_score(clf_xgb, train_df, target, scoring='accuracy', cv = 10)

# print("The accuracy from xgb model is {0}".format(accuracy_xgb.mean()))



decision_tree = DecisionTreeClassifier(random_state=0, max_depth=6)

decision_tree = decision_tree.fit(train_df, target)

pred=decision_tree.predict(test_df)

# print('BEFORE DT Parameters: ', decision_tree.get_params())
# param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini

#               #'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best

#               'max_depth': [2,4,6,8,10,None], #max depth tree can grow; default is none

#               #'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2

#               #'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1

#               #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all

#               'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation

#              }
# tune_model = model_selection.GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = 10)

# tune_model.fit(train_df, target)
# print('AFTER DT Parameters: ', tune_model.best_params_)
# pred=tune_model.predict(test_df)


# accuracy_tune_model = cross_val_score(tune_model, train_df, target, scoring='accuracy', cv = 10)

# print("The accuracy from tune_model is {0}".format(accuracy_tune_model.mean()))
data = {'PassengerId':test_passengerId, 'Survived':pred}

submission=pd.DataFrame(data)

submission['Survived']=submission['Survived'].apply(lambda x: int(x))

submission.head(25)
submission.to_csv('submission.csv', index=False)