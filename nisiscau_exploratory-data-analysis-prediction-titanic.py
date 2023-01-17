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
import math

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.columns
# 1 - Data Types

train.info()
# 2 - Missing Values

display(train.isna().sum()) #absolute values

train.isna().sum() / len(train) * 100 #as a proportion of the whole column (%)
# 3 - Values Distribution

train[ list(train.columns) [1:] ].describe() #not taking into account PassengerId
from string import ascii_uppercase



train['Floor_Num'] = train['Cabin'].apply(lambda x: str(x)[0] if x != np.nan else np.nan)



for i in range(len(train['Floor_Num'])):

    if train.loc[i,'Floor_Num'] =='n' or train.loc[i,'Floor_Num'] =='T':

        train.loc[i,'Floor_Num'] = 'None'



train['Floor_Num'] = train['Floor_Num'].astype('category')



train['Floor_Num']
train['Floor_Num'].unique()
sns.set_context('notebook')



_ = sns.violinplot(data = train, x = 'Floor_Num', y = 'Survived')
print('Mean survival rate for category A : {:.2%}'.format(train[train['Floor_Num'] == 'A']['Survived'].mean()))

print('Mean survival rate for category B : {:.2%}'.format(train[train['Floor_Num'] == 'B']['Survived'].mean()))

print('Mean survival rate for category C : {:.2%}'.format(train[train['Floor_Num'] == 'C']['Survived'].mean()))

print('Mean survival rate for category D : {:.2%}'.format(train[train['Floor_Num'] == 'D']['Survived'].mean()))

print('Mean survival rate for category E : {:.2%}'.format(train[train['Floor_Num'] == 'E']['Survived'].mean()))

print('Mean survival rate for category F : {:.2%}'.format(train[train['Floor_Num'] == 'F']['Survived'].mean()))

print('Mean survival rate for category G : {:.2%}'.format(train[train['Floor_Num'] == 'G']['Survived'].mean()))

print('Mean survival rate for category T : {:.2%}'.format(train[train['Floor_Num'] == 'T']['Survived'].mean()))
train['Name'].apply(lambda x: x.split(' ')[1]).unique()
titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Don.', 'Rev.',

       'Dr.','Mme.', 'Ms.', 'Major.', 'Mlle.', 'Col.', 'Capt.']



def find_title(s): #this way, we can find the title even if it is not located at the exact same place!

    for t in titles:

        if t in s:

            return t



train['Title'] = train['Name'].apply(find_title)
display(train.Age) #see what the column looks like before

m = []

for t in titles:

    m.append(train[train.Title == t].Age.mean())

mean_age_by_title = dict(zip(titles,m))



for i in range(len(train)):

    if train.loc[i,'Title'] is None:

        pass

    else:

        train.Age.fillna(mean_age_by_title[train.loc[i,'Title']], inplace = True)

train.Age
train.Age.isna().sum()
def cat_age(a):

    return min(int(a- int(a)%10), 70)

train['Age Tranche'] = train['Age'].apply(cat_age).astype('category')
train.head(20)
_ = sns.barplot(data = train, x = 'Sex', y = 'Survived')
_ = sns.barplot(data = train, x = 'Age Tranche', y = 'Survived')
_ = sns.barplot(data = train, x = 'SibSp', y = 'Survived')
_ = sns.barplot(data = train, x = 'Parch', y = 'Survived')
_ = sns.barplot(data = train, x = 'Pclass', y = 'Survived')
_ = sns.barplot(data = train, x = 'Floor_Num', y = 'Survived')
def fam_cat(n):

    if n>=4:

        return '4+'

    else:

        return str(n)

train['fam_cat'] = pd.DataFrame([train['SibSp'], train['Parch']]).max().apply(fam_cat).astype('category')
train.columns
travel_class = pd.get_dummies(train.Pclass, prefix = 'class',prefix_sep='_')

sex = pd.get_dummies(train.Sex, prefix = 'sex',prefix_sep='_')

age_cat = pd.get_dummies(train['Age Tranche'], prefix = 'agecat', prefix_sep='_')

fam_cat = pd.get_dummies(train['fam_cat'], prefix = 'famcat', prefix_sep='_')

floor_num = pd.get_dummies(train['Floor_Num'], prefix = 'floornum', prefix_sep='_')

embarkation = pd.get_dummies(train['Embarked'], prefix = 'embarkation', prefix_sep='_')



train_final = pd.concat([train[['PassengerId','Survived']], travel_class, sex, age_cat, fam_cat,floor_num, embarkation], axis = 1)
train_final
#splitting

from sklearn.model_selection import train_test_split



#models

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier





#hyperparameter tuning

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
X = train_final[list(train_final.columns)[2:]].values

y = train_final['Survived'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.25)
models = [LogisticRegression(), DecisionTreeClassifier(), GaussianNB(), SVC(), RandomForestClassifier(), BaggingClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()]



accuracy_scores = []

f1_scores = []

roc_auc_scores = []



for m in models:

    m.fit(X_train, y_train)

    y_prediction = m.predict(X_test)

    accuracy_scores.append(accuracy_score(y_test, y_prediction))

    f1_scores.append(f1_score(y_test, y_prediction))

    roc_auc_scores.append(roc_auc_score(y_test, y_prediction))



models_names = ['Logistic Regression', 'Decision Tree Classifier', 'Gaussian Naive Bayes', 'SVC', 'Random Forest Classifier', 'Bagging Classifier', 'AdaBoost Classifier', 'Gradient Boosting Classifier']



models_performances = zip(accuracy_scores,f1_scores,roc_auc_scores)



models_names_and_perfs = dict(zip(models_names,models_performances))



models_names_and_perfs
svc = SVC()

rfc = RandomForestClassifier()

gb = GradientBoostingClassifier()



vote = VotingClassifier(estimators = [('svc', svc),('rfc', rfc),('gb', gb)])



vote = vote.fit(X_train, y_train)

y_pred_vote = vote.predict(X_test)



vote_acc_score = accuracy_score(y_test, y_pred_vote)

vote_f1_score = f1_score(y_test, y_pred_vote)



vote_acc_score, vote_f1_score
# params_rf = {'max_features': ['auto', 'sqrt'],

#              'min_samples_leaf': [1, 2, 4],

#              'min_samples_split': [2, 5, 10,15],

#              'n_estimators': [100,200,500]}



# rf = RandomForestClassifier(n_jobs = -1)



# clf = GridSearchCV(rf, params_rf)

# clf.fit(X_train, y_train)



# display(clf.best_params_)



# y_true, y_pred = y_test, clf.predict(X_test)

# print(classification_report(y_true, y_pred))
best_params = {'max_features': 'auto',

               'min_samples_leaf': 2,

               'min_samples_split': 5,

               'n_estimators': 200}
best_rfc = RandomForestClassifier(max_features = 'auto', min_samples_leaf = 2, min_samples_split = 5, n_estimators = 200)

best_rfc.fit(X_train, y_train)

y_pred_best_rfc = best_rfc.predict(X_test)



best_rfc_acc_score = accuracy_score(y_test, y_pred_best_rfc)

best_rfc_f1_score = f1_score(y_test, y_pred_vote)



best_rfc_acc_score, best_rfc_f1_score
# params_gbc = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 

#               'n_estimators':[100,250,500,750,1000,1250,1500,1750]}



# gbc = GradientBoostingClassifier()



# clf = GridSearchCV(gbc, params_gbc)

# clf.fit(X_train, y_train)



# display(clf.best_params_)



# y_true, y_pred = y_test, clf.predict(X_test)

# print(classification_report(y_true, y_pred))
best_params = {'learning_rate': 0.01, 'n_estimators': 1000}
best_gbc = GradientBoostingClassifier(learning_rate = 0.01, n_estimators = 1000)

best_gbc.fit(X_train, y_train)

y_pred_best_gbc = best_gbc.predict(X_test)



best_gbc_acc_score = accuracy_score(y_test, y_pred_best_gbc)

best_gbc_f1_score = f1_score(y_test, y_pred_vote)



best_gbc_acc_score, best_gbc_f1_score
gbc_vote = GradientBoostingClassifier(learning_rate = 0.01, n_estimators = 1000)

rfc_vote = RandomForestClassifier(max_features = 'auto', min_samples_leaf = 2, min_samples_split = 5, n_estimators = 200)

svm_vote = SVC()



best_vote = VotingClassifier(estimators = [('svm', svm_vote),('rfc', rfc_vote),('gbc', gbc_vote)])



best_vote = best_vote.fit(X_train, y_train)

y_pred_best_vote = best_vote.predict(X_test)



best_vote_acc_score = accuracy_score(y_test, y_pred_best_vote)

best_vote_f1_score = f1_score(y_test, y_pred_best_vote)



best_vote_acc_score, best_vote_f1_score
test.head()
test['Floor_Num'] = test['Cabin'].apply(lambda x: str(x)[0] if x != np.nan else np.nan)



m = []

for i in range(len(test['Floor_Num'])):

    if test.loc[i,'Floor_Num'] =='n' or test.loc[i,'Floor_Num'] =='T':

        test.loc[i,'Floor_Num'] = 'None'



test['Floor_Num'] = test['Floor_Num'].astype('category')



test['Title'] = test['Name'].apply(find_title)



for t in titles:

    m.append(test[test.Title == t].Age.mean())

mean_age_by_title = dict(zip(titles,m))



for i in range(len(test)):

    if test.loc[i,'Title'] is None:

        pass

    else:

        test.Age.fillna(mean_age_by_title[test.loc[i,'Title']], inplace = True)



def fam_cat(n):

    if n>=4:

        return '4+'

    else:

        return str(n)

    

test['Age Tranche'] = test['Age'].apply(cat_age).astype('category')

test['fam_cat'] = test['SibSp'].clip(lower = test['Parch']).apply(fam_cat).astype('category')



test_travel_class = pd.get_dummies(test.Pclass, prefix = 'class',prefix_sep='_')

test_sex = pd.get_dummies(test.Sex, prefix = 'sex',prefix_sep='_')

test_age_cat = pd.get_dummies(test['Age Tranche'], prefix = 'agecat', prefix_sep='_')

test_fam_cat = pd.get_dummies(test['fam_cat'], prefix = 'famcat', prefix_sep='_')

test_floor_num = pd.get_dummies(test['Floor_Num'], prefix = 'floornum', prefix_sep='_')

test_embarkation = pd.get_dummies(test['Embarked'], prefix = 'embarkation', prefix_sep='_')



test_final = pd.concat([test['PassengerId'], test_travel_class, test_sex, test_age_cat, test_fam_cat,test_floor_num, test_embarkation], axis = 1)
X = test_final[list(test_final.columns)[1:]].values



y_final_pred = best_gbc.predict(X)



output = pd.DataFrame(data = y_final_pred, index = test_final.PassengerId)



output.columns = ['Survived']



output.to_csv('Titanic_test_prediction.csv')