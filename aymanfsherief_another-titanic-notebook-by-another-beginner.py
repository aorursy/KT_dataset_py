# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
titanic_data = pd.read_csv('../input/train.csv')
titanic_data.head()
titanic_data[pd.isna(titanic_data.Age)]
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

def describe_series(srs):
    print("describing: {}".format(srs.name))
    unique_vals = srs.unique()
    print("{} unique values".format(unique_vals.size))
    plt.figure()
    plt.hist(srs.dropna().values)
    plt.title('{}\nvalue percentages'.format(srs.name))
   
    srs_surv = srs[titanic_data.Survived == 1].dropna().value_counts()
    srs_persh = srs[titanic_data.Survived == 0].dropna().value_counts()
    [srs_surv.set_value(exc_val, 0) for exc_val in srs_persh.index if exc_val not in srs_surv.index]
   
    plt.figure()
    plt.bar(srs_surv.index, srs_surv, alpha=.5)
    plt.bar(srs_persh.index, srs_persh, alpha=.5)
    plt.legend(['Survived', 'Perished'])
        
    
        
    
titanic_data[['Sex', 'Cabin', 'Ticket', 'Embarked']]
titanic_data['Cabin_initial'] = titanic_data.Cabin.apply(lambda cab: cab[0] if not pd.isna(cab) else '0')
describe_series(titanic_data.Cabin_initial)
import re
titanic_data['Cabin_num'] = titanic_data.Cabin.apply(lambda cab: re.search(r'\d+\s+',cab).group(0).strip() if not pd.isna(cab) and re.search(r'\d+\s+',cab)
                                                     else '0')
describe_series(titanic_data.Cabin_num)
titanic_data.Embarked = titanic_data.Embarked.fillna('0')
describe_series(titanic_data.Embarked)
for col in titanic_data:
    if pd.isna(titanic_data[col]).any():
        print(col)
describe_series(titanic_data.Age)
titanic_data.Age = titanic_data.Age.fillna(titanic_data.Age.dropna().median())
train_data = titanic_data.drop(['Cabin', 'Survived', 'Name', 'Ticket', 'PassengerId'], axis = 1)
pd.isna(train_data.Embarked).any()
from sklearn.preprocessing import LabelEncoder
sex_le = LabelEncoder()
train_data.Sex = sex_le.fit_transform(train_data.Sex)

cabin_le = LabelEncoder()
train_data.Cabin_initial = cabin_le.fit_transform(train_data.Cabin_initial)
embark_le = LabelEncoder()
train_data.Embarked = embark_le.fit_transform(train_data.Embarked)
train_data.Cabin_num = train_data.Cabin_num.apply(float)
train_data[pd.isna(train_data.Age)]
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import RidgeClassifier

lr = LogisticRegression()
scoring = ['precision_macro', 'recall_macro']
svm = SVC(probability=True)
rfc = RandomForestClassifier(bootstrap=True, n_estimators=100)
clf = VotingClassifier([('rfc', rfc), ('lr', lr), ('svm', svm)], voting='soft')
scores = cross_validate(clf, train_data, titanic_data.Survived, scoring=scoring,
                        cv=5, return_train_score=False)
print(sorted(scores.keys()))

                   

for metr in scores:
    print(metr)
    print(scores[metr].mean())
test = pd.read_csv('../input/test.csv')
for col in test_prepped:
    if(pd.isna(test_prepped[col]).any()):
        print(col)
import re
def prep_data(data):
    data.Age = data
    data['Cabin_initial'] = data.Cabin.apply(lambda cab: cab[0] if not pd.isna(cab) else '0')
    
    data['Cabin_num'] = data.Cabin.apply(lambda cab: re.search(r'\d+\s+',cab).group(0).strip()
                                                         if not pd.isna(cab) and re.search(r'\d+\s+',cab) else '0')
    data.Embarked = data.Embarked.fillna('0')
    data.Age = data.Age.fillna(data.Age.dropna().median())
    data.Fare = data.Fare.fillna(data.Fare.dropna().mean())
    data.Sex = sex_le.transform(data.Sex)
    data.Cabin_initial = cabin_le.transform(data.Cabin_initial)
    data.Embarked = embark_le.transform(data.Embarked)
    data.Cabin_num = train_data.Cabin_num.apply(float)
    return data.copy()
test_prepped = prep_data(test)
test_prepped = test_prepped[list(train_data.columns)+['PassengerId']]
clf.fit(train_data, titanic_data.Survived)
test_prepped['Survived'] = clf.predict(test_prepped.drop('PassengerId', axis=1))
sub = test_prepped[['PassengerId', 'Survived']]
sub = sub.set_index('PassengerId')
sub.to_csv('titanic_sub.csv')
