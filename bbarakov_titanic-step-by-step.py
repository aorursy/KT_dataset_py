# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)
#train.head(30)
missing_age = train.loc[train['Age'].isnull()]
len(missing_age)
def survival_rate(df):
    return len(df.loc[df['Survived'] == 1]['Survived']) / len(df)

rates = np.array([survival_rate(missing_age), survival_rate(train)])
index = np.arange(2)
plt.bar(index, rates)
plt.xticks(index, ['Missing Age', 'Total'])
train.groupby(['Pclass', 'Sex']).apply(lambda df: df.Age.mean()).plot.bar()
#train.groupby(['Sex']).apply(lambda a: len(a) / len(train) * 100)
#plt.figure(figsize=(8,7))
#sns.countplot(data=train, x='Pclass', hue='Sex')
def fill_age(data):
    return (
        data
        .groupby(['Pclass', 'Sex'])
        .Age
        .transform(lambda d: d.fillna(d.mean()))
    )
#https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
title_dict = { 'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Misc': 4}

def set_title(data):
    data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    title_names = (data['Title'].value_counts() < 10)
    #return title_names
    data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    return data['Title'].map(lambda c: title_dict[str(c)])
    
#set_title(train)
missing_cabin = train.loc[train['Cabin'].isnull()]
missing_ticket = test.loc[test['Embarked'].isnull()]

len(missing_cabin), len(missing_ticket)
cabin_dict = { 'n': 0, 'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8 } 

def set_cabin_index(data):
    return (
        data
        .Cabin
        .map(lambda c: cabin_dict[str(c)[0]])
    )
def transform(df):
    df.Age = fill_age(df)
    df['age_1'] = df.apply(lambda d: 1 if d.Age <= 1 else 0, axis='columns')
    df['age_2'] = df.apply(lambda d: 1 if 1 < d.Age <= 10 else 0, axis='columns')
    df['age_3'] = df.apply(lambda d: 1 if 10 < d.Age <= 20 else 0, axis='columns')
    df['age_4'] = df.apply(lambda d: 1 if 20 < d.Age <= 30 else 0, axis='columns')
    df['age_5'] = df.apply(lambda d: 1 if 30 < d.Age <= 40 else 0, axis='columns')
    df['age_6'] = df.apply(lambda d: 1 if 40 < d.Age <= 50 else 0, axis='columns')
    df['age_7'] = df.apply(lambda d: 1 if d.Age > 50 else 0, axis='columns')
    
    #df.Sex = df.apply(lambda d: 1 if d.Sex == 'female' else 0, axis='columns')
    df['is_male'] = df.apply(lambda d: 1 if d.Sex == 'male' else 0, axis='columns')
    df['is_female'] = df.apply(lambda d: 1 if d.Sex == 'female' else 0, axis='columns')
    
    # added on the second iteration and achieved 0.80 accuracy with LogisticRegression
    df.Embarked = df.apply(lambda d: 0 if d.Embarked == 'S' else (1 if d.Embarked == 'C' else 2), axis='columns')
    df['embarked_s'] = df.apply(lambda d: 1 if d.Embarked == 0 else 0, axis='columns')
    df['embarked_c'] = df.apply(lambda d: 1 if d.Embarked == 1 else 0, axis='columns')
    df['embarked_q'] = df.apply(lambda d: 1 if d.Embarked == 2 else 0, axis='columns')
    
    # added on third iteration, slightly better
    df['hasBigFamily'] = df.apply(lambda d: 1 if d.SibSp + d.Parch >= 3 else 0, axis='columns')
    df['familySize'] = df.apply(lambda d: d.SibSp + d.Parch + 1, axis='columns')
    df['familyName'] = df['Name'].str.split(", ", expand=True)[0]
    #df['family'] = df.apply(lambda d: d.SibSp + d.Parch, axis='columns')
    
    df['cabinIndex'] = set_cabin_index(df)
    
    df['Title'] = set_title(df)
    df['is_mr'] = df.apply(lambda d: 1 if d.Title == 0 else 0, axis='columns')
    df['is_miss'] = df.apply(lambda d: 1 if d.Title == 1 else 0, axis='columns')
    df['is_mrs'] = df.apply(lambda d: 1 if d.Title == 2 else 0, axis='columns')
    df['is_master'] = df.apply(lambda d: 1 if d.Title == 3 else 0, axis='columns')
    df['is_misc'] = df.apply(lambda d: 1 if d.Title == 4 else 0, axis='columns')
    
    #https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
    df.Fare.fillna(df.Fare.mean(), inplace=True)
    df['fare_1'] = df.apply(lambda d: 1 if d.Fare <= 7.91 else 0, axis='columns')
    df['fare_2'] = df.apply(lambda d: 1 if 7.91 < d.Fare <= 14.454 else 0, axis='columns')
    df['fare_3'] = df.apply(lambda d: 1 if 14.454 < d.Fare <= 31 else 0, axis='columns')
    df['fare_4'] = df.apply(lambda d: 1 if d.Fare > 31 else 0, axis='columns')
        
    df['ticketTrunc'] = df['Ticket'].str[:-1] + str(df['Embarked']) + str(df['Fare'])
      
    return df
# https://www.kaggle.com/sateesh14siv/82-3-using-ticket-fare-pclass-and-embarked
def set_family_group(row, all_survived, all_died):
    if row['Age'] < 16 and row['Sex'] == 'male' and (row['familyName'] in all_survived):
        return 1
    if row['Sex'] == 'female':
        if row['familyName'] in all_died:
            return 0
        else:
            return 1
    else:
        return 0
    
def set_ticket_group(row, all_survived, all_died):
    if row['Age'] < 16 and row['Sex'] == 'male' and (row['ticketTrunc'] in all_survived):
        return 1
    if row['Sex'] == 'female':
        if row['ticketTrunc'] in all_died:
            return 0
        else:
            return 1
    else:
        return 0
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import train_test_split

data = pd.read_csv("../input/train.csv", index_col=0)
df = transform(data)
boys_and_fem = df[((df['Age'] < 16) & (df['Sex'] == 'male')) | (df['Sex'] == 'female')]
# by familyName
family_survival_list = boys_and_fem.groupby(['familyName'])['Survived'].mean().to_frame()

family_all_survived = family_survival_list[family_survival_list['Survived'] == 1].index
family_all_died = family_survival_list[family_survival_list['Survived'] == 0].index
df['familyGroup'] = df.apply( set_family_group, axis='columns', args=(family_all_survived, family_all_died) )

#by Ticket
ticket_survival_list = boys_and_fem.groupby(['ticketTrunc'])['Survived'].mean().to_frame()

ticket_all_survived = ticket_survival_list[ticket_survival_list['Survived'] == 1].index
ticket_all_died = ticket_survival_list[ticket_survival_list['Survived'] == 0].index
df['ticketGroup'] = df.apply( set_ticket_group, axis='columns', args=(ticket_all_survived, ticket_all_died) )

data = pd.read_csv("../input/test.csv", index_col=0)
test_df = transform(data)
test_df['familyGroup'] = test_df.apply( set_family_group, axis='columns', args=(family_all_survived, family_all_died) )
test_df['ticketGroup'] = df.apply( set_ticket_group, axis='columns', args=(ticket_all_survived, ticket_all_died) )

cols = ['Pclass',
        'SibSp',
        'Parch',
        #'cabinIndex',
        #'Fare',
        'fare_1', 'fare_2', 'fare_3', 'fare_4',
        'ticketGroup',
        #'hasBigFamily',
        #'familySize',
        'familyGroup',
        #'Sex',
        'is_male', 'is_female',
        #'Title',
        'is_mrs', 'is_mr', 'is_miss', 'is_master', 'is_misc',
        #'Age',
        'age_1', 'age_2', 'age_3', 'age_4', 'age_5', 'age_6', 'age_7',
        #'Embarked',
        #'embarked_s', 'embarked_c', 'embarked_q'
       ]

X = df[cols]
X_test = test_df[cols]
y = df.Survived
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

logreg = LogisticRegression()
logreg_mean = cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean().round(3)

svc = SVC(gamma='auto')
svc_mean = cross_val_score(svc, X, y, cv=10, scoring='accuracy').mean().round(3)

xgb = XGBClassifier()
xgb_mean = cross_val_score(xgb, X, y, cv=10, scoring='accuracy').mean().round(3)

#(0.827, 0.833, 0.823)
#(0.888, 0.891, 0.888)
logreg_mean, svc_mean, xgb_mean
xgb.fit(X, y)
res = xgb.predict(X_test)
#X_test[X_test['Age'] <= 1].index
pd.DataFrame(data=res, index=test_df.index, columns=['Survived']).to_csv('result.csv')
#k_range = list(range(1,31))
#k_scores = []

#for k in k_range:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    scores = cross_val_score(knn, X, y, cv=17, scoring='accuracy')
#    k_scores.append(scores.mean())

#plt.plot(k_range, k_scores)
#plt.xlabel('Value of K for KNN')
#plt.ylabel('Cross-Validated Accuracy')
#knn = KNeighborsClassifier(n_neighbors=9)
#cross_val_score(knn, X, y, cv=17, scoring='accuracy').mean()