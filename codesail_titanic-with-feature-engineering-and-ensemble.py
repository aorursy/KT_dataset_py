import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

IDtest = test["PassengerId"]
# concat all data to analyze

dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

dataset_origin = dataset

len(dataset)
train_len = len(train)
dataset.head()
dataset.info()
# null includes: Age, Cabin, Embarked

dataset.isnull().sum()
dataset.describe()
dataset[dataset['Survived'] == 1]
plt.figure(figsize=(12, 8))

plt.title('Pearson correlation of Features')

sns.heatmap(train.drop(labels=['PassengerId'], axis=1).corr(), cmap=plt.cm.RdBu, annot=True)
import re

def get_title(name):

    title_search = re.search(r'([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    

    return ''



dataset['Title'] = dataset['Name'].apply(get_title)
dataset['Title'].value_counts()
#replacing all titles with mr, mrs, miss, master, rare

def replace_titles(title):

    if title in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:

        return 'Rare'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title



dataset['Title'] = dataset['Title'].apply(replace_titles)
dataset['Title'].value_counts()
# Mapping titles

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

dataset['Title'] = dataset['Title'].map(title_mapping)
sns.factorplot(x='Title', y='Survived', data=dataset, kind='bar')
dataset['HasCabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
sns.factorplot(x='HasCabin', y='Survived', data=dataset, kind='bar')
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset['FamilySize'].value_counts()
sns.factorplot(x='FamilySize', y='Survived', data=dataset)
# FamilySize which equal to 1 is the same as "IsAlone"

# dataset['IsAlone'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)

# sns.factorplot(x='IsAlone', y='Survived', data=dataset, kind='bar')
dataset['Embarked'].value_counts()
dataset['Embarked'].fillna('S', inplace=True)
sns.factorplot(x='Embarked', y='Survived', data=dataset, kind='bar')
dataset["Sex"] = dataset["Sex"].map({"male": 1, "female": 0})
sns.factorplot(x='Sex', y='Survived', data=dataset, kind='bar')
plt.figure(figsize=(8, 6))

plt.title('Pearson correlation of Age')

sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(), cmap=plt.cm.RdBu, annot=True)
# Explore Age vs Survived

g = sns.FacetGrid(train, col='Survived')

g = g.map(sns.distplot, "Age")
# Explore Age vs Parch , Pclass and SibSP

g = sns.factorplot(y="Age",x="Pclass", data=dataset,kind="box")

g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="box")

g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="box")
# Filling missing value of Age 



## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)



age_med = dataset["Age"].median()

for i in index_NaN_age :

    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) &

                                (dataset['Parch'] == dataset.iloc[i]["Parch"]) &

                                (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        dataset['Age'].iloc[i] = age_pred

    else :

        dataset['Age'].iloc[i] = age_med

dataset[:len(train)][['Age', 'Survived']].groupby(by=['Age']).sum()
dataset['NameParenthesis'] = dataset['Name'].apply(lambda x: 1 if re.search(r'\(.*?\)', x) else 0)
dataset[:train_len][['NameParenthesis', 'Survived']].groupby(by=['NameParenthesis']).sum()
dataset[:train_len]['NameParenthesis'].value_counts()
sns.factorplot(x='NameParenthesis', y='Survived', data=dataset, kind='bar')
dataset['NameLength'] = dataset['Name'].apply(len)
name_subset = dataset[['NameLength', 'Survived']]
name_length_counts = dataset[:train_len]['NameLength'].value_counts().reset_index()
plt.figure(figsize=(16, 8))

sns.countplot(x='NameLength', hue='Survived', data=name_subset)
# sns.factorplot(x='NameLength', y='Survived', data=dataset, kind='bar')

# dataset[['NameLength', 'Survived']].groupby(['NameLength']).mean().plot()
# dataset[:train_len]['NameLength'].value_counts(sort=False).plot.bar()
# dataset[:train_len][['NameLength', 'NameParenthesis', 'Survived']].groupby(by=['NameLength'], as_index=False).mean()
plt.figure(figsize=(12, 8))

sns.heatmap(dataset.corr(), cmap=plt.cm.RdBu, annot=True)
name_set = dataset[:]
name_set['Name2'] = name_set['Name'].apply(lambda x: x.split('.')[1])



# #计算数量,然后合并数据集

Name2_sum = name_set['Name2'].value_counts().reset_index()

Name2_sum.columns=['Name2','Name2Sum']
name_set = pd.merge(left=name_set, right=Name2_sum, how='left',on='Name2')
# The Second name occured only once is a useless feature value, set to be 'one'

name_set.loc[name_set['Name2Sum'] == 1 , 'Name2New'] = 'one'

name_set.loc[name_set['Name2Sum'] > 1 , 'Name2New'] = name_set['Name2']
sns.countplot(x='Name2New', hue='Survived', data=name_set[name_set['Name2New'] != 'one'][name_set['Survived'] >= 0.7])
name_new_s = name_set[['Name2New', 'Survived']].groupby(['Name2New']).mean() 
name_new_s[name_new_s['Survived'] >= 0.7]
Name2_sum.columns=['Name2New', 'Name2Sum']
pd.merge(left=name_new_s[name_new_s['Survived'] > 0.5], right=Name2_sum, how='left', on='Name2New')
dataset = pd.concat([dataset, name_set.loc[:, ['Name2New']]], axis=1)
dataset['TicketLetter'] = dataset['Ticket'].apply(lambda x: 0 if re.search('^\d+$', x) else 1)
dataset[:train_len][['TicketLetter', 'Pclass']].groupby(['Pclass']).mean()
sns.countplot(x='Pclass', hue='TicketLetter', data=dataset[:train_len])
dataset[:train_len][['TicketLetter', 'Survived']].groupby(['TicketLetter']).mean()
sns.countplot(x='TicketLetter', hue='Survived', data=dataset[:train_len])
plt.figure(figsize=(12, 8))

sns.heatmap(dataset.corr(), cmap=plt.cm.RdBu, annot=True)
dataset.head(3)
#dataset = dataset.drop(labels=['PassengerId', 'Name', 'Cabin', 'Parch', 'SibSp', 'Ticket'], axis=1)
dataset = dataset.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)
dataset.head()
# Mapping Fare

dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

dataset['Fare'] = dataset['Fare'].astype(int)



# Mapping Age

dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

dataset['Age'] = dataset['Age'].astype(int)



# Mapping Embarked

dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# Create new feature of family size

dataset['Single'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallFamily'] = dataset['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedFamily'] = dataset['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeFamily'] = dataset['FamilySize'].map(lambda s: 1 if s >= 5 else 0)

del dataset['FamilySize']
plt.figure(figsize=(12, 8))

sns.heatmap(dataset.corr(), cmap=plt.cm.RdBu, annot=True)
# one-hot values Title and Embarked 

dataset = pd.get_dummies(dataset, columns = ["Age"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"])

dataset = pd.get_dummies(dataset, columns = ["Fare"])

# dataset = pd.get_dummies(dataset, columns = ["Parch", 'SibSp'])

dataset = pd.get_dummies(dataset, columns = ["Pclass"])

dataset = pd.get_dummies(dataset, columns = ["Title"])
# dataset = pd.get_dummies(dataset, columns = ["NameLength"])

dataset = pd.get_dummies(dataset, columns = ["Name2New"])
dataset.head()
#### Feature Correlation List
plt.figure(figsize=(24, 6))

dataset.corr()['Survived'].sort_values()[-40:]
## Separate train dataset and test dataset

train = dataset[:len(train)]

test = dataset[len(train):]

test.drop(labels=["Survived"],axis = 1,inplace=True)
## Separate train features and label 

train["Survived"] = train["Survived"].astype(int)

X_train = train.drop(labels = ["Survived"],axis = 1)

y_train = train["Survived"]
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier, \

                              ExtraTreesClassifier,AdaBoostClassifier,\

                              BaggingClassifier, VotingClassifier)

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from pprint import pformat
random_state = 2
cv = StratifiedShuffleSplit(n_splits=10,)
clf_list = [

    GradientBoostingClassifier(),

    RandomForestClassifier(),

    ExtraTreesClassifier(),

    AdaBoostClassifier(),

    BaggingClassifier(),

    DecisionTreeClassifier(),

    SVC(),

    KNeighborsClassifier(),

    LogisticRegression(),

    GaussianNB(),

    LGBMClassifier(),

    XGBClassifier(),

]

accuracy_dict = {}

for clf in clf_list:

    acc = cross_val_score(clf, X_train, y=y_train, cv=cv, scoring = "accuracy")    

    accuracy_dict[clf.__class__.__name__] = [acc.min(), acc.mean(), acc.max()]
accuracy_df = pd.DataFrame(accuracy_dict).transpose()

accuracy_df
accuracy_df.plot(kind='bar',rot=60)
gbm = LGBMClassifier(num_leaves=20,

                        learning_rate=0.5,

                        n_estimators=100)

gbm.fit(X_train, y_train,

        eval_metric='l1')

print('Feature importances:', list(gbm.feature_importances_))
from sklearn.metrics import accuracy_score

predictions = gbm.predict(X_train)

accuracy_score(predictions, y_train)
cv = StratifiedKFold(n_splits=10)
lgbm = LGBMClassifier(num_leaves=9, learning_rate=0.01, n_estimators=300)

print(cross_val_score(lgbm, X_train, y_train, cv=cv))

# print(cross_validate(lgbm, X_train, y_train, cv=cv))
lgbm.fit(X_train, y_train)
# xgbt = XGBClassifier(learning_rate=0.03, n_estimators=300)

# cross_val_score(xgbt, X_train, y_train, cv=cv)



# xgbt.fit(X_train, y_train)
feats = {}

for x, y in zip(dataset.columns, lgbm.feature_importances_):

   feats[x] = y

feats_df = pd.DataFrame(list(feats.items()), columns=['feat', 'importance'])
feats_df.sort_values(by=['importance'], ascending=False)[:40]
test_Survived = pd.Series(lgbm.predict(test), name="Survived")



results = pd.concat([IDtest, test_Survived],axis=1)



results.to_csv("titanic_with_ensemble.csv",index=False)