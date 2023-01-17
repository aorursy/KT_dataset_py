import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir('../input'))
train_data= pd.read_csv('../input/train.csv')

print(train_data.columns)

train_data.describe()
train_data.info()
train_data['Died'] = 1-train_data['Survived']

train_data.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar', stacked=True, figsize=(10,6));
fig, axis = plt.subplots(1,2,figsize=(15,8))

sns.barplot(x="Embarked", y="Survived", hue="Sex", ax=axis[(0)], data=train_data);

sns.barplot(x="Pclass", y="Survived", hue="Sex", ax=axis[(1)], data=train_data);

plt.figure(figsize=(10,5))

sns.barplot(x="Parch", y="Survived", hue="Sex", data=train_data);
plt.figure(figsize=(10,6))

sns.violinplot(x='Sex',y='Age',hue='Survived', data=train_data, split=True);
plt.figure(figsize=(15,10))

plt.hist([train_data[train_data['Survived'] == 1]['Fare'], train_data[train_data['Died'] == 1]['Fare']],

        stacked=True, color=['g','r'], bins=70, label = ['Survived','Died'])

plt.xlabel('Fare')

plt.ylabel('Number of passnegers')

plt.legend()

plt.grid()
plt.figure(figsize=(25,10))

ax=plt.subplot()



ax.scatter(train_data[train_data['Survived'] == 1]['Age'], train_data[train_data['Survived'] == 1]['Fare'],

          c='green', s=train_data[train_data['Survived'] == 1]['Fare'])

ax.scatter(train_data[train_data['Died'] == 1]['Age'], train_data[train_data['Died'] == 1]['Fare'],

          c='red', s=train_data[train_data['Died'] == 1]['Fare']);

plt.xlabel('Age')

plt.ylabel('Fare');
ax=plt.subplot()

ax.set_ylabel('Average fare')

train_data.groupby('Pclass').mean()['Fare'].plot(kind='bar', ax=ax, figsize=(10,6) );

#the line above is the same as :

#train_data.groupby('Pclass').agg({'Fare':'mean'}).plot(kind='bar', ax=ax)
X_train = train_data.drop(['Survived','Died'],axis=1)

y_train = train_data['Survived']

X_test = pd.read_csv('../input/test.csv')

df_combined = X_train.append(X_test)

print(X_train.shape[1])

print(df_combined.shape[1])
def process_family(df):

    # introducing a new feature : the size of families (including the passenger)

    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

    

    # introducing other features based on the family size

    df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

    df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)    

    return df
df_combined = process_family(df_combined)

df_combined.head()
print(df_combined.Embarked.describe())

df_combined.loc[df_combined.Embarked.isna()]
def process_embarked(df):

    # two missing embarked values - filling them with the most frequent one in the train  set(S)

    df.Embarked.fillna('S', inplace=True)

    # dummy encoding 

    df_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')

    df = pd.concat([df, df_dummies], axis=1)

    df.drop('Embarked', axis=1, inplace=True)

#     status('embarked')

    return df
df_combined = process_embarked(df_combined)

df_combined.head()
def process_cabin(df):

    # replacing missing cabins with U (for Uknown)

    df.Cabin.fillna('U', inplace=True)

    

    # mapping each Cabin value with the cabin letter

    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])

    

    # dummy encoding ...

    cabin_dummies = pd.get_dummies(df['Cabin'], prefix='Cabin')    

    df = pd.concat([df, cabin_dummies], axis=1)



    df.drop('Cabin', axis=1, inplace=True)

    return df
df_combined = process_cabin(df_combined)

df_combined.head()
titles = set()

for name in df_combined['Name']:

    titles.add(name.split(',')[1].split('.')[0].strip())

titles
Title_Dictionary = {

     'Capt':'Officier',

     'Col':'Officier',

     'Don':'Royalty',

     'Dona':'Royalty',

     'Dr':'Officier',

     'Jonkheer':'Royalty',

     'Lady':'Royalty',

     'Major':'Officier',

     'Master':'Master',

     'Miss':'Miss',

     'Mlle':'Miss',

     'Mme':'Mrs',

     'Mr':'Mr',

     'Mrs':'Mrs',

     'Ms':'Mrs',

     'Rev':'Officier',

     'Sir':'Royalty',

     'the Countess':'Royalty'   

}

def passenger_title(df):

    df['Title'] = df['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())

    df['Title'] = df['Title'].apply( lambda x : Title_Dictionary[x])

    return df
df_combined = passenger_title(df_combined)

df_combined.head()
grouped_train = df_combined.groupby(['Sex','Pclass','Title'])

grouped_median_train = grouped_train.median()

grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

grouped_median_train.head()
df_combined.groupby(['Sex','Pclass','Title']).agg({'Age':'median'}).reset_index().head()
def fill_age(row):

    condition = (

        (grouped_median_train['Sex'] == row['Sex']) & 

        (grouped_median_train['Title'] == row['Title']) & 

        (grouped_median_train['Pclass'] == row['Pclass'])

    ) 

    return grouped_median_train[condition]['Age'].values[0]



def process_age(df):

    # a function that fills the missing values of the Age variable

    df['Age'] = df.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

    return df
df_combined = process_age(df_combined)

df_combined.head()
def process_name(df):

    #removing the name column since we have the title column

    df.drop('Name', axis=1, inplace=True)

    

    #dummification Title column

    titles_dummies = pd.get_dummies(df['Title'], prefix='Title')

    df = pd.concat([df, titles_dummies], axis=1)

    

    #removing the title column since we have its dummies

    df.drop('Title', axis=1, inplace=True)

    return df
df_combined = process_name(df_combined)

df_combined.head()
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

df_combined['Sex'] = enc.fit_transform(df_combined['Sex'])

df_combined.head()
enc = LabelEncoder()

df_combined['Ticket'] = enc.fit_transform(df_combined['Ticket'])

df_combined.head()
train_data.shape
X_train = df_combined[:891]

X_test = df_combined[891:]

print(X_train.shape)

print(X_test.shape)
def split_vals(a,n): return a[:n], a[n:]

valid_count =60

n_trn = len(X_train)-valid_count

X_train1, X_valid1 = split_vals(X_train, n_trn)

y_train1, y_valid1 = split_vals(y_train, n_trn)
X_train1.shape,y_train1.shape,X_valid1.shape,y_valid1.shape
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



rfc = RandomForestClassifier(n_estimators=180,

                             min_samples_leaf=3,

                             max_features=0.5,

                             n_jobs=-1)

rfc.fit(X_train1,y_train1)

rfc.score(X_train1,y_train1)
y_predict=rfc.predict(X_valid1)

from sklearn.metrics import accuracy_score

accuracy_score(y_valid1,y_predict)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_valid1,y_predict))
print(confusion_matrix(y_valid1,y_predict))
!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887

!apt update && apt install -y libsm6 libxext6
from fastai.imports import *

from fastai.structured import *

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,GradientBoostingClassifier

from IPython.display import display

from sklearn import metrics

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

import seaborn as sns

import pylab as plot

#Feature importance

fi = rf_feat_importance(rfc, X_train1); fi[:10]
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(fi[:30]);
# Keeping only the variables which are significant for the model(>0.01)

to_keep = fi[fi.imp>0.01].cols; len(to_keep)

to_keep
#Now training the model on the entire data with only the important features.

X_train = X_train[to_keep]

X_train
rfc = RandomForestClassifier(n_estimators=200,min_samples_leaf=3,max_features=0.5,n_jobs=-1)

rfc.fit(X_train,y_train)

rfc.score(X_train,y_train)
X_test = X_test[to_keep]

X_test.isna().sum()

X_test.Fare.fillna(200, inplace=True)

output=rfc.predict(X_test)
output.size
data_test = pd.read_csv('../input/test.csv')

df_output = pd.DataFrame()

df_output['PassengerId'] = data_test['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('submission2', index=False)
df_output.head()