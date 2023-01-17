import pandas as pd

import numpy as np

import sys

from sklearn.externals import joblib

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



%matplotlib inline  
train = pd.read_csv('../input/train.csv', dtype={'Ticket' : 'U', 'Cabin' : 'U'})

train.insert(0, 'Filename', 'train.csv')
test = pd.read_csv('../input/test.csv', dtype={'Ticket' : 'U', 'Cabin' : 'U'})

test.insert(1, 'Survived', np.nan)

test.insert(0, 'Filename', 'test.csv')
ds = pd.concat([train, test])

ds.head()
for col in ds:

    print('Column {:s} ({}) - missing values : {:d}'.format(ds[col].name, ds[col].dtype, ds[col].isnull().sum()))
dummies = pd.get_dummies(ds["Pclass"], prefix='is_class')

ds = pd.concat([ds, dummies], axis=1)
ds.insert(4,'Title', np.nan)

ds.insert(6,'Last Name', np.nan)
ds["Last Name"], ds["Name"] = ds["Name"].str.split(',',1).str

ds["Title"], ds["Name"] = ds["Name"].str.split('.',1).str



# Trim whitespaces

for col in ["Last Name", "Title", "Name"]:

    ds[col] = ds[col].str.strip()



# Show splitted values

ds[['Title','Name','Last Name']].head()
ds["Title"].value_counts()
ds['agg_title'] = ds['Title'].copy()



ds['agg_title'].replace(['Mlle','Ms'], 'Miss', inplace=True)

ds['agg_title'].replace(['Mme','Dona'], 'Mrs', inplace=True)

ds['agg_title'].replace(['Don'], 'Mr', inplace=True)

ds['agg_title'].replace(['Col','Major','Capt'], 'Military', inplace=True)

ds['agg_title'].replace(['Lady','the Countess','Jonkheer','Sir'], 'Noble', inplace=True)



# Checkng values

ds["agg_title"].value_counts()
dummies = pd.get_dummies(ds["agg_title"], prefix='is')

ds = pd.concat([ds, dummies], axis=1)
dummies = pd.get_dummies(ds["Sex"], prefix='is')

ds = pd.concat([ds, dummies], axis=1)
ds['age_discrete_cut'] = pd.cut(ds.Age, 7)

# Generating dummies

dummies = pd.get_dummies(ds["age_discrete_cut"], prefix='age_discrete_cut')

ds = pd.concat([ds, dummies], axis=1)

ds.head()
ds[ds['Fare'].isnull()][['PassengerId','Pclass','SibSp','Parch','Embarked','Ticket']]
# Group by ticket

grouped = ds[(ds['Pclass'] == 3) & (ds['SibSp'] == 0)  & (ds['Parch'] == 0) & (ds['Embarked'] == 'S')].groupby('Ticket').size()



# Only there where single ticket

missinFare = ds[ds['Ticket'].isin(grouped[grouped == 1].keys())]['Fare'].median()

missinFare
ds.loc[ds['PassengerId'] == 1044, 'Fare'] = missinFare

ds[ds['PassengerId'] == 1044]['Fare']
ds[ds['Embarked'].isnull()][['Pclass','Ticket', 'Fare']]
# Group by ticket

grouped = ds[(ds['Fare'] >= 79) & (ds['Fare'] <= 83)  & (ds['Pclass'] == 1)].groupby('Ticket').size()



ds[ds['Ticket'].isin(grouped[grouped == 2].keys())][['Embarked','Fare']]
ds.loc[ds['Ticket'] == '113572', 'Embarked'] = 'C'
# Replace values to full one

ds['Embarked'].replace(['C','Q','S'], ['Cherbourg','Queenstown','Southampton'], inplace=True)

# Generating dummies

dummies = pd.get_dummies(ds["Embarked"], prefix='embarked')

ds = pd.concat([ds, dummies], axis=1)
non_age_columns = ['SibSp','Parch','Fare','is_class_1','is_class_2','is_class_3','is_Dr','is_Master','is_Military','is_Miss','is_Mr','is_Mrs','is_Noble','is_Rev','is_female','is_male','embarked_Cherbourg','embarked_Queenstown','embarked_Southampton','age_discrete_cut_(11.574, 22.979]','age_discrete_cut_(22.979, 34.383]','age_discrete_cut_(34.383, 45.787]','age_discrete_cut_(45.787, 57.191]','age_discrete_cut_(57.191, 68.596]','age_discrete_cut_(68.596, 80]']

train = ds[ds.age_discrete_cut.notnull()]
rf = RandomForestClassifier(100)

rf.fit(train[non_age_columns], train.age_discrete_cut)

ds.loc[ds.age_discrete_cut.isnull(), 'age_discrete_cut'] = rf.predict(ds[ds.age_discrete_cut.isnull()][non_age_columns])

cross_val_score(rf, train[non_age_columns], train.age_discrete_cut)
ds['TicketGroup'] = ds.Ticket.groupby(ds.Ticket).transform('count')

dummies = pd.get_dummies(ds["TicketGroup"], prefix='ticket_group')

ds = pd.concat([ds, dummies], axis=1)
g = ds.groupby(['Ticket','Survived']).size().unstack(fill_value =0).reset_index()

g.columns = ['Ticket', 'ticket_group_survived_0', 'ticket_group_survived_1']



ds = pd.merge(ds, g, how='left')



ds.loc[ds.Survived == 0.0, 'ticket_group_survived_0'] = ds[ds.Survived == 0.0]['ticket_group_survived_0'].sub(1)

ds.loc[ds.Survived == 1.0, 'ticket_group_survived_1'] = ds[ds.Survived == 1.0]['ticket_group_survived_1'].sub(1)
ds['ticket_group_ratio_survived_0'] = ds.ticket_group_survived_0 / (ds.TicketGroup.sub(1))

ds['ticket_group_ratio_survived_1'] = ds.ticket_group_survived_1 / (ds.TicketGroup.sub(1))



ds.ticket_group_ratio_survived_0 = ds.ticket_group_ratio_survived_0.fillna(0)

ds.ticket_group_ratio_survived_1 = ds.ticket_group_ratio_survived_1.fillna(0)
categorical_columns = ['Filename','PassengerId','Survived','Pclass','Title','Name','Last Name','Sex','Ticket','Cabin','Embarked']



for col in ds:

    print(col)
len(ds)
from sklearn.ensemble import RandomForestClassifier



non_age_columns = ['TicketGroup','SibSp', 'Parch', 'Fare', 'is_class_1', 'is_class_2', 'is_class_3', 'is_Dr', 'is_Master', 'is_Military', 'is_Miss', 'is_Mr', 'is_Mrs', 'is_Noble', 'is_Rev', 'is_female', 'is_male', 'age_discrete_cut_(0.0902, 11.574]', 'age_discrete_cut_(11.574, 22.979]', 'age_discrete_cut_(22.979, 34.383]', 'age_discrete_cut_(34.383, 45.787]', 'age_discrete_cut_(45.787, 57.191]', 'age_discrete_cut_(57.191, 68.596]', 'age_discrete_cut_(68.596, 80]', 'embarked_Cherbourg', 'embarked_Queenstown', 'embarked_Southampton',  'ticket_group_1', 'ticket_group_2', 'ticket_group_3', 'ticket_group_4', 'ticket_group_5', 'ticket_group_6', 'ticket_group_7', 'ticket_group_8', 'ticket_group_11']



train = ds[ds.Survived.notnull()]



# 'ticket_group_ratio_survived_0', 'ticket_group_ratio_survived_1',

selected_column = ['ticket_group_ratio_survived_0', 'ticket_group_ratio_survived_1', 'ticket_group_survived_1','ticket_group_survived_0','SibSp','Parch','Fare','is_class_1','is_class_2','is_class_3','is_Dr','is_Master','is_Military','is_Miss','is_Mr','is_Mrs','is_Noble','is_Rev','is_female','is_male','age_discrete_cut_(0.0902, 11.574]','age_discrete_cut_(11.574, 22.979]','age_discrete_cut_(22.979, 34.383]','age_discrete_cut_(34.383, 45.787]','age_discrete_cut_(45.787, 57.191]','age_discrete_cut_(57.191, 68.596]','age_discrete_cut_(68.596, 80]','embarked_Cherbourg','embarked_Queenstown','embarked_Southampton','ticket_group_1','ticket_group_2','ticket_group_3','ticket_group_4','ticket_group_5','ticket_group_6','ticket_group_7','ticket_group_8','ticket_group_11',]



rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)

rf.fit(train[selected_column], train.Survived)

np.mean(cross_val_score(rf, train[selected_column], train.Survived, n_jobs = -1))