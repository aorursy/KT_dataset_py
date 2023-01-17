# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/titanic/train.csv")

df_test=pd.read_csv("../input/titanic/test.csv")

df_gender_submission=pd.read_csv("../input/titanic/gender_submission.csv")
df_train.head()
df_train.info()
df_train.shape
df_train.describe()
df_train['Age'].fillna(df_train['Age'].median(),inplace=True)
df_train['Died'] = 1-df_train['Survived']
df_train.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar', figsize=(5, 5),

                                                          stacked=True, color=['b','r']);
df_train.groupby('Sex').agg('mean')[['Survived','Died']].plot(kind='bar', figsize=(5, 5),

                                                          stacked=True, color=['g','r']);
fig = plt.figure(figsize=(5, 5))

sns.violinplot(x='Sex', y='Age', 

               hue='Survived', data=df_train, 

               split=True,

               palette={0: "r", 1: "g"}

              );
figure = plt.figure(figsize=(20, 7))

plt.hist([df_train[df_train['Survived'] == 1]['Fare'], df_train[df_train['Survived'] == 0]['Fare']], 

         stacked=True, color = ['g','r'],

         bins = 50, label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend();
plt.figure(figsize=(20, 7))

ax = plt.subplot()



ax.scatter(df_train[df_train['Survived'] == 1]['Age'], df_train[df_train['Survived'] == 1]['Fare'], 

           c='green', s=df_train[df_train['Survived'] == 1]['Fare'])

ax.scatter(df_train[df_train['Survived'] == 0]['Age'], df_train[df_train['Survived'] == 0]['Fare'], 

           c='red', s=df_train[df_train['Survived'] == 0]['Fare']);
ax = plt.subplot()

ax.set_ylabel('Average fare')

df_train.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(20, 7), ax = ax);
def merge_train_test_data():

    # reading train data

    train = pd.read_csv('../input/titanic/train.csv')

    

    # reading test data

    test = pd.read_csv('../input/titanic/test.csv')



    # extracting and then removing the targets from the training data 

    targets = train.Survived

    train.drop(['Survived'], 1, inplace=True)

    



    # merging train data and test data for future feature engineering

    # we'll also remove the PassengerID since this is not an informative feature

    merged_data_frame = train.append(test)

    merged_data_frame.reset_index(inplace=True)

    merged_data_frame.drop(['index', 'PassengerId'], inplace=True, axis=1)

    

    return merged_data_frame
merged_df = merge_train_test_data()
merged_df.shape
merged_df.head()
merged_df['Name'][1].split(',')[1].strip().split('.')[0].strip()
merged_df['Title'] = merged_df['Name'].map(lambda str_name:str_name.split(',')[1].strip().split('.')[0].strip())
merged_df['Title'].unique()
title_dict={'Mr':'Mr',

           'Mrs':'Mrs',

           'Ms':'Mrs',

           'Mme':'Miss',

           'Miss':'Miss',

           'Mlle':'Miss',

           'Master':'Master',

            'Col':'Defence_Officer',

            'Major':'Defence_Officer',

            'Capt':'Defence_Officer',

            'Dr':'Dr',

            'Jonkheer':'Jonkheer',

            'Don':'Don',

            'Rev':'Rev',

            'Lady':'Lady',

            'Sir':'Sir',

            'the Countess':'the Countess',

            'Dona':'Dona'

           }
merged_df['Title'] = merged_df['Title'].map(title_dict)
merged_df['Title'].unique()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(merged_df['Title'], drop_first=True,prefix='Title')

# Adding the results to the master dataframe

merged_df = pd.concat([merged_df, dummy1], axis=1)

merged_df.drop('Title',axis=1,inplace=True)
merged_df.drop('Name',axis=1,inplace=True)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(merged_df['Pclass'].astype('category'), drop_first=True,prefix='Pclass')

# Adding the results to the master dataframe

merged_df = pd.concat([merged_df, dummy1], axis=1)

merged_df.drop('Pclass',axis=1,inplace=True)
sex_dict = {'male':1,

           'female':0}

merged_df['Sex'] = merged_df['Sex'].map(sex_dict)
merged_df['Cabin'] = merged_df['Cabin'].str[0]
merged_df['Cabin'].unique()
merged_df['Cabin'].fillna('U',inplace=True)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(merged_df['Cabin'], drop_first=True,prefix='Cabin')

# Adding the results to the master dataframe

merged_df = pd.concat([merged_df, dummy1], axis=1)

merged_df.drop('Cabin',axis=1,inplace=True)
merged_df.groupby('Embarked').count()['Sex']
merged_df['Embarked'].isnull().sum()
#Replacing the value of Embarked with 'S' based on the frequency

merged_df['Embarked'].fillna('S',inplace=True)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(merged_df['Embarked'], drop_first=True,prefix='Embarked')

# Adding the results to the master dataframe

merged_df = pd.concat([merged_df, dummy1], axis=1)

merged_df.drop('Embarked',axis=1,inplace=True)
merged_df['Age'].isnull().sum()
def age_groups(num_age):

    if num_age>=0 and num_age <=2:

        return 'infant'

    elif num_age>=3 and num_age<=15:

        return 'kids'

    else:

        return 'adult'
merged_df['age_group']=merged_df['Age'].map(lambda age:age_groups(age))
age_group=merged_df.groupby('age_group').median()['Age'].reset_index()
age_group
age_group[age_group['age_group']=='adult']['Age'][0]
def age(row):

    if row['age_group']=='adult':

        return age_group[age_group['age_group']=='adult']['Age'][0]

    elif row['age_group']=='infant':

        return age_group[age_group['age_group']=='infant']['Age'][0]

    else:

        return age_group[age_group['age_group']=='kids']['Age'][0]
merged_df['Age']=merged_df.apply(lambda row: age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
merged_df.head()
merged_df.drop('age_group',axis=1,inplace=True)
merged_df['traveller_cnt'] = merged_df['SibSp'] + merged_df['Parch']+1
merged_df.drop('Ticket',axis=1,inplace=True)
def get_train_test_target():

    targets = pd.read_csv('../input/titanic/train.csv', usecols=['Survived'])['Survived'].values

    train = merged_df.iloc[:891]

    test = merged_df.iloc[891:]

    

    return train, test, targets
train, test, targets = get_train_test_target()
train.head()
test.isnull().sum()
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
# Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, max_features='sqrt')

rf = rf.fit(train, targets)
features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = rf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(25, 25))
from sklearn.feature_selection import SelectFromModel

model = SelectFromModel(rf, prefit=True)

train_reduced = model.transform(train)

print(train_reduced.shape)
test_reduced = model.transform(test)

print(test_reduced.shape)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

#logreg_cv = LogisticRegressionCV()

rf = RandomForestClassifier()

#gboost = GradientBoostingClassifier()

#models = [logreg, logreg_cv, rf, gboost]

models = [logreg, rf]
from sklearn.model_selection import cross_val_score

def compute_score(rf, X, y, scoring='accuracy'):

    xval = cross_val_score(rf, X, y, cv = 5, scoring=scoring)

    return np.mean(xval)
for model in models:

    print('Cross-validation of : {0}'.format(model.__class__))

    score = compute_score(rf=model, X=train_reduced, y=targets, scoring='accuracy')

    print('CV score = {0}'.format(score))

    print('****')
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

# turn run_gs to True if you want to run the gridsearch again.

run_gs = False



if run_gs:

    parameter_grid = {

                 'max_depth' : [4, 6, 8],

                 'n_estimators': [50, 10],

                 'max_features': ['sqrt', 'auto', 'log2'],

                 'min_samples_split': [2, 3, 10],

                 'min_samples_leaf': [1, 3, 10],

                 'bootstrap': [True, False],

                 }

    forest = RandomForestClassifier()

    cross_validation = StratifiedKFold(n_splits=5)



    grid_search = GridSearchCV(forest,

                               scoring='accuracy',

                               param_grid=parameter_grid,

                               cv=cross_validation,

                               verbose=1,

                               n_jobs=-1

                              )



    grid_search.fit(train, targets)

    model = grid_search

    parameters = grid_search.best_params_



    print('Best score: {}'.format(grid_search.best_score_))

    print('Best parameters: {}'.format(grid_search.best_params_))

    

else: 

    parameters = {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 50, 

                  'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 6}

    

    model = RandomForestClassifier(**parameters)

    model.fit(train, targets)
test.head()
predictions = model.predict(test)
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(predictions)
y_pred_1.head()
# Renaming the column 

y_pred_1= y_pred_1.rename(columns={ 0 : 'Survived'})
df_gender_submission['Survived'] = y_pred_1['Survived'] 
df_gender_submission.head()