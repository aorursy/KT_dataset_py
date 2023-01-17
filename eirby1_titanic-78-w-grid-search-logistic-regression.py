import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

import warnings  

warnings.filterwarnings('ignore')

import re

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn import linear_model

from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#Save the PassengerId for later construction of the submission file

PassengerId = test['PassengerId']
(train.shape[0], test.shape[0])
train.head(3)
train.count() / len(train)
test.count() / len(test)
def drop_incomplete_cols(df):

    s = df.count() / len(df)

    threshold = .4

    for col_name in s.index:

        if s[col_name] < threshold:

            df.drop(col_name, axis=1, inplace=True)
all_data = [train, test]



for df in all_data:

    df.drop(['PassengerId'], axis=1, inplace=True)

    #impute missing Age with the mean across Sex

    df['Age'].fillna(df.groupby(['Sex'])['Age'].transform(np.mean), inplace=True)

    #impute missing Fare with the mean across Pclass and Embarked

    df['Fare'].fillna(df.groupby(['Pclass', 'Embarked'])['Fare'].transform(np.mean), inplace=True)

    #impute missing Embarked with mode of data set

    df['Embarked'].fillna(df['Embarked'].mode().iloc[0], inplace=True)

    drop_incomplete_cols(df)
train.count()/len(train)
test.count()/len(test)
(train.shape[0], test.shape[0])
train.corr().abs().unstack().sort_values(ascending=False)[len(train.corr().columns):len(train.corr().columns) + 10]
fig, axes = plt.subplots(1, 2, figsize=(30, 8))

women = train[train['Sex'] == 'female']

men = train[train['Sex'] == 'male']

ax = sns.distplot(women[women['Survived']==1].Age, bins=65, label='survived', ax=axes[0], kde=False)

ax = sns.distplot(women[women['Survived']==0].Age, bins=65, label='not survived', ax=axes[0], kde=False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age, bins=85, label= \

                  'survived', ax=axes[1], kde=False)

ax = sns.distplot(men[men['Survived']==0].Age, bins=85, label= \

                  'not survived', ax=axes[1], kde=False)

ax.legend()

ax.set_title('Male')

all_data = [train, test]

for df in all_data:

    df['num_relatives'] = df['SibSp'] + df['Parch']

    df.loc[df['num_relatives'] > 0, 'solo'] = 0

    df.loc[df['num_relatives'] == 0, 'solo'] = 1

    df['solo'] = df['solo'].astype(int)

    #drop 'SibSp' and 'Parch' columns:

    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
axes = sns.catplot('num_relatives', 'Survived',

                  data=train, kind='point', aspect=2)
fig, axes = plt.subplots(1, 2, figsize=(30, 8))

women = train[train['Sex'] == 'female']

men = train[train['Sex'] == 'male']

ax = sns.distplot(women[women['Survived']==1].num_relatives, label='survived', ax=axes[0], kde=False)

ax = sns.distplot(women[women['Survived']==0].num_relatives, label='not survived', ax=axes[0], kde=False)

ax.legend()

ax.set_title('Female')

ax.set_xticks(range(0,10))

ax = sns.distplot(men[men['Survived']==1].num_relatives, label= \

                  'survived', ax=axes[1], kde=False)

ax = sns.distplot(men[men['Survived']==0].num_relatives, label= \

                  'not survived', ax=axes[1], kde=False)

ax.legend()

ax.set_xticks(range(0,10))

ax.set_title('Male')

def add_title_col(df):

    pattern = r'(Mr\.|Mrs|Ms|Miss|Master|Dr\.|Don\.|Dona\.|Rev\.|Sir\.|Lady|Mme|Mlle|Major|Col\.|Capt\.|Countess|Jonkheer)'

    title = df.Name.str.extract(pattern).fillna('NONE')

    title.columns = ['title']

    return pd.concat((df, title), axis=1)
def add_social_status_col(df):

    classes = ['peerage', 'upper', 'officer', 'clergy', 'middle', 'lower']

    peerage = ['Don.', 'Dona.', 'Sir.', 'Lady', 'Mme', 'Mlle', 'Countess', 'Jonkheer']

    officer = ['Col.', 'Major', 'Capt.']

    clergy = ['Rev.']

    basic_honorific = ['Mr.', 'Mrs', 'Ms', 'Miss', 'Master', 'Dr.']

    

    df.loc[df['title'].isin(peerage), 'social_status'] = 'peerage'

    df.loc[(df['title'].isin(basic_honorific) & (df['Pclass'] == 1)), 'social_status'] = 'upper'

    df.loc[df['title'].isin(officer), 'social_status'] = 'officer'

    df.loc[df['title'].isin(clergy), 'social_status'] = 'clergy'

    df.loc[(df['title'].isin(basic_honorific) & (df['Pclass'] == 2)), 'social_status'] = 'middle'

    df.loc[(df['title'].isin(basic_honorific) & (df['Pclass'] == 3)), 'social_status'] = 'lower'

    

    #test:

    if len(df[~df['social_status'].isin(classes)]) == 0:

        print('All passengers have been assigned a social status')

    else:

        print('social status assignment was NOT successful.')

        

    return df
train = add_title_col(train)

train = add_social_status_col(train)



test = add_title_col(test)

test = add_social_status_col(test)
train.head()
def create_ticket_digit_col(df):

    if 'first_digit' not in df.columns:

        pattern = r'(\d{1})\d+$'

        first_digit = df.Ticket.str.extract(pattern).fillna('0')

        first_digit.columns = ['first_digit']

        df = pd.concat([df, first_digit], axis=1)  

        return df

    else:

        return df
train = create_ticket_digit_col(train)

test = create_ticket_digit_col(test)
train.head()
def view_ticket_survival_stats(df):

    pre = []

    m = []

    c = []

    fd_list = df['first_digit'].value_counts().index.sort_values()

    for i in fd_list:

        pre.append(i)

        m.append(df[df['first_digit'] == i].loc[:, 'Survived'].mean())

        c.append(df[df['first_digit'] == i].loc[:, 'Survived'].count())



    prefix_survival_pct = pd.DataFrame({'prefix': pre, 'count': c, 'survival_rate': m})

    return prefix_survival_pct.sort_values(by='survival_rate', ascending=False)
view_ticket_survival_stats(train)
train.head()
def safe_column_remove(df, columns):

    for col in columns:

        if col in df.columns:

            df.drop(col, axis=1, inplace=True)

    return df
#save the labels before removing the column:

train_labels = train['Survived']



cols_to_remove = ['Survived', 'Name', 'Ticket', 'title']

train = safe_column_remove(train, cols_to_remove)

test = safe_column_remove(test, cols_to_remove)
def titanic_scaler_encoder(df, isTreeInput=True):

    if isTreeInput==True:

        ss_cols  = ['Age', 'Fare', 'num_relatives']

        encoded_cols = ['Pclass', 'Sex', 'Embarked', 'social_status', 'first_digit']

        unchd_cols = ['solo']

        

        scaler = StandardScaler()  

        scaled_data  = scaler.fit_transform(df[ss_cols])  

        label_encoded_data = df[encoded_cols].apply(LabelEncoder().fit_transform)

        

        return np.concatenate([scaled_data, label_encoded_data, df[unchd_cols]], axis=1)     

    else:

        ss_cols  = ['Age', 'Fare', 'num_relatives']

        unchd_cols = ['solo', 'Pclass', 'Sex', 'Embarked', 'social_status', 'first_digit']

    

        scaler = StandardScaler()  

        scaled_data  = scaler.fit_transform(df[ss_cols])

        

        return np.concatenate([scaled_data, df[unchd_cols]], axis=1)
train_prepared = titanic_scaler_encoder(train)

test_prepared = titanic_scaler_encoder(test)
train_prepared
test_prepared
param_grid = [{'penalty' : ['l1', 'l2'], 'C' : np.logspace(-4, 4, 20), 'solver' : ['liblinear']}]

lg_clf = GridSearchCV(linear_model.LogisticRegression(), param_grid, cv=5, scoring='roc_auc')

lg_clf.fit(train_prepared, train_labels)

lg_predictions = lg_clf.best_estimator_.predict(test_prepared)
lg_clf.best_params_, lg_clf.best_score_
submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': lg_predictions})

submission.Survived.astype(int)

submission.head(20)
submission.to_csv('titanic_submission_5_1.csv', float_format='%.f', index=False)