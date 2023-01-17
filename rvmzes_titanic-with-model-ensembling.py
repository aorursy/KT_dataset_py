import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier)

from sklearn.svm import SVC

from sklearn.model_selection import cross_validate, KFold, GridSearchCV

from sklearn.preprocessing import StandardScaler, LabelEncoder
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



target = train['Survived']



dataset = train.append(test, sort=False)



PassengerId = test['PassengerId']



print('train', train.shape)

print('test', test.shape)



train.head()
print(train.info())

print(test.info())
label = LabelEncoder()



# fill nan values

dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

dataset['Age'].fillna(dataset['Age'].median(), inplace=True)



# place Fare and Age into buckets

dataset['Fare'] = pd.qcut(dataset['Fare'], 4)

dataset['Age'] = pd.cut(dataset['Age'].astype('int'), 5)



# create new features from existing

dataset['Title'] = dataset['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

title_names = (dataset['Title'].value_counts() < 10)

dataset['Title'] = dataset['Title'].apply(lambda x: 'Other' if title_names.loc[x] == True else x)

dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

dataset['Alone'] = 0

dataset['Alone'].loc[dataset['FamilySize'] > 1] = 1

dataset['HasCabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

dataset['Name_length'] = dataset['Name'].apply(len)



# transform non-numerical features 

dataset['Sex'] = label.fit_transform(dataset['Sex'])

dataset['Title'] = label.fit_transform(dataset['Title'])

dataset['Embarked'] = label.fit_transform(dataset['Embarked'])

dataset['Fare'] = label.fit_transform(dataset['Fare'])

dataset['Age'] = label.fit_transform(dataset['Age'])
dataset['LastName'] = dataset['Name'].apply(lambda x: str.split(x, ",")[0])

DEFAULT_SURVIVAL_VALUE = 0.5

dataset['FamilySurvival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in dataset.groupby(['LastName', 'Fare']):

    if(len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                dataset.loc[dataset['PassengerId'] == passID, 'FamilySurvival'] = 1

            elif (smin==0.0):

                dataset.loc[dataset['PassengerId'] == passID, 'FamilySurvival'] = 0

for _, grp_df in dataset.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['FamilySurvival'] == 0) | (row['FamilySurvival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    dataset.loc[dataset['PassengerId'] == passID, 'FamilySurvival'] = 1

                elif (smin==0.0):

                    dataset.loc[dataset['PassengerId'] == passID, 'FamilySurvival'] = 0
# drop unnecessary features and scale features to have mean=0 and std=1

dataset.drop(['Name', 'LastName', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)



ss = StandardScaler()

ss.fit_transform(dataset)



train = dataset[:891]

test = dataset[891:]
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())

print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

print(train[['Alone', 'Survived']].groupby(['Alone'], as_index=False).mean())

print(train[['HasCabin', 'Survived']].groupby(['HasCabin'], as_index=False).mean())

print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

print(train[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean())

print(train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean())

print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
train_y = train['Survived']



train.drop(['Survived'], axis=1, inplace=True)

test.drop(['Survived'], axis=1, inplace=True)



train_X = train
rf_parameters = {

    'n_estimators': [300, 500],

    'warm_start': [True],

    'max_depth'   : [n for n in range(6, 12)],

    'max_features': [n for n in range(5, 8)],

    "min_samples_split": [n for n in range(8, 11)],

    "bootstrap": [True, False]

}



et_parameters = {

    'n_estimators': [300, 500],

    'max_depth'   : [n for n in range(2, 10)],

    'max_features': [n for n in range(5, 8)],

    "min_samples_leaf": [n for n in range(1, 6)],

    "bootstrap": [True, False]

}



ada_parameters = {

    'n_estimators': [300, 500],

    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1]

}



gb_parameters = {

    'loss' : ["deviance","exponential"],

    'n_estimators' : [300, 500],

    'learning_rate': [0.0025, 0.005, 0.0075, 0.01, 0.05, 0.1],

    'max_depth':  [n for n in range(2, 6)],

    'max_features': [n for n in range(2, 6)],

    'min_samples_leaf': [n for n in range(1, 7)],

}



svc_parameters = {

    'kernel': ['rbf', 'linear'],

    'C': [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],

    'gamma': [0.001 ,0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 10],

    'probability': [True]

}
rfc_cv = GridSearchCV(RandomForestClassifier(), rf_parameters, cv=4, n_jobs=8).fit(train_X, train_y)

etc_cv = GridSearchCV(ExtraTreesClassifier(), et_parameters, cv=4, n_jobs=4).fit(train_X, train_y)

ada_cv = GridSearchCV(AdaBoostClassifier(), ada_parameters, cv=4, n_jobs=4).fit(train_X, train_y)

gbc_cv = GridSearchCV(GradientBoostingClassifier(), gb_parameters, cv=4, n_jobs=4).fit(train_X, train_y)

svc_cv = GridSearchCV(SVC(), svc_parameters, cv=4, n_jobs=4).fit(train_X, train_y)
cols = train.columns.values



feature_dataframe = pd.DataFrame(

    {

        'Random Forest feature importances': rfc_cv.best_estimator_.feature_importances_,

        'Extra Trees feature importances': etc_cv.best_estimator_.feature_importances_,

        'AdaBoost feature importances': ada_cv.best_estimator_.feature_importances_,

        'Gradient Boost feature importances': gbc_cv.best_estimator_.feature_importances_

    }

)
trace = go.Scatter(

    y = feature_dataframe['Random Forest feature importances'].values,

    x = cols,

    mode='markers',

    marker=dict(

        sizemode='diameter',

        sizeref=1,

        size=25,

        color=feature_dataframe['Random Forest feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = cols

)

data = [trace]



layout = go.Layout(

    autosize= True,

    title='Random Forest Feature Importance',

    hovermode='closest',

    yaxis=dict(

        title='Feature Importance',

        ticklen=5,

        gridwidth=2

    ),

    showlegend=False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



trace = go.Scatter(

    y=feature_dataframe['Extra Trees feature importances'].values,

    x=cols,

    mode='markers',

    marker=dict(

        sizemode='diameter',

        sizeref=1,

        size=25,

        color=feature_dataframe['Extra Trees feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = cols

)

data = [trace]



layout = go.Layout(

    autosize=True,

    title='Extra Trees Feature Importance',

    hovermode='closest',

    yaxis=dict(

        title='Feature Importance',

        ticklen=5,

        gridwidth=2

    ),

    showlegend=False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



trace = go.Scatter(

    y=feature_dataframe['AdaBoost feature importances'].values,

    x=cols,

    mode='markers',

    marker=dict(

        sizemode='diameter',

        sizeref=1,

        size=25,

        color=feature_dataframe['AdaBoost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = cols

)

data = [trace]



layout = go.Layout(

    autosize=True,

    title='AdaBoost Feature Importance',

    hovermode='closest',

    yaxis=dict(

        title='Feature Importance',

        ticklen=5,

        gridwidth=2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='scatter2010')



trace = go.Scatter(

    y = feature_dataframe['Gradient Boost feature importances'].values,

    x = cols,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['Gradient Boost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = cols

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title='Gradient Boosting Feature Importance',

    hovermode='closest',

    yaxis=dict(

        title='Feature Importance',

        ticklen=5,

        gridwidth=2

    ),

    showlegend=False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
feature_dataframe['mean'] = feature_dataframe.mean(axis=1)

feature_dataframe.head(3)
y = feature_dataframe['mean'].values

x = cols

data = [go.Bar(

            x=x,

            y=y,

            width=0.5,

            marker=dict(

                color = feature_dataframe['mean'].values,

                colorscale='Portland',

                showscale=True,

                reversescale = False

            ),

            opacity=0.6

)]

layout = go.Layout(

    autosize=True,

    title='Barplots of Mean Feature Importance',

    hovermode='closest',

    yaxis=dict(

        title='Feature Importance',

        ticklen=5,

        gridwidth=2

    ),

    showlegend=False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='bar-direct-labels')
vc = VotingClassifier(estimators=[

    ('rfc', rfc_cv.best_estimator_), 

    ('etc', etc_cv.best_estimator_), 

    ('ada', ada_cv.best_estimator_),

    ('gbc', gbc_cv.best_estimator_), 

    ('svm', svc_cv.best_estimator_)

], voting='soft', n_jobs=4)



vc = vc.fit(train_X, train_y)
predictions = vc.predict(test)



submission = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": predictions

    }, dtype='int64')



submission.head(15)
submission.to_csv('titanic.csv', index=False)