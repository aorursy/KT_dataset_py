import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

import os



os.chdir('../input/titanic/')

plt.rcParams['figure.figsize'] = (15.0, 6.0)

plt.style.use('ggplot')
# just tallying the format



submission_format = pd.read_csv('gender_submission.csv')

submission_format.shape

submission_format.head()
# test data



test_df = pd.read_csv('test.csv')

test_df.shape

test_df.head()
# train data



train_df = pd.read_csv('train.csv')

train_df.shape

train_df.head()
train_df.dtypes
train_df.isnull().sum()
# validating unique IDs

train_df['PassengerId'].nunique(), train_df.shape[0]
# proportion of survivalists

# 0-No, 1-Yes



train_df['Survived'].value_counts()

train_df['Survived'].value_counts(normalize=True)*100

train_df['Survived'].value_counts(normalize=True).plot.bar(figsize=(16, 5))
# class-wise proportion (the class of class has a descending ordinality)



train_df['Pclass'].value_counts()

train_df['Pclass'].value_counts(normalize=True)*100

train_df['Pclass'].value_counts(normalize=True).plot(kind='bar', figsize=(16, 5))
# percentage of casualties, class wise



(train_df.groupby(['Pclass', 'Survived']).size()/train_df.shape[0])*100

train_df.groupby(['Pclass', 'Survived']).size().plot.bar(label='(Pclass, Survived)', figsize=(16, 5))
# percentage of casualties within a class



(train_df.groupby(['Pclass', 'Survived']).size()/train_df.groupby(['Pclass']).size())*100
# class wise survival ratio



(train_df.groupby(['Survived', 'Pclass']).size()/train_df.groupby(['Survived']).size())*100
# had to

[i for i in set(train_df['Name']) if 'rose' in i or 'jack' in i]
# sex ratio



train_df['Sex'].value_counts()

train_df['Sex'].value_counts(normalize=True)

train_df['Sex'].value_counts(normalize=True).plot(kind='bar', figsize=(16, 5))
# deaths of passengers by SEX x Pclass

train_df[train_df['Survived']==0].groupby(['Sex', 'Pclass']).size().plot(kind='bar', label='(Sex, Pclass)', figsize=(16, 5))
# percentage of deaths, among the classes, within gender

(train_df[train_df['Survived']==0].groupby(['Sex', 'Pclass']).size()/train_df[train_df['Survived']==0].groupby(['Sex']).size())*100
# percentage of deaths, among the genders, within class

(train_df[train_df['Survived']==0].groupby(['Pclass', 'Sex']).size()/train_df[train_df['Survived']==0].groupby(['Pclass']).size())*100
# Age distribution



train_df['Age'].hist(bins=40, figsize=(16, 5))
# total unique age values



train_df['Age'].nunique()
# get some statistical insights



train_df['Age'].describe()
train_df.groupby('Survived')['Age'].describe()
# null check



train_df['Age'].isnull().sum(), train_df.shape[0]

(train_df['Age'].isnull().sum()/train_df.shape[0]) * 100
# SibSp (# of siblings/spouses aboard the ship)



train_df['SibSp'].value_counts()

train_df['SibSp'].value_counts(normalize=True)

train_df['SibSp'].value_counts().plot(kind='bar', figsize=(16, 5))
# Class wise deaths of sib/sp



(train_df[train_df['Survived']==0].groupby(['Pclass'])['SibSp'].sum()/train_df[train_df['Survived']==0]['SibSp'].sum())*100
# Parch (# of parents/children aboard)

train_df['Parch'].value_counts()

train_df['Parch'].value_counts(normalize=True)

train_df['Parch'].value_counts().plot(kind='bar', figsize=(16, 5))
# Class wise deaths of parch



(train_df[train_df['Survived']==0].groupby(['Pclass'])['Parch'].sum()/train_df[train_df['Survived']==0]['Parch'].sum())*100
# does having family aboard increase or decrease the chances of survival?

(train_df[(train_df['Parch']>0) | (train_df['SibSp']>0)].groupby('Survived').size()/train_df[(train_df['Parch']>0) | (train_df['SibSp']>0)].shape[0])*100
# Fare



train_df['Fare'].describe()

train_df['Fare'].hist(bins=60, figsize=(16, 5))
train_df[train_df['Fare']==0].shape
# class wise fares



train_df.groupby(['Pclass'])['Fare'].describe()
# binning fare



train_df['fare_range'] = pd.cut(x=train_df['Fare'], bins=[i for i in range(0, 100, 10)]+[i for i in range(100, 700, 100)], include_lowest=True)
(train_df[train_df['Survived']==0].groupby(['Pclass', 'fare_range']).size()/train_df[train_df['Survived']==0].shape[0])

(train_df[train_df['Survived']==0].groupby(['Pclass', 'fare_range']).size()/train_df[train_df['Survived']==0].shape[0]).plot(kind='barh', figsize=(16, 6))
(train_df.groupby(['Pclass', 'fare_range']).size()/train_df.shape[0])

(train_df.groupby(['Pclass', 'fare_range']).size()/train_df.shape[0]).plot(kind='barh', figsize=(16, 6))
# gender wise fare

train_df.groupby(['Pclass', 'Sex'])['Fare'].describe()
# # imputing with mean

train_df['Age_Imputed'] = train_df.groupby(['Pclass', 'Sex', 'fare_range'])['Age'].transform(lambda x: x.fillna(x.mean()))
train_df['Age_Imputed'].isnull().sum()
# imputing with mean

train_df['Age_Imputed'] = train_df.groupby(['Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
train_df['Age_Imputed'].isnull().sum()
train_df['Age_Imputed'] = train_df['Age_Imputed'].astype('int64')
train_df['Age_Imputed'].describe()
train_df.loc[train_df['Age_Imputed']==0, 'Age_Imputed'] = 1
train_df['age_range'] = pd.cut(x=train_df['Age_Imputed'], bins=[i for i in range(0, 90, 10)])

train_df['age_range'].value_counts(normalize=True)

train_df['age_range'].value_counts(normalize=True).plot(kind='bar', figsize=(16, 5))
# age wise death dist among classes



(train_df[train_df['Survived']==0].groupby(['Pclass', 'age_range']).size()/train_df[train_df['Survived']==0].groupby(['Pclass']).size()) * 100
(train_df[train_df['Survived']==0].groupby(['age_range', 'Pclass']).size()/train_df[train_df['Survived']==0].groupby(['age_range']).size()) * 100
train_df.groupby(['Pclass', 'age_range'])['Fare'].describe()
# % nulls in Cabin columns (ignoring)

(train_df['Cabin'].isnull().sum() / train_df.shape[0])*100
# Embarked (Point of Embarking)

# C = Cherbourg, Q = Queenstown, S = Southampton

train_df['Embarked'].value_counts()

train_df['Embarked'].value_counts(normalize=True)

train_df['Embarked'].value_counts().plot(kind='bar', figsize=(16, 5))
train_df['Embarked'].isnull().sum()
train_df['Embarked'].fillna('S', inplace=True)
# surival perc amongst poe

(train_df.groupby(['Embarked', 'Survived']).size()/train_df.shape[0]) * 100
# survival perc

(train_df.groupby(['Embarked', 'Survived']).size()/train_df.groupby('Embarked').size()) * 100
# of all that didn't survive, poe dist



(train_df.groupby(['Survived', 'Embarked']).size()/train_df.groupby(['Survived']).size()) * 100
(train_df.groupby(['Embarked', 'Pclass']).size()/train_df.groupby(['Embarked']).size()) * 100
train_df.groupby(['Pclass', 'Embarked'])['Fare'].describe()
train_df['Deck'] = train_df['Cabin'].apply(lambda x: x[:1] if pd.notnull(x) else x)
(train_df.groupby(['Deck', 'Survived']).size()/train_df.groupby(['Deck']).size())*100
(train_df[train_df['Survived']==0].groupby('Deck').size()/train_df[train_df['Survived']==0].shape[0])*100
train_df['Family_Size'] = train_df['SibSp'] + train_df['Parch']
train_df.head()
test_df.isnull().sum()
#imputing null fare

test_df['Fare'] = test_df.groupby(['Pclass', 'Sex'])['Fare'].transform(lambda x: x.fillna(x.mean()))
test_df['fare_range'] = pd.cut(x=test_df['Fare'], bins=[i for i in range(0, 100, 10)]+[i for i in range(100, 700, 100)], include_lowest=True)
# # imputing with mean

test_df['Age_Imputed'] = test_df.groupby(['Pclass', 'Sex', 'fare_range'])['Age'].transform(lambda x: x.fillna(x.mean()))
test_df['Age_Imputed'].isnull().sum()
# imputing with mean

test_df['Age_Imputed'] = test_df.groupby(['Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
test_df['Age_Imputed'].isnull().sum()
test_df['Age_Imputed'] = test_df['Age_Imputed'].astype('int64')
test_df['Age_Imputed'].describe()
test_df.loc[test_df['Age_Imputed']==0, 'Age_Imputed'] = 1
test_df['age_range'] = pd.cut(x=test_df['Age_Imputed'], bins=[i for i in range(0, 90, 10)])

test_df['age_range'].value_counts(normalize=True)

test_df['age_range'].value_counts(normalize=True).plot(kind='bar', figsize=(16, 5))
test_df['Family_Size'] = test_df['SibSp'] + test_df['Parch']
test_df.head()
train_df['model_type'] = 'Train'

test_df['model_type'] = 'Test'
feature_df = train_df.append(test_df, sort=False, ignore_index=True)

feature_df.groupby('model_type').size()

feature_df.shape

feature_df.head()
feature_df.isnull().sum()
feature_df['fare_range'] = feature_df['fare_range'].astype(str)

feature_df['age_range'] = feature_df['age_range'].astype(str)
feature_df.loc[feature_df['Family_Size']==0, 'Family_Onboard'] = '0'

feature_df.loc[feature_df['Family_Size']!=0, 'Family_Onboard'] = '1'
# One Hot Encoded columns

feature_df = feature_df.join(pd.get_dummies(feature_df[['Sex', 'Embarked', 'Family_Onboard', 'fare_range', 'age_range']]))

feature_df.columns
# median fare

feature_df['median_fare'] = feature_df.groupby('fare_range')['Fare'].transform('median')
# delta fare

feature_df['delta_fare'] =  (feature_df['Fare'] - feature_df['median_fare'])/feature_df['median_fare']

feature_df['delta_fare'].describe()
# cat coding features

feature_df['Sex_category'] = feature_df['Sex'].astype('category').cat.codes

feature_df['Embarked_category'] = feature_df['Embarked'].astype('category').cat.codes

feature_df['fare_range_category'] = feature_df['fare_range'].astype('category').cat.codes

feature_df['age_range_category'] = feature_df['age_range'].astype('category').cat.codes

feature_df['Family_Onboard_category'] = feature_df['Family_Onboard'].astype('category').cat.codes
feature_df.columns
l_features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age_Imputed', 'Family_Size', 'Sex_female', 

              'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Family_Onboard_0', 'Family_Onboard_1',

              'fare_range_(-0.001, 10.0]', 'fare_range_(10.0, 20.0]',

              'fare_range_(100.0, 200.0]', 'fare_range_(20.0, 30.0]',

              'fare_range_(200.0, 300.0]', 'fare_range_(30.0, 40.0]',

              'fare_range_(40.0, 50.0]', 'fare_range_(50.0, 60.0]',

              'fare_range_(500.0, 600.0]', 'fare_range_(60.0, 70.0]',

              'fare_range_(70.0, 80.0]', 'fare_range_(80.0, 90.0]',

              'fare_range_(90.0, 100.0]', 'age_range_(0, 10]', 'age_range_(10, 20]',

              'age_range_(20, 30]', 'age_range_(30, 40]', 'age_range_(40, 50]',

              'age_range_(50, 60]', 'age_range_(60, 70]', 'age_range_(70, 80]',

              'median_fare', 'delta_fare', 'Sex_category', 'Embarked_category',

              'fare_range_category', 'age_range_category', 'Family_Onboard_category']



pred_y = 'Survived' 



len(l_features)
feature_df[l_features].isnull().sum()
plt.figure(figsize=(16, 16))

sns.heatmap(feature_df[l_features].corr(), cmap='RdYlGn')
l_features = list(set(l_features) - {'Parch', 'SibSp','median_fare'})

len(l_features)
def scale_matrix(X, infer=False, scaler=None):

    if not infer:

        scaler = MinMaxScaler().fit(X)

        t_X = scaler.transform(X)

    else:

        t_X = scaler.transform(X)

    return t_X, scaler
# creating train-val-test split

d_matrix = feature_df[feature_df['model_type']=='Train'].reset_index(drop=True).copy()

train_X, val_X, train_y, val_y = train_test_split(d_matrix[l_features].copy(), d_matrix[pred_y].copy(), test_size=0.2)



test_X = feature_df[feature_df['model_type']=='Test'][l_features].reset_index(drop=True).copy()



d_matrix_X = d_matrix[l_features].copy()

d_matrix_y = d_matrix[pred_y].copy()



d_matrix_X.shape, d_matrix_y.shape

train_X.shape, train_y.shape

val_X.shape, val_y.shape

test_X.shape
# scaling features

train_X, mm_scaler = scale_matrix(train_X.copy())

val_X, _ = scale_matrix(val_X.copy(), infer=True, scaler=mm_scaler)



d_matrix_X, act_mm_scaler = scale_matrix(d_matrix_X.copy())



test_X, _ = scale_matrix(test_X.copy(), infer=True, scaler=act_mm_scaler)
def evaluation_fn(y_true, y_pred):

    print('accuracy', accuracy_score(y_true, y_pred))

    print('f1_score', f1_score(y_true, y_pred))

    print()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    conf_mat = pd.DataFrame()

    conf_mat[0] = ['True', 'False']

    conf_mat['Positive'] = [tp, fp]

    conf_mat['Negative'] = [tn, fn]

    conf_mat = conf_mat.set_index(0)

    print(conf_mat)
def logistic_regression():

    return LogisticRegression(random_state=1511, n_jobs=-1, max_iter=300,

                                        solver='saga', fit_intercept=True, penalty='elasticnet', C=1.0, l1_ratio=0.6)



def random_forest_classifier():

    return RandomForestClassifier(random_state=1511, n_jobs=-1, n_estimators=200,

                                            max_depth=11, min_samples_split=8, min_samples_leaf=2,

                                            max_features='auto', bootstrap=True)



def multilayer_perceptron_classifier():

    return MLPClassifier(hidden_layer_sizes=(100, 100), batch_size=256, 

                         max_iter=1000, random_state=1511, early_stopping=True)



def k_neighbours_classifier():

    return KNeighborsClassifier(n_neighbors=7, n_jobs=-1)



def support_vector_machine():

    return SVC(random_state=1511)



def model_blueprint(X, y, model, infer=False, this_model=None):

    model_mapper_dict = {

        'rf': random_forest_classifier(),

        'lr': logistic_regression(),

        'mlp': multilayer_perceptron_classifier(),

        'knn': k_neighbours_classifier(),

        'svm': support_vector_machine()

    }

    if not infer:

        this_model = model_mapper_dict[model]

        this_model = this_model.fit(X, y)        



    predicted_y = this_model.predict(X)

    if len(y)>0:

        evaluation_fn(y, predicted_y)



    return predicted_y, this_model
champ_model = 'rf'



print('*'*7, 'TRAIN', '*'*7, '\n')

_, model = model_blueprint(train_X, train_y, champ_model)



print()



print('\n', '*'*7, 'VAL', '*'*7, '\n')

_, _ = model_blueprint(val_X, val_y, champ_model, True, model)



test_y, _ = model_blueprint(test_X, pd.Series(), champ_model, True, model)

test_X.shape, test_y.shape



s_output = test_df[['PassengerId']].copy()

s_output[pred_y] = test_y

s_output[pred_y] = s_output[pred_y].astype('int64')

s_output.shape
s_output[pred_y].value_counts()

s_output.head()
s_output.to_csv('my_submission.csv', index=False)