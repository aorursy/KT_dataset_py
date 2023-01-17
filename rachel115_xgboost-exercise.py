# import required libraries

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('darkgrid')

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split,GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from scipy.stats import chi2_contingency

from scipy.stats import chi2
# read train and test data

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

combine = [train_df, test_df]
# shape of train and test data, there are 12 columns in train data and 11 columns in test data since test data has no target.

print(train_df.shape)

print(test_df.shape)
# show summary of train_df

train_df.describe()
train_df.describe(include=['O'])
# show info of train_df. See 'PassengerId','Survived ','Pclass','Name','Sex','SibSp','Parch','Ticket','Fare' have no missing values

# 'Cabin' has many nan values

train_df.info()
test_df.info()
# check type of each column

train_df.dtypes
# try to find outlier for numerical features

df_num = train_df.select_dtypes(include = ['float64', 'int64'])

print(df_num.columns)

df_num.drop(['PassengerId','Survived'], axis=1, inplace=True)
# We see age is almost normal distribution and Fare is skewed right, and seems 500 is outlier, maybe could try log(Fare) to get normal distribution

# Parch，SibSp and Pclass need to seem as categorical features

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
# remove outlier of Fare

train_df = train_df[train_df['Fare'] < 300] 

combine = [train_df, test_df]
# use chi-square test to find the relationship between categorical features and target



# contingency table

for i in cat_list:

    

    feature_select = pd.crosstab(index=train_df[i], 

                           columns=train_df["Survived"])



    stat, p, dof, expected = chi2_contingency(feature_select)

    # interpret p-value

    alpha = 0.05

    print('%s, significance=%.3f, p=%.3f' % (i, alpha, p))

    if p <= alpha:

        print('Dependent (reject H0)')

    else:

        print('Independent (fail to reject H0)')       

        
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
guess_ages = np.zeros((2,3))
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass

freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]
train_X = train_df.drop("Survived", axis=1)

train_y = train_df["Survived"]

test_X  = test_df.drop("PassengerId", axis=1).copy()

train_X.shape, train_y.shape, test_X.shape
# check columns with NaN

cols_with_missing_train = [col for col in train_X.columns 

                                 if train_X[col].isnull().any()]

cols_with_missing_train
cols_with_missing_test = [col for col in test_X.columns 

                                 if test_X[col].isnull().any()]

cols_with_missing_test
# two lists contain numerical columns' name and categorical columns' name

num_cols = ["Age","Fare"]

cat_cols = ["Sex","SibSp","Parch","Embarked","Pclass"]
# handling missing value in num_cols using impute

num_imputer = SimpleImputer()

X = pd.concat([train_X,test_X], axis=0, ignore_index=True)



X[num_cols] = num_imputer.fit_transform(X[num_cols])
# handling missing value in Embarked using the most frequent value 

#  fill every column with its own most frequent value

# df = df.apply(lambda x:x.fillna(x.value_counts().index[0])) or df = df.fillna(df.mode().iloc[0])

X = X.fillna(X['Embarked'].value_counts().index[0])
# handling categorical columns using number label

for col in cat_cols:

    cat = LabelEncoder()

#     cat.fit(list(train_X[col].values.astype('str')) + list(test_X[col].values.astype('str')))

#     train_X[col] = cat.transform(list(train_X[col].values.astype('str')))

#     test_X[col] = cat.transform(list(test_X[col].values.astype('str')))

    cat.fit(list(X[col].values.astype('str')) )

    X[col] = cat.transform(list(X[col].values.astype('str')))
from sklearn.model_selection import *

from sklearn.metrics import accuracy_score, precision_score, recall_score

# create XGBClassifier instance



X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, train_size=0.66, random_state=42)

# X_valid, X_test, y_valid, y_test = train_test_split(X_val, y_val, train_size=0.2, 

#                                                           random_state=42)



weights = (train_y == 0).sum() / (1.0 * (train_y == 1).sum())

clf_XGB = XGBClassifier(learning_rate=0.06,

                    n_estimators=300,

                    max_depth=7,

                    scale_pos_weight=weights,

                    # n_jobs=-1,

                    subsample=0.8,

                    colsample_bytree=0.8,

                    objective='binary:logistic',

                    seed=27)

clf_XGB.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error", eval_set=[(X_val, y_val)], verbose=True)

y_tr_pred = clf_XGB.predict(X_train)

y_pred = clf_XGB.predict(X_val)

# y_pred_test = clf_XGB.predict(X_test)

print("train accuracy,recall and precision:", accuracy_score(y_train, y_tr_pred),

  recall_score(y_train, y_tr_pred),

  precision_score(y_train, y_tr_pred))



print("validation accuracy,recall and precision:", accuracy_score(y_val, y_pred), recall_score(y_val, y_pred),

  precision_score(y_val, y_pred))



# print("test accuracy,recall and precision:", accuracy_score(y_test, y_pred_test), recall_score(y_test, y_pred_test),

#   precision_score(y_test, y_pred_test))



# set hypermeters and the below values are trained in order to run fast

# grid_param = {"learning_rate" : [0.1],

#               'n_estimators': [300],

#              'colsample_bytree': [0.8],

# #               'reg_alpha': [0.04]

#               }



# gd_sr = GridSearchCV(estimator=classifier,  

#                      param_grid=grid_param,

#                      scoring='roc_auc',

#                      cv=StratifiedKFold( n_splits=5, shuffle=True, random_state = 1001),

#                      n_jobs=-1,

#                     verbose=1)

# classifier.fit(train_X, train_y) 
# predict test dataset

predictions = clf_XGB.predict(test_X)
# satisfy submission format

my_submission = pd.DataFrame({'PassengerId':test_df["PassengerId"],'Survived':predictions})
# export as csv file

my_submission.to_csv("sub.csv", index=False)