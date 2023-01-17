#importing dependencies

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import math, random, time, datetime

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from catboost import CatBoostClassifier, Pool, cv



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv') #importing training datasetr

test = pd.read_csv('../input/test.csv') #importing testing dataset

gender_submission = pd.read_csv('../input/gender_submission.csv') #example of what a submission should look like

#view the training data

train.head(15)
train.Age.plot.hist()
#view test dataset

test.head()
#view example submission 

gender_submission.head()
train.describe()
#plot graph of missing values

missingno.matrix(train, figsize = (30, 10))
df_bin = pd.DataFrame() #for discretised continuous variables

df_con = pd.DataFrame() #for continuous variables
#number of survivors

fig = plt.figure(figsize=(20,1))

sns.countplot(y='Survived', data=train)

print(train.Survived.value_counts())
df_bin['Survived'] = train['Survived']

df_con['Survived'] = train['Survived']
#pclass: ticket class of passenger

#1 = 1st, 2 = 2nd, 3 = 3rd



sns.distplot(train.Pclass)

#print number of null values

train.Pclass.isnull().sum()
df_bin['Pclass'] = train['Pclass']

df_con['Pclass'] = train['Pclass']
#sex distribution

plt.figure(figsize=(20, 5))

sns.countplot(y="Sex", data=train)



#print no of null values

train.Sex.isnull().sum()
df_bin['Sex'] = train['Sex']

# change row values to 0 for male and 1 for female

df_bin['Sex'] = np.where(df_bin['Sex'] == 'female', 1, 0)



df_con['Sex'] = train['Sex']
#Sex data compared to Survival

fig = plt.figure(figsize=(10, 10))

sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'label': 'Survived'})

sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'label': 'Did not survive'})
def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):

    """

    Function to plot counts and distibutions of a local variable and target variable side by side

    """

    

    if use_bin_df:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=bin_df)

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], kde_kws={'label': 'Survived'})

        sns.distplot(data.loc[data[label_column] == 0][target_column], kde_kws={'label': 'Did not survive'})

        

    else:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=data)

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], kde_kws={'label': 'Survived'})

        sns.distplot(data.loc[data[label_column] == 0][target_column], kde_kws={'label': 'Did not Survive'})
#number of Siblings/Spouses

print(train.SibSp.isnull().sum())

print(train.SibSp.value_counts())
df_bin['SibSp'] = train['SibSp']

df_con['SibSp'] = train['SibSp']
#counts of SibSp and distribution of values and survived



plot_count_dist(train, bin_df=df_bin, label_column='Survived', target_column='SibSp', figsize=(20, 10))

#number of Parents/Children

df_bin['Parch'] = train['Parch']

df_con['Parch'] = train['Parch']
#counts of Parch and distribution of values against survived



plot_count_dist(train, bin_df=df_bin, label_column='Survived', target_column='Parch', figsize=(20, 10))

#ticket fare

print(train.Fare.isnull().sum())

print('Unique Fare values:', len(train.Fare.unique()))
sns.countplot(y='Fare', data=train)
df_con['Fare'] = train['Fare']

df_bin['Fare'] = pd.cut(train['Fare'], bins=5) #discretised

df_bin.Fare.value_counts()
#Fare bin counts as well as Fare distribution against Survived

plot_count_dist(data=train, bin_df=df_bin, label_column='Survived', target_column='Fare', figsize=(20, 10), use_bin_df=True)
#Embarked

print(train.Embarked.isnull().sum())

print(train.Embarked.value_counts())
sns.countplot(y='Embarked', data=train)
df_bin['Embarked'] = train['Embarked']

df_con['Embarked'] = train['Embarked']
#drop null values

df_con = df_con.dropna(subset=['Embarked'])

df_bin = df_bin.dropna(subset=['Embarked'])
#One-hot encode binned variables

one_hot_cols = df_bin.columns.tolist()

one_hot_cols.remove('Survived')

df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)



df_bin_enc.head()
#One-hot encode the categorical columns

df_embarked_one_hot = pd.get_dummies(df_con['Embarked'],prefix='embarked')

df_sex_one_hot = pd.get_dummies(df_con['Sex'], prefix='sex')

df_pclass_one_hot = pd.get_dummies(df_con['Pclass'], prefix='pclass')
#combine the encoded columns with df_con_enc

df_con_enc = pd.concat([df_con, df_embarked_one_hot, df_sex_one_hot, df_pclass_one_hot], axis=1)



#drop the original categorical columns

df_con_enc = df_con_enc.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
df_con_enc.head()
#select dataframe for predictions

selected_df = df_con_enc

selected_df.head()
#split dataframe into data and labels

X_train = selected_df.drop('Survived', axis=1) #data

y_train = selected_df.Survived #labels



print(X_train.shape, y_train.shape)
#CatBoost Algorithm

#define categorical data for Catboost model

cat_features = np.where(X_train.dtypes != np.float)[0]

cat_features
#using CatBoost Pool() function to pool together training data and categorical feature labels

train_pool = Pool(X_train, y_train, cat_features)
#CatBoost model deifnition

catboost_model = CatBoostClassifier(iterations=1000, custom_loss=['Accuracy'], loss_function='Logloss')



#Fit CatBoost model

catboost_model.fit(train_pool, plot=True)



#Accuracy

acc_catboost =  round(catboost_model.score(X_train, y_train) * 100, 2)
#Perform CatBoost cross validation



#set params for cross-validation 

cv_params = catboost_model.get_params()



#run cross validation for 10 folds

cv_data = cv(train_pool, cv_params, fold_count=10,plot=True)



#CatBoost CV results, save into a dataframe

acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)
print("CatBoost Metrics")

print("Accuracy:", acc_catboost)

print("Accuracy cross validation 10 fold:", acc_cv_catboost)
#Feature Importance

def feature_importance(model, data):

    

    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': data.columns})

    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]

    fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))

    return fea_imp
feature_importance(catboost_model, X_train)
#Submission



X_train.head()
test.head()
#One-hot-encode columns in test dataframe



test_embarked_one_hot = pd.get_dummies(test['Embarked'], prefix='embarked')

test_sex_one_hot = pd.get_dummies(test['Sex'], prefix='sex')

test_pclass_one_hot = pd.get_dummies(test['Pclass'], prefix='pclass')
#combine the encoded columns with test

test = pd.concat([test, test_embarked_one_hot, test_sex_one_hot, test_pclass_one_hot], axis=1)



test.head()
#Create a list of columns to be used for predictions

wanted_cols = X_train.columns

wanted_cols
predictions = catboost_model.predict(test[wanted_cols])
predictions[:20]
submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = predictions
submission['Survived'] = submission['Survived'].astype(int)

submission.head()
submission.to_csv('submit.csv', index=False)
submission_check = pd.read_csv('submit.csv')

submission_check.head()