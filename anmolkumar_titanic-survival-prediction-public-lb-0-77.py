# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import re

from numpy import mean, std

from sklearn.impute import SimpleImputer

import seaborn as sns

from matplotlib import *

from matplotlib import pyplot as plt

from catboost import CatBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.base import TransformerMixin

from lightgbm import LGBMClassifier
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

sample_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train_data.columns = train_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

test_data.columns = test_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
print('Train Data Shape: ', train_data.shape)

print('Test Data Shape: ', test_data.shape)

train_data.head()
train_data.dtypes
train_data.isnull().sum()
test_data.isnull().sum()
# Unique values for all the columns

for col in train_data.columns[~(train_data.columns.isin(['age', 'passengerid', 'survived', 'name', 'ticket', 'cabin', 'fare']))].tolist():

    print(" Unique Values --> " + col, ':', len(train_data[col].unique()), ': ', train_data[col].unique())
i = 1

for column in train_data.columns[~(train_data.columns.isin(['age', 'passengerid', 'name', 'ticket', 'cabin', 'fare']))].tolist():

    plt.figure(figsize = (40, 10))

    plt.subplot(3, 3, i)

    sns.barplot(x = train_data[column].value_counts().index, y = train_data[column].value_counts())

    i += 1

    plt.show()
train_data['type'] = 'train'

test_data['type'] = 'test'

master_data = pd.concat([train_data, test_data])

#master_data = master_data.sort_values(['id', 'type'], ascending = [True, False])

master_data.head()
plt.figure(figsize = (15, 6))

sns.distplot(master_data.loc[(master_data['sex'] == 'male'), 'age'], kde_kws = {"color": "b", "lw": 1, "label": "Male"})

sns.distplot(master_data.loc[(master_data['sex'] == 'female'), 'age'], kde_kws = {"color": "r", "lw": 1, "label": "Female"})

plt.show()
plt.figure(figsize = (15, 6))

sns.distplot(master_data.loc[(master_data['sex'] == 'male'), 'fare'], kde_kws = {"color": "b", "lw": 1, "label": "Male"})

sns.distplot(master_data.loc[(master_data['sex'] == 'female'), 'fare'], kde_kws = {"color": "r", "lw": 1, "label": "Female"})

plt.show()
plt.figure(figsize = (15, 6))

sns.boxplot(x = 'sex', y = 'age', hue = 'survived', palette = ['m', 'g'], data = train_data)

plt.title('Suvival status by sex')

sns.despine(offset = 10, trim = True)
plt.figure(figsize = (15, 6))

sns.distplot(train_data.loc[(train_data['survived'] == 0), 'age'], kde_kws = {"color": "b", "lw": 1, "label": "Not survived"})

sns.distplot(train_data.loc[(train_data['survived'] == 1), 'age'], kde_kws = {"color": "r", "lw": 1, "label": "Survived"})

plt.title('Survived Age Density')

sns.despine()
# Getting titles from names 



master_data['title']=master_data['name'].map(lambda x:x.split(',')[1].split('.')[0].strip())

master_data['title'].value_counts()



TitleDict={}

TitleDict['Mr']='Mr'

TitleDict['Mlle']='Miss'

TitleDict['Miss']='Miss'

TitleDict['Master']='Master'

TitleDict['Jonkheer']='Master'

TitleDict['Mme']='Mrs'

TitleDict['Ms']='Mrs'

TitleDict['Mrs']='Mrs'

TitleDict['Don']='Royalty'

TitleDict['Sir']='Royalty'

TitleDict['the Countess']='Royalty'

TitleDict['Dona']='Royalty'

TitleDict['Lady']='Royalty'

TitleDict['Capt']='Officer'

TitleDict['Col']='Officer'

TitleDict['Major']='Officer'

TitleDict['Dr']='Officer'

TitleDict['Rev']='Officer'



master_data['title'] = master_data['title'].map(TitleDict)
master_data['travelled_alone'] = np.where(master_data['sibsp'] + master_data['parch'] == 0, 'Y', 'N')

#master_data['family_name'] = master_data['name'].apply(lambda x: x.split(',')[0].lower())

master_data['cabin'] = master_data['cabin'].fillna('U')

master_data['cabin_class'] = master_data['cabin'].apply(lambda x: x.split()[0].lower()[0])

columns = ['cabin', 'cabin_class']

master_data[columns] = master_data[columns].replace({'U': np.nan, 'u': np.nan})

master_data['ticket'] = master_data['ticket'].apply(lambda x: re.sub(r'\W+', '', x))

master_data['ticket'] = master_data['ticket'].str.replace('\d+', '')

master_data['ticket'] = master_data['ticket'].str.replace(r'^\s*$', 'GEN', regex = True)

master_data.loc[master_data['ticket'].str.startswith('STO'), 'ticket'] = master_data.loc[master_data['ticket'].str.startswith('STON'), 'ticket'] = 'Z'

master_data.loc[master_data['ticket'].str.startswith('SOT'), 'ticket'] = master_data.loc[master_data['ticket'].str.startswith('SO'), 'ticket'] = 'Z'

master_data.loc[master_data['ticket'].str.endswith('PP'), 'ticket'] = master_data.loc[master_data['ticket'].str.startswith('SC'), 'ticket'] = 'Y'

master_data.loc[master_data['ticket'].str.startswith('SC'), 'ticket'] = master_data.loc[master_data['ticket'].str.startswith('SC'), 'ticket'] = 'X'

master_data = master_data.loc[~master_data.ticket.str.contains('SP')]

master_data['ticket'] = master_data['ticket'].str[0]



master_data['is_child'] = np.where(master_data['age'] < 18, 1, 0)

master_data['is_female_child'] = np.where((master_data['age'] < 18) & (master_data['sex'] == 'female'), 1, 0)



master_data.head()
le = LabelEncoder()

cat_cols = ['pclass', 'sex', 'ticket', 'embarked', 'travelled_alone', 'cabin_class', 'title']



for col in cat_cols:

    master_data[col] = master_data[col].astype(str)

    LE = le.fit(master_data[col])

    master_data[col] = LE.transform(master_data[col])

    

train_data = master_data.loc[master_data['type'] == 'train']

test_data = master_data.loc[master_data['type'] == 'test']



testIDs = test_data.passengerid.values



train_data = train_data.drop(['passengerid', 'name', 'cabin', 'type'], axis = 1)

test_data = test_data.drop(['passengerid', 'name', 'cabin', 'type', 'survived'], axis = 1)



train_data = train_data.fillna('NaN')

test_data = test_data.fillna('NaN')



train_data['survived'] = train_data['survived'].astype(np.int8)



# Partitioning the features and the target



X = train_data[train_data.columns[~(train_data.columns.isin(['survived']))].tolist()].values

y = train_data['survived'].values



train_data.head()
kfold, scores = KFold(n_splits = 6, shuffle = True, random_state = 22), list()

for train, test in kfold.split(X):

    X_train, X_test = X[train], X[test]

    y_train, y_test = y[train], y[test]

    

    model = CatBoostClassifier(random_state = 22, max_depth = 6, n_estimators = 200, verbose = 100)

    model.fit(X_train, y_train, cat_features = [0,1,5,7,8,9,10,11,12])

    preds = model.predict(X_test)

    score = accuracy_score(y_test, preds)

    scores.append(score)

    print('Validation Accuracy:', score)

print("Average Validation Accuracy: ", sum(scores)/len(scores))
pred = pd.DataFrame()

#pred['ID'] = test_data['id'].values

pred['PassengerId'] = testIDs

pred['Survived'] = pd.Series((model.predict(test_data)).ravel())

pred.to_csv('catboost_v3.csv', index = None)

pred.head()
class DataFrameImputer(TransformerMixin):



    def __init__(self):

        """Impute missing values.



        Columns of dtype object are imputed with the most frequent value 

        in column.



        Columns of other types are imputed with mean of column.



        """

    def fit(self, X, y=None):



        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

            index=X.columns)



        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)
# Imputer for numerical age: strategy - mean, and categorical Cabin: strategy - mode



master_data = master_data.replace({'NaN': np.nan})

master_data = DataFrameImputer().fit_transform(master_data)



train_data = master_data.loc[master_data['type'] == 'train']

test_data = master_data.loc[master_data['type'] == 'test']



testIDs = test_data.passengerid.values



train_data = train_data.drop(['passengerid', 'name', 'cabin', 'type'], axis = 1)

test_data = test_data.drop(['passengerid', 'name', 'cabin', 'type', 'survived'], axis = 1)



train_data['survived'] = train_data['survived'].astype(np.int8)



# Partitioning the features and the target



X = train_data[train_data.columns[~(train_data.columns.isin(['survived']))].tolist()].values

y = train_data['survived'].values



train_data.head()
model = LGBMClassifier()

cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 22)

n_scores = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1, error_score = 'raise')

print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# fit the model on the whole dataset

model = LGBMClassifier()

model.fit(X, y)

# make a single prediction

yhat = model.predict(test_data)
pred = pd.DataFrame()

#pred['ID'] = test_data['id'].values

pred['PassengerId'] = testIDs

pred['Survived'] = pd.Series((model.predict(test_data)).ravel())

pred.to_csv('lightgbm_v3.csv', index = None)

pred.head()
model = RandomForestClassifier(n_estimators = 100)

cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 22)

n_scores = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1, error_score = 'raise')

print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# fit the model on the whole dataset

model = RandomForestClassifier()

model.fit(X, y)

# make a single prediction

yhat = model.predict(test_data)
pred = pd.DataFrame()

#pred['ID'] = test_data['id'].values

pred['PassengerId'] = testIDs

pred['Survived'] = pd.Series((model.predict(test_data)).ravel())

pred.to_csv('rf_v3.csv', index = None)

pred.head()