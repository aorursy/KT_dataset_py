# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
gender_submission = pd.read_csv('../input/gender_submission.csv')
train_df.head()
import seaborn as sns
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train_df)
numeric_features = ['Age', 'Fare']
numeric_corr = train_df[numeric_features].corr()
sns.heatmap(numeric_corr)
categorical_features = ['Survived', 'Pclass', 'SibSp', 'Parch']
cat_corr = train_df[categorical_features].corr()
sns.heatmap(cat_corr)
sns.catplot(x='Pclass', y='SibSp', hue='Survived', data=train_df)
sns.catplot(x='Parch', y='SibSp', hue='Survived', data=train_df)
sns.catplot(x='Pclass', y='Parch', hue='Survived', data=train_df)
set(train_df['Cabin'])
from sklearn.preprocessing import LabelEncoder

sex_le = LabelEncoder()
sex_df = sex_le.fit_transform(train_df['Sex'])
embarked_le = LabelEncoder()
train_df['Embarked'].fillna('X', inplace=True)
embarked_df = embarked_le.fit_transform(train_df['Embarked'])
plot_df = pd.DataFrame({
    'survived': train_df['Survived'],
    'sex': sex_df,
    'embarked': embarked_df 
})
sns.catplot(x='embarked', y='sex', data=plot_df, hue='survived')
ord_corr = plot_df.corr()
sns.heatmap(ord_corr)
import re
from sklearn.preprocessing import LabelEncoder

def is_a_miss(name):
    if 'miss' in name.lower():
        return 1
    return 0

def get_cabin_level(cabin):
    return ord(cabin.split()[0][0]) - ord('A')
    
def get_cabin_loc(cabin):
    cabin_loc = re.sub('[A-Z]', '', cabin.split()[0])
    try:
        if cabin_loc:
            loc = int(cabin_loc)
        else:
            loc = 0
    except:
        print(cabin)
        print(cabin_loc)
        raise Exception('nope')
    return loc

def prep_features(train_X):
    features_to_keep = ['Fare', 'Age', 'Parch', 'Pclass', 'Sex', 'Name', 'Cabin']
    train_X = train_X[features_to_keep]
    train_X['Fare'].fillna(train_X['Fare'].median(), inplace=True)
    train_X['Age'].fillna(train_X['Age'].median(), inplace=True)
    le1 = LabelEncoder()
    train_X['Sex'] = le1.fit_transform(train_X['Sex'])
    train_X['Unmarried_female'] = train_X['Name'].apply(is_a_miss)
    train_X.drop('Name', axis=1, inplace=True)
    train_X['Cabin'].fillna('X', inplace=True)
    train_X['Cabin_level'] = train_X['Cabin'].apply(get_cabin_level)
    train_X['Cabin_loc'] = train_X['Cabin'].apply(get_cabin_loc)
    train_X.drop('Cabin', axis=1, inplace=True)
    return train_X

train_labels = train_df['Survived']
train_features = prep_features(train_df)
train_features.head()
import xgboost as xgb
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
gbm.fit(train_features, train_labels)
test_features = prep_features(test_df)
test_features.head()
preds = gbm.predict(test_features)
sub = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': preds
})
sub.head()
sub.to_csv('submission.csv', index=False)
