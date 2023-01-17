# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit

from xgboost import XGBClassifier



# Input data files are available in the "../  put/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
# Create dataframes for each

train_raw = pd.read_csv('/kaggle/input/titanic/train.csv')

test_raw = pd.read_csv('/kaggle/input/titanic/test.csv')

# Copy dataframes

train_df = train_raw.copy(deep=True)

test_df = test_raw.copy(deep=True)

data_cleaner = [train_df, test_df]

# list column indices

test_df.info()

train_df.info()
print(train_df.isnull().sum())

print(test_df.isnull().sum())
sns.boxplot(x=train_df['Pclass'], y=train_df['Age'], hue=train_df['Sex'])

plt.show()
for df in data_cleaner:

    df['Age_na'] = df['Age'].isna()

    df['Age'].fillna(df['Age'].median(), inplace=True)

    #for pc in train_df.Pclass.unique():

    #    df.loc[ (df['Age'].isna()) & (df['Sex']=='male'), 'Age'] = df.loc[(df['Sex']=='male')].Age.median()

    #    df.loc[ (df['Age'].isna()) & (df['Sex']=='female'), 'Age'] = df.loc[(df['Sex']=='female')].Age.median()

    df['Embarked_na'] = df['Embarked'].isna()

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    df['Fare_na'] = df['Fare'].isna()

    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    print(df.isnull().sum())
for df in data_cleaner:

    df['Age_bin'] = pd.cut(x=df['Age'].astype(int), bins=5)

    df['Fare_bin'] = pd.qcut(x=df['Fare'], q=4)

    print(df.sample(10))
for df in data_cleaner:

    # The titles may affect the survivale rate, i.e. doctors may be of higher priority.

    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    stat_min = 10

    title_names = (df['Title'].value_counts() < stat_min)

    df['Title'] = df['Title'].apply(lambda x: 'Misc' if title_names[x] else x)

    # combination of SibSp and Parch

    df['FamilySize'] = df['SibSp'] + df['Parch']

    # whether person is alone

    df['Alone'] = df['FamilySize'].apply(lambda x: 1 if x > 0 else 0)

    # People on the lowest floor probably died first or something

    df['Level'] = df.loc[~(df['Cabin'].isnull()), 'Cabin'].str.split(" ", expand=True)[0].str[0]

    print(df['Level'].head())

    #print(df['Title'].value_counts())

    #print(df.sample(10))
print(train_df[['Pclass', 'Level']].groupby(by=['Level', 'Pclass']).Pclass.count())

print(train_df[['Level', 'Survived']].groupby(by=['Level']).Survived.mean())

print(train_df.loc[train_df['Pclass']==1, 'Level'].mode()[0])

print(train_df.loc[train_df['Pclass']==2, 'Level'].mode()[0])

print(train_df.loc[train_df['Pclass']==3, 'Level'].mode()[0])
label_encoder = LabelEncoder()

for df in data_cleaner:

    df.loc[df['Level'].isna() & (df['Pclass']==1), 'Level'] = df.loc[df['Pclass']==1, 'Level'].mode()[0]

    df.loc[df['Level'].isna() & (df['Pclass']==2), 'Level'] = df.loc[df['Pclass']==2, 'Level'].mode()[0]

    df.loc[df['Level'].isna() & (df['Pclass']==3), 'Level'] = df.loc[df['Pclass']==3, 'Level'].mode()[0]

    df['Floor'] = label_encoder.fit_transform(df['Level'])

    df['Floor_bin'] = pd.cut(x=df['Floor'].astype(int), bins=3)

    print(df[['Floor','Floor_bin']].sample(20))

    print(df['Floor'].unique())

    print(df['Level'].unique())

label_encoder = LabelEncoder()

for df in data_cleaner:

    df['Title_en'] = label_encoder.fit_transform(df['Title'])

    df['Embarked_en'] = label_encoder.fit_transform(df['Embarked'])

    df['Sex_en'] = label_encoder.fit_transform(df['Sex'])

    df['AgeBin_en'] = label_encoder.fit_transform(df['Age_bin'])

    df['FareBin_en'] = label_encoder.fit_transform(df['Fare_bin'])

    df['FloorBin_en'] = label_encoder.fit_transform(df['Floor_bin'])

    print(df.sample(5))
train_df.columns.values
y = 'Survived'

features = {

    'feat_basic' : ['Pclass', 'Sex_en', 'Age', 'Title_en', 'SibSp', 'Parch', 'Fare', 'Embarked_en'],

    'feat_family' : ['Pclass', 'Sex_en', 'Age', 'Title_en', 'FamilySize', 'Fare', 'Embarked_en'],

    'feat_alone' : ['Pclass', 'Sex_en', 'Age', 'Title_en', 'Alone', 'Fare', 'Embarked_en'],

    'feat_basic_na' : ['Pclass', 'Sex_en', 'Age', 'Age_na', 'Title_en', 'SibSp', 'Parch', 'Fare', 'Fare_na', 'Embarked_en', 'Embarked_na'],

    'feat_family_na' : ['Pclass', 'Sex_en', 'Age', 'Age_na', 'Title_en', 'FamilySize', 'Fare', 'Fare_na', 'Embarked_en', 'Embarked_na'],

    'feat_alone_na' : ['Pclass', 'Sex_en', 'Age', 'Age_na', 'Title_en', 'Alone', 'Fare', 'Fare_na', 'Embarked_en', 'Embarked_na'],

    'feat_bin' : ['Pclass', 'Sex_en', 'AgeBin_en', 'Title_en', 'SibSp', 'Parch', 'FareBin_en', 'Embarked_en'],

    'feat_bin_family' : ['Pclass', 'Sex_en', 'AgeBin_en', 'Title_en', 'FamilySize', 'FareBin_en', 'Embarked_en'],

    'feat_bin_alone' : ['Pclass', 'Sex_en', 'AgeBin_en', 'Title_en', 'Alone', 'FareBin_en', 'Embarked_en'],

    'feat_bin_na' : ['Pclass', 'Sex_en', 'AgeBin_en', 'Age_na', 'Title_en', 'SibSp', 'Parch', 'FareBin_en', 'Fare_na', 'Embarked_en', 'Embarked_na'],

    'feat_bin_family_na' : ['Pclass', 'Sex_en', 'AgeBin_en', 'Age_na', 'Title_en', 'FamilySize', 'FareBin_en', 'Fare_na', 'Embarked_en', 'Embarked_na'],

    'feat_bin_alone_na' : ['Pclass', 'Sex_en', 'AgeBin_en', 'Age_na', 'Title_en', 'Alone', 'FareBin_en', 'Fare_na', 'Embarked_en', 'Embarked_na'],

    'feat_basic_floor' : ['Pclass', 'Sex_en', 'Age', 'Title_en', 'SibSp', 'Parch', 'Fare', 'Embarked_en', 'Floor'],

    'feat_family_floor' : ['Pclass', 'Sex_en', 'Age', 'Title_en', 'FamilySize', 'Fare', 'Embarked_en', 'Floor'],

    'feat_family_na_floor' : ['Pclass', 'Sex_en', 'Age', 'Age_na', 'Title_en', 'FamilySize', 'Fare', 'Fare_na', 'Embarked_en', 'Embarked_na', 'FloorBin_en'],

    'feat_bin_family_na_floor' : ['Pclass', 'Sex_en', 'AgeBin_en', 'Age_na', 'Title_en', 'FamilySize', 'FareBin_en', 'Fare_na', 'Embarked_en', 'Embarked_na', 'FloorBin_en'],

    'feat_bin_family_floor' : ['Pclass', 'Sex_en', 'AgeBin_en', 'Title_en', 'FamilySize', 'FareBin_en', 'Embarked_en', 'FloorBin_en'],



}
corr = train_df.corr()

plt.figure(figsize=(9,9))

sns.heatmap(corr,vmax=0.9,square=True, annot=True)

plt.show()
print(train_df.groupby(by=['Pclass'])[['Pclass', 'Survived']].mean().sort_values(by='Survived', ascending=False))

sns.swarmplot(data=train_df, x='Pclass', y='Age', hue='Survived')

plt.show()
fg = sns.FacetGrid(train_df, col='Survived')

fg.map(plt.hist, 'Fare', bins=10)
train_df['AgeGroup'] = (train_df['Age'] / 10).round(0)

train_df[['AgeGroup', 'Survived']].groupby(by='AgeGroup').mean().sort_values(by='Survived', ascending=False)
train_df.drop('AgeGroup', axis=1)

sns.swarmplot(data=train_df, x='Survived', y='Age', hue='Sex')

plt.show()

train_df.groupby(by='Sex')[['Sex', 'Survived']].mean()
sibvsur = train_df.groupby(by=['SibSp'])[['SibSp', 'Survived']].mean().sort_values(by='Survived', ascending=False)

print(sibvsur)

sns.regplot(data=sibvsur, x='SibSp', y='Survived')

plt.show()
parvsur = train_df.groupby(by=['Parch'])[['Parch', 'Survived']].mean().sort_values(by='Survived', ascending=False)

print(parvsur)

sns.regplot(data=parvsur, x='Parch', y='Survived')

plt.show()
sns.swarmplot(data=train_df, x='Embarked', y='Age', hue='Survived')

plt.show()

train_df[['Embarked', 'Survived']].groupby(by='Embarked').mean()
train_x = train_df[features['feat_bin_family']]

train_y = train_df[y]

test_x = test_df[features['feat_bin_family']]

rfr_model = XGBClassifier()

rfr_model.fit(train_x, train_y)

result = rfr_model.predict(test_x)
output = pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':result})

output.to_csv('predictions2.csv', index=False)