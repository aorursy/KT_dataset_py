# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bokeh.plotting as bkp  # for nice plotting

import bokeh.charts as bkc  # for nice plotting

import bokeh.models as bkm  # for nice plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/train.csv')

df.head()
df['Survived'].value_counts(dropna = False)
# Pclass 

survived_pclass = df.groupby('Pclass')['Survived'].value_counts().unstack()

survived_pclass['Rate'] = survived_pclass[1]/(survived_pclass[1] + survived_pclass[0])

survived_pclass
bkp.output_notebook()

bar1 = bkc.Bar(df, values = 'Survived', label = 'Pclass', agg = 'count',

            tools='pan,box_zoom,reset,resize,save,hover', 

               stack=bkc.attributes.cat(columns='Survived', sort=False), 

            legend='top_left', plot_width=600, plot_height=300)

hover = bar1.select(dict(type = bkm.HoverTool))

hover.tooltips = dict([("Num", "@height{int}")])

bar1.yaxis.axis_label = 'Number of passengers'

bkp.show(bar1)
# Name

import re

title = df['Name'].map(lambda x: re.split('[,.]', x)[1].strip())

df['Title'] = title

survived_title = df['Survived'].groupby(df['Title']).value_counts().unstack()

survived_title.fillna(0, inplace=True)

survived_title['Rate'] = survived_title[1]/survived_title.sum(axis=1)

survived_title.sort_values(by='Rate', ascending=False, inplace=True)

survived_title
# gender (or sex)

survived_sex = df.groupby('Sex')['Survived'].value_counts().unstack()

survived_sex['Rate'] = survived_sex[1]/(survived_sex.sum(axis=1))

survived_sex
# age histogram of survivors

survived_age = df[['Survived', 'Age', 'Sex']].copy()

survived_age['Survived'] = survived_age['Survived'].astype(int)

print('Total number of NAs in Age: {}'.format(survived_age['Age'].isnull().sum()))

survived_age.dropna(inplace=True)

hist1 = bkc.Histogram(survived_age, values = 'Age', color = 'Sex', bins = 50,

                     plot_width=600, plot_height=300)

bkp.show(hist1)
# SibSp and Parch

survived_sibsp = df['Survived'].groupby(df['SibSp']).value_counts().unstack()

survived_sibsp.fillna(0, inplace=True)

survived_sibsp['Rate'] = survived_sibsp[1]/survived_sibsp.sum(axis=1)

survived_sibsp.sort_values(by='Rate', ascending=False, inplace=True)

print(survived_sibsp)

print('Total number of NAs in SibSp: {}'.format(df['SibSp'].isnull().sum()))



# Parch

survived_parch = df['Survived'].groupby(df['Parch']).value_counts().unstack()

survived_parch.fillna(0, inplace=True)

survived_parch['Rate'] = survived_parch[1]/survived_parch.sum(axis=1)

survived_parch.sort_values(by='Rate', ascending=False, inplace=True)

print('\n', survived_parch)

print('Total number of NAs in Parch: {}'.format(df['Parch'].isnull().sum()))



# family size

df['Family Size'] = df['SibSp'] + df['Parch']

survived_family = df['Survived'].groupby(df['Family Size']).value_counts().unstack()

survived_family.fillna(0, inplace=True)

survived_family['Rate'] = survived_family[1]/survived_family.sum(axis=1)

survived_family.sort_values(by='Rate', ascending=False, inplace=True)

print('\n', survived_family)
# Fare

p = bkc.Scatter(df, x = 'Fare', y = 'Age', color = 'Survived',

                plot_width = 700, plot_height = 500, legend = 'top_right')

bkp.show(p)
# cabin

print('Total number of non-NAs in Cabin: {}'.format(df['Cabin'].notnull().sum()))

print('Total number of NAs in Cabin: {}'.format(df['Cabin'].isnull().sum()))

cabin = df[['Survived', 'Cabin']].copy()

cabin.dropna(inplace=True)

def find_num(x):

    result = re.search('([0-9]+)', x)

    if result:

        return result.group()

    else:

        return '0'

cabin['Header'] = cabin['Cabin'].map(lambda x: re.findall('[A-Z]', x)[0])

cabin['Number'] = cabin['Cabin'].map(find_num)

survived_cabin_h = cabin['Survived'].groupby(cabin['Header']).value_counts().unstack()

survived_cabin_h.fillna(0, inplace=True)

survived_cabin_h['Rate'] = survived_cabin_h[1]/survived_cabin_h.sum(axis=1)

survived_cabin_h.sort_values(by='Rate', inplace=True, ascending=False)

print(survived_cabin_h)
# Embarked

survived_embarked = df['Survived'].groupby(df['Embarked']).value_counts().unstack()

survived_embarked.fillna(0, inplace=True)

survived_embarked['Rate'] = survived_embarked[1]/survived_embarked.sum(axis=1)

survived_embarked.sort_values(by='Rate', ascending=False, inplace=True)

print(survived_embarked)
# how many na values for each column?

print(df.isnull().sum())

df_clean = df.copy()
age_mean = df[df['Age'].notnull()]['Age'].mean()

df_clean.loc[df['Age'].isnull(), 'Age'] = age_mean
bins = [0, 20, 40, 60, 80, 100]

df_clean['Age range'] = pd.cut(df_clean['Age'], bins, labels=False)
df[df['Cabin'].isnull()]['Survived'].value_counts()
df_clean.loc[df_clean['Cabin'].isnull(), 'Cabin'] = 'X000'

df_clean['Cabin_h'] = df_clean['Cabin'].map(lambda x: x[0])
df_clean.loc[df['Embarked'].isnull(), 'Embarked'] = 'S'
df_clean.isnull().sum()
df_clean.dtypes
df_clean['Cabin_h'].value_counts()

df_clean['Title'].value_counts()
# group titles

def group_title(x):

    if x not in ['Mr', 'Miss', 'Mrs']:

        return 'Other'

    else:

        return x

df_clean['Title'] = df_clean['Title'].map(group_title)
df['Survived'].groupby([df['Sex'], df['Survived']]).count().unstack()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

# make a copy, for no reason...

df_clean_run = df_clean.copy()

# do the OneHot encoding

# the added derived features are 'Age range', 'Family Size', 'Cabin_h', 'Title'

df_clean_run = pd.get_dummies(df_clean_run, columns=['Sex', 'Cabin_h', 'Pclass', 'Embarked', 'Title'])

# initilize the classifier

clf = RandomForestClassifier(n_estimators=1000, max_depth=5)

# split the training set, for x-validation

train, test = train_test_split(df_clean_run, test_size = 0.2)

features = train.columns.tolist()

remove_list = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 

               'Age range', 'SibSp', 'Parch']

for item in remove_list:

    features.remove(item)

print(features, '\n')

clf.fit(train[features], train['Survived'])

importances = [(f, i) for f, i in zip(features, clf.feature_importances_)]

importances.sort(key = lambda x: x[1], reverse=True)

#for f, i in importances:

#    print('Importance: {:>10}:{:4.3f}'.format(f, i))

print('\nTraining Accurancy: {:<30}'.format(clf.score(train[features], train['Survived'])))

print('Test Accurancy: {:<30}'.format(clf.score(test[features], test['Survived'])))
test_df = pd.read_csv('../input/test.csv')

test_df_clean = test_df.copy()

# preprocessing

# fill missing values

test_df_clean.isnull().sum()

age_mean = test_df[test_df['Age'].notnull()]['Age'].mean()

test_df_clean.loc[test_df['Age'].isnull(), 'Age'] = age_mean

test_df_clean.loc[test_df_clean['Cabin'].isnull(), 'Cabin'] = 'X000'

test_df_clean['Cabin_h'] = test_df_clean['Cabin'].map(lambda x: x[0])

test_df_clean.loc[test_df_clean['Fare'].isnull(), 'Fare'] = test_df[test_df['Fare'].notnull()]['Fare'].mean()
# 2. Add derived features

# the added derived features are 'Age range', 'Family Size', 'Cabin_h', 'Title'

test_df_clean['Age range'] = pd.cut(test_df_clean['Age'], bins, labels=False)

test_df_clean['Family Size'] = test_df_clean['SibSp'] + test_df_clean['Parch']

test_df_clean['Cabin_h'] = test_df_clean['Cabin'].map(lambda x: x[0])

test_df_clean['Title'] = test_df_clean['Name'].map(lambda x: re.split('[,.]', x)[1].strip())

test_df_clean['Title'] = test_df_clean['Title'].map(group_title)
# OneHot encoding

test_df_clean = pd.get_dummies(test_df_clean, columns=['Sex', 'Cabin_h', 'Pclass', 'Embarked', 'Title'])

# check for possible missing features

for fe in features:

    if fe not in test_df_clean.columns.tolist():

        test_df_clean[fe] = 0.0
# predict!

output = clf.predict(test_df_clean[features])
df_submit = test_df_clean[['PassengerId']].copy()

df_submit['Survived'] = pd.Series(output)

df_submit.to_csv('output.csv', index=False)