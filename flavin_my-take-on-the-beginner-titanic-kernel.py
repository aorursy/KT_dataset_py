import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
# Read train and test data into pandas dataframes
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Take a look at the training data
train.sample(5)
train.describe()
import category_encoders as ce

ohe = ce.one_hot.OneHotEncoder(cols=['Sex'], handle_unknown='ignore', use_cat_names=True)
train_basic = ohe.fit_transform(train[['Pclass', 'Age', 'Sex']])

# If this were our actual model for submission, we would transform the test data as well
# test_basic = ohe.transform(test[['Pclass', 'Age', 'Sex']])

train_basic.head()
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
train_basic = imp.fit_transform(train_basic)
# test_basic = imp.transform(test_basic)
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(train_basic, train['Survived'], random_state=43210)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.score(X_validation, y_validation)
train['Cabin'].describe()
train['Cabin'].unique()
train[(train['Cabin'] == 'T') | (train['Cabin'] == 'B51 B53 B55') | (train['Cabin'] == 'F E69') | (train['Cabin'] == 'F G73')]
import re

singleLetterRe = re.compile(r"^[A-Z]$") # This will clean the weird 'T' value
cabinRe = re.compile(r"^([A-Z] )?([A-Z])\d+.*$")
decks = dict(zip('ABCDEFG', range(7, 0, -1)))

# First, make the numeric deck column for train and test, preserving nan
for df in (train, test):
    df['Deck'] = (df['Cabin'].replace(singleLetterRe, np.nan)
                  .replace(cabinRe, '\\2')
                  .map(decks, na_action='ignore'))

# Next, fill in missing deck values
# We group decks by pclass and take the mean, then fill all the missing values
# with the mean for their plass
deckmeans = train[['Pclass', 'Deck']].groupby('Pclass')['Deck'].mean()
for pclass in 1,2,3:
    train.loc[(train['Pclass'] == pclass) & (train['Deck'].isna()), 'Deck'] = deckmeans[pclass]
    test.loc[(test['Pclass'] == pclass) & (test['Deck'].isna()), 'Deck'] = deckmeans[pclass]
print(train.groupby('Deck')['Deck'].count())

plt.figure(figsize=(14,6))
sns.barplot(x="Deck", y="Survived", data=train, ax=plt.gca());
for df in train, test:
    df['cabin_was_recorded'] = ~df['Cabin'].isna()
sns.barplot(x='cabin_was_recorded', y='Survived', data=train);
# This process of splitting gets us the word immediately before a '.'
for df in train,test:
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train.groupby('Title')['Title'].count()
plt.figure(figsize=(18,6));
sns.barplot(x='Title', y='Survived', data=train, ax=plt.gca());
fig, axes = plt.subplots(2, 2, figsize=(18,6))
bins = range(0, 70, 5)
for title, ax in zip(('Master', 'Miss', 'Mr', 'Mrs'), axes.flatten()):
    sns.distplot(train.loc[train['Title']==title, 'Age'].dropna(), bins=bins, kde=False, ax=ax, label=title)
    ax.legend()
titleagemeans = train[['Title', 'Age']].groupby('Title')['Age'].mean()
for title in train['Title'].unique():
    if titleagemeans[title] == np.nan:
        # If, say, one of the rare titles is missing all age values,
        # its mean will still be nan.
        # Skip it for now. We can check later if there are 
        # still nan values for Age to fill in
        continue
    train.loc[(train['Title'] == title) & (train['Age'].isna()), 'Age'] = titleagemeans[title]
    test.loc[(test['Title'] == title) & (test['Age'].isna()), 'Age'] = titleagemeans[title]    
# Now sweep up any values left over
imp = SimpleImputer(strategy='mean')
train['Age'] = imp.fit_transform(train['Age'].values.reshape(-1, 1))
test['Age'] = imp.transform(test['Age'].values.reshape(-1, 1))
np.any(train['Age'].isna())
binsize = 16
for df in train, test:
    df['Age band'] = (df['Age'] - df['Age'].mod(binsize)).div(binsize).astype(int)
train['Age band'].value_counts().to_frame()
for df in train, test:
    df['Alone'] = 0
    df.loc[(df['SibSp'] == 0) & (df['Parch'] == 0), 'Alone'] = 1
sns.barplot(x='Alone', y='Survived', hue='Sex', data=train);
g = sns.FacetGrid(train, col='Alone', row='Sex', margin_titles=True, size=5)
g.map(sns.barplot, 'Age band', 'Survived');
for df in train, test:
    df['Alone_male'] = 0
    df.loc[(df['Alone'] == 1) & (df['Sex'] == 'male'), 'Alone_male'] = 1
    
    df['Accompanied_female_age_band'] = -1
    accompanied_females = (df['Alone'] == 0) & (df['Sex'] == 'female')
    df.loc[accompanied_females, 'Accompanied_female_age_band'] = df.loc[accompanied_females, 'Age band']

fig, axes = plt.subplots(1, 2, figsize=(14,6))
sns.barplot(x='Alone_male', y='Survived', data=train, ax=axes[0]);
sns.barplot(x='Accompanied_female_age_band', y='Survived', data=train, ax=axes[1]);
plt.figure(figsize=(10,6))
sns.distplot(train['Fare'], ax=plt.gca());
train[train['Fare'] > 500]
fig, axes = plt.subplots(1, 3, figsize=(18,6))

for i in range(3):
    sns.distplot(train.loc[train['Pclass'] == i+1, 'Fare'], ax=axes[i])
    axes[i].set_title('Fares in Pclass {}'.format(i+1));
drop_cols = ['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title']
test_ids = test['PassengerId']
survived = train['Survived']
train = train.drop(drop_cols + ['Survived'], axis=1)
test = test.drop(drop_cols, axis=1)
ohe = ce.one_hot.OneHotEncoder(cols=['Sex'], handle_unknown='ignore', use_cat_names=True)
train = ohe.fit_transform(train)
test = ohe.transform(test)
train.head()
X_train, X_validation, y_train, y_validation = train_test_split(train, survived, random_state=43210)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.score(X_validation, y_validation)
X_train, X_validation, y_train, y_validation = train_test_split(train[['Pclass', 'Age band', 'Sex_male', 'Sex_female']], survived, random_state=43210)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.score(X_validation, y_validation)
import xgboost as xg
from sklearn.model_selection import cross_val_score

xgb = xg.XGBClassifier(n_estimators=900, learning_rate=0.1)
result=cross_val_score(xgb, train, survived, cv=5, scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())
xgb.fit(train, survived)
predictions = xgb.predict(test)
results = pd.DataFrame()
results['PassengerId'] = test_ids
results['Survived'] = predictions
results.head()
results.to_csv('results.csv', index=False)
