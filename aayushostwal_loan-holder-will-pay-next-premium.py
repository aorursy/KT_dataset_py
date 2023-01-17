#importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train_qnU1GcL (1).csv')

test = pd.read_csv('../input/test_LxCaReE_DvdCKVT_v2s0Z4r.csv')
## Combining the train and test dataset for EDA

df = pd.concat([train, test], axis=0)
train.head()
print(train.shape)

print(test.shape)
train.info()
train.describe()
sns.distplot(df['perc_premium_paid_by_cash_credit'], bins = 80)
sns.distplot(df['age_in_days']//365, bins=90)

plt.xticks(range(0,100,5))
sns.boxplot(y = df['age_in_days']//365)

plt.yticks(range(0,110,10))
sns.distplot(df['Income'])
sns.boxplot(y = df['Income'])
sns.distplot(df['application_underwriting_score'])
plt.figure(figsize=(12,4))

sns.countplot(df['Count_3-6_months_late'])
plt.figure(figsize=(12,4))

sns.countplot(df['Count_6-12_months_late'])
plt.figure(figsize=(12,4))

sns.countplot(df['Count_more_than_12_months_late'])
data = df[pd.isnull(df['Count_3-6_months_late'])]
data['no_of_premiums_paid'].value_counts()
plt.figure(figsize=(15,4))

sns.countplot(df['no_of_premiums_paid'])
sns.countplot(df['sourcing_channel'])
sns.countplot(df['residence_area_type'])
y = pd.crosstab(df['no_of_premiums_paid'], df['target'])

plt.figure(figsize=(15,4))

sns.countplot(df['no_of_premiums_paid'], hue = df['target'])
plt.figure(figsize=(15,4))

data = df[df['no_of_premiums_paid']>25]

sns.countplot(data['no_of_premiums_paid'], hue = data['target'])
sns.boxplot(x = train['target'], y = train['perc_premium_paid_by_cash_credit'])
sns.boxplot(x = train['target'], y = train['application_underwriting_score'])
sns.countplot(train['sourcing_channel'], hue = train['target'])
train.groupby('sourcing_channel')['target'].value_counts()
sns.countplot(train['residence_area_type'], hue = train['target'])
train['application_underwriting_score'].fillna(train['application_underwriting_score'].mean(), inplace = True)
test['application_underwriting_score'].fillna(train['application_underwriting_score'].mean(), inplace = True)
train['Count_more_than_12_months_late'].fillna(np.mean(train['Count_more_than_12_months_late']),inplace=True)
test['Count_more_than_12_months_late'].fillna(np.mean(test['Count_more_than_12_months_late']),inplace=True)

test['Count_6-12_months_late'].fillna(np.min(test['Count_6-12_months_late']),inplace=True)

test['Count_3-6_months_late'].fillna(np.min(test['Count_3-6_months_late']),inplace=True)
train.dropna(axis=0, inplace = True)
train.isnull().sum()
test.isnull().sum()
train['age_in_days'] = train['age_in_days']//365
test['age_in_days'] = test['age_in_days']//365
train.describe()
sns.boxplot(train['age_in_days'])

plt.xticks(range(15,115,5))
train.loc[train['age_in_days']>90, 'age_in_days'] = np.mean(train['age_in_days'])
sns.boxplot(train['no_of_premiums_paid'])
train.loc[train['no_of_premiums_paid']>25, 'no_of_premiums_paid'] = np.mean(train['no_of_premiums_paid'])
q1 = train['Income'].quantile(0.25)

q3 = train['Income'].quantile(0.75)

iqr = q3-q1

llim = q1 - 1.5*iqr

ulim = q3+1.5*iqr
train.loc[train['Income']>ulim, 'Income'] = np.mean(train['Income'])
sns.boxplot(train['application_underwriting_score'])
sns.boxplot(train['Count_3-6_months_late'])

plt.xticks(range(0,16))
train.loc[train['Count_3-6_months_late']>1, 'Count_3-6_months_late'] = np.mean(train['Count_3-6_months_late'])
train.loc[train['Count_6-12_months_late']>5, 'Count_6-12_months_late'] = np.mean(train['Count_6-12_months_late'])
train.loc[train['Count_more_than_12_months_late']>4, 'Count_more_than_12_months_late'] = np.mean(train['Count_more_than_12_months_late'])
train['Income'] = train['Income']**0.5
test['Income'] = test['Income']**0.5
train['sourcing_channel'] = train['sourcing_channel'].replace({'A':1, 'B':2,'C':3,'D':4,'E':5})

test['sourcing_channel'] = test['sourcing_channel'].replace({'A':1, 'B':2,'C':3,'D':4,'E':5})
train['residence_area_type'] = train['residence_area_type'].replace({'Urban':1, 'Rural':0})

test['residence_area_type'] = test['residence_area_type'].replace({'Urban':1, 'Rural':0})
plt.figure(figsize=(12,8))

sns.heatmap(train.corr(), cmap = 'viridis', annot = True)
test.shape, train.shape
sub = pd.DataFrame(test['id'])

train.drop('id', axis=1, inplace = True)

test.drop('id', axis=1,inplace = True)
X = train.drop('target',1)

y = train['target']
from sklearn.model_selection import train_test_split
X1_train, X1_valid, y1_train, y1_valid = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
clf1 = DecisionTreeClassifier()
clf1.fit(X1_train, y1_train)
clf1.score(X1_train, y1_train)
clf1.score(X1_valid, y1_valid)
roc_auc_score(y1_valid, clf1.predict(X1_valid))
Pre = pd.DataFrame(clf1.predict(test))
Pre.columns = ['target']
pre = pd.concat([sub, Pre], axis=1)
pre.to_csv('final_submission01.csv', index = False)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X1_train, y1_train)
roc_auc_score(y1_valid ,model.predict(X1_valid))
Pre = pd.DataFrame(model.predict(test))
Pre.columns = ['target']
pre = pd.concat([sub, Pre], axis=1)
pre.to_csv('final_submission02.csv', index = False)