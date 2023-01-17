import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import StratifiedKFold

import gc

from sklearn import metrics, preprocessing

import scipy
train=pd.read_csv("../input/cat-in-the-dat-ii/train.csv")
test=pd.read_csv("../input/cat-in-the-dat-ii/test.csv")
train_le=len(train)
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
import seaborn as sns
sns.countplot(x='target',data=train)
def summary(df):

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name', 'dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

    return summary

summary(train)
def summary(df):

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name', 'dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

    return summary

summary(test)
plt.figure(figsize=(20,5))

sns.countplot(x='month',data=train)
plt.figure(figsize=(20,5))

sns.countplot(x='nom_0',data=train)
data = pd.concat([train, test]).reset_index(drop=True)
# converting bin_3&bin_4 values into numerical binary features

data['bin_3']=data['bin_3'].replace({'F':0,'T':1})

data['bin_4']=data['bin_4'].replace({'N':0,'Y':1})
numerical_features = [

    'bin_0', 'bin_1', 'bin_2','bin_3', 'bin_4',

    'ord_0',

    'day', 'month']



string_features = [

    'ord_1', 'ord_2',

    'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
for i in numerical_features:

    print(data[i].mean())
for i in numerical_features:

    print(data[i].median())
for j in string_features:

    print(j,data[j].mode())
plt.figure(figsize=(20,5))

sns.countplot(x='ord_1',data=data)
plt.figure(figsize=(20,5))

sns.countplot(x='ord_2',data=data)
plt.figure(figsize=(20,5))

sns.countplot(x='ord_3',data=data)
plt.figure(figsize=(20,5))

sns.countplot(x='ord_4',data=data)
plt.figure(figsize=(20,5))

sns.countplot(x='ord_5',data=data)
ord_feat=['ord_3','ord_4','ord_5']
for i in numerical_features:

    data[i]=data[i].fillna(data[i].mean())




data[string_features]=data[string_features].fillna(data.mode().iloc[0])
data[ord_feat]=data[ord_feat].fillna(data.mode().iloc[0])
nom_feat1=['nom_0','nom_1','nom_2','nom_3','nom_4']
data.head()
train=data[:train_le]
test=data[train_le:]
train=train.drop('id',axis=1)
test=test.drop('id',axis=1)
y_train = train['target']

x_train = train.drop('target', axis=1)

x_test = test.drop('target',axis=1)
from sklearn.preprocessing import LabelEncoder

for cols in ord_feat:

    if train[cols].dtype==np.number:

        continue

    train[cols]=LabelEncoder().fit_transform(train[cols])
for cols in ord_feat:

    if test[cols].dtype==np.number:

        continue

    test[cols]=LabelEncoder().fit_transform(test[cols])
oe_features = [

    'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',

    'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',

    'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5','day', 'month'

]



ohe_features = oe_features



target_features = [

    'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

]
from sklearn.preprocessing import OneHotEncoder



ohe = OneHotEncoder(dtype='uint16', handle_unknown="ignore")

ohe_x_train = ohe.fit_transform(x_train[ohe_features])

ohe_x_test = ohe.transform(x_test[ohe_features])
from sklearn.preprocessing import OrdinalEncoder



oe = OrdinalEncoder()

oe_x_train = oe.fit_transform(x_train[oe_features])

oe_x_test = oe.transform(x_test[oe_features])
from sklearn.model_selection import StratifiedKFold
def transform(transformer, x_train, y_train, cv):

    oof = pd.DataFrame(index=x_train.index, columns=x_train.columns)

    for train_idx, valid_idx in cv.split(x_train, y_train):

        x_train_train = x_train.loc[train_idx]

        y_train_train = y_train.loc[train_idx]

        x_train_valid = x_train.loc[valid_idx]

        transformer.fit(x_train_train, y_train_train)

        oof_part = transformer.transform(x_train_valid)

        oof.loc[valid_idx] = oof_part

    return oof
from category_encoders import TargetEncoder

target = TargetEncoder(drop_invariant=True, smoothing=0.2)



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

target_x_train = transform(target, x_train[target_features], y_train, cv).astype('float')



target.fit(x_train[target_features], y_train)

target_x_test = target.transform(x_test[target_features]).astype('float')
x_train = scipy.sparse.hstack([ohe_x_train, target_x_train]).tocsr()

x_test = scipy.sparse.hstack([ohe_x_test,  target_x_test]).tocsr()
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(C=0.54321, solver='lbfgs', max_iter=10000)

logit.fit(x_train, y_train)

y_pred_logit = logit.predict_proba(x_test)[:, 1]
sub=pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')
sub.head()
test1=pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
data_to_submit = pd.DataFrame({

    'id':test1['id'],

    'target':y_pred_logit

})

data_to_submit.to_csv('categorical.csv', index = False)
data_to_submit.head()