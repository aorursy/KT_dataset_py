import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import missingno as msno

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()

import plotly.graph_objs as go

import plotly.express as px

import seaborn as sns

train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

sample_sub = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
train.head()
test.head()
sample_sub.head()
print('Train has {:,} rows and Test has {:,} rows.'.format(len(train), len(test)))
cols = ['image', 'ID', 'sex', 'age', 'anatomy_site', 'diagnosis', 'benign_malignant', 'target']

train.columns = cols

test.columns = cols[:5]
print(train.columns)

print(test.columns)
print(train.isnull().sum())

print(test.isnull().sum())
# msno.bar(train)



plt.style.use('seaborn-colorblind')

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



msno.bar(train, ax = ax1, color=(193/255, 53/255, 192/255), fontsize=10)

msno.bar(test, ax = ax2, color=(251/255, 0/255, 0/255), fontsize=10)



ax1.set_title('Train Missing Values Map', fontsize = 16)

ax2.set_title('Test Missing Values Map', fontsize = 16);
print(plt.style.available)
plt.style.use('seaborn-colorblind')

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))



msno.matrix(train, ax = ax1, color=(0/255, 248/255, 0/255), fontsize=10 ,sparkline=False)

msno.matrix(test, ax = ax2, color=(0/255, 0/255, 220/255), fontsize=10 ,sparkline=False)



ax1.set_title('Missing Values in Training Data', fontsize = 16)

ax2.set_title('Missing Value in Testing Data', fontsize = 16);
# msno.dendrogram(train)

# plt.style.use('seaborn-notebook')

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (18, 5))



msno.dendrogram(train, ax = ax1, fontsize=12)

msno.dendrogram(test, ax = ax2, fontsize=12)



# ax1.set_title('Train Missing Values Map', fontsize = 16)

# ax2.set_title('Test Missing Values Map', fontsize = 16);

train.nunique()
# train['anatomy_site'].value_counts().sort_index().plot.bar()



print(train['anatomy_site'].value_counts())

print(train['age'].value_counts())

print(train['sex'].value_counts())
# train['age'].value_counts().sort_index().plot.bar()

plt.style.use('dark_background')

plt.figure(figsize=(10,8))  

# sns.set(style="darkgrid")

ax = sns.countplot(x = train['age'])
## Can use this as well to plot. 

# train['sex'].value_counts().sort_index().plot.bar()





## Alternate and better plot:



# plt.figure(figsize=(10,8))  

sns.set(style="darkgrid")

plt.style.use('seaborn-notebook')

ax = sns.countplot(x = train['sex'])
print(plt.style.available)
plt.figure(figsize=(10,8))

sns.set(style="darkgrid")

plt.style.use('seaborn-notebook')

ax = sns.countplot(y = train['anatomy_site'],facecolor=(0, 0, 0, 0),

                   linewidth=5,

                   edgecolor=sns.color_palette("dark", 3))
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))



sns.set(style="darkgrid")

a = sns.countplot(x = train['anatomy_site'], ax = ax1)

b = sns.countplot(x = test['anatomy_site'], ax = ax2)



a.set_xticklabels(a.get_xticklabels(), rotation=45, ha="right")

b.set_xticklabels(b.get_xticklabels(), rotation=45, ha="right")



ax1.set_title('Anatomy Site Distribution in Training Data', fontsize = 16)

ax2.set_title('Anatomy Site Distribution in Testing Data', fontsize = 16);
plt.figure(figsize = (10,8))

plt.style.use('fivethirtyeight')

ax = sns.countplot(x= "anatomy_site", hue="sex", data=train)
plt.style.use('dark_background')

plt.figure(figsize = (10,8))

ax = sns.countplot(x= "sex", hue= "age", data=train)
plt.figure(figsize = (10,8))

ax = sns.countplot(x = "target", hue = "sex", data = train)
train['age'].median()
train['age'].fillna(50,inplace = True) 

train['sex'].fillna('male', inplace = True) 

train['anatomy_site'].fillna('torso', inplace = True) 

test['anatomy_site'].fillna('torso', inplace = True)



print(train.isnull().sum())

print(test.isnull().sum())
print(train['target'].value_counts())



plt.style.use('dark_background')

plt.figure(figsize=(10,8))  

# sns.set(style="darkgrid")

sns.countplot(x = train['target'])
# x = dict(train['diagnosis'].value_counts())

# print(x)

# print(x.keys())





plt.figure(figsize=(10,8))  

sns.set(style="darkgrid")

a = sns.countplot(x = train['diagnosis'])

a.set_xticklabels(a.get_xticklabels(), rotation=45, ha="right")

# b.set_xticklabels(b.get_xticklabels(), rotation=35, ha="right")

y = train['target']

plt.style.use('dark_background')

# plt.style.use('seaborn-paper')

a = y.value_counts().plot.bar()

a.set_xticklabels(a.get_xticklabels(), rotation=0, ha="right")
from sklearn.preprocessing import LabelEncoder



# Drop the unwanted columns

train = train.drop(['image','ID','diagnosis','benign_malignant'],axis=1)



# Label Encode categorical features

train['age'].fillna(50,inplace = True) 

train['sex'].fillna('male', inplace = True) 

train['anatomy_site'].fillna('torso', inplace = True) 



le_sex = LabelEncoder()

le_site = LabelEncoder()

train.sex = le_sex.fit_transform(train.sex)



## Getting dummies for anatomy_site

x = pd.get_dummies(train['anatomy_site'], drop_first = True)



## Concat dummies and actual data.

train_x = pd.concat([train,x], axis = 1)

train_x.head()
from imblearn.under_sampling import NearMiss



X = train_x.drop(['target','anatomy_site'], axis = 1)

n = NearMiss()

X_new,y_new = n.fit_sample(X,y)
plt.style.use('dark_background')

# plt.style.use('seaborn-paper')

a = y_new.value_counts().plot.bar()

a.set_xticklabels(a.get_xticklabels(), rotation=0, ha="right")
from imblearn.combine import SMOTETomek

smk = SMOTETomek(random_state=42)

X_new_over,y_new_over = smk.fit_sample(X,y)
plt.style.use('dark_background')

# plt.style.use('seaborn-paper')

a = y_new_over.value_counts().plot.bar()

a.set_xticklabels(a.get_xticklabels(), rotation=0, ha="right")