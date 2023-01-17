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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train.head()
train.columns
train.describe()
train.shape ,test.shape
numerical_features = train.select_dtypes(include=[np.number])

numerical_features.columns
numerical_features.head()
#list of variables that contain year information.



year_feature = [feature for feature in numerical_features

               if 'Yr' in feature or 'Year' in feature]



year_feature
# Let us explore the contents of temporal Variables.

for feature in year_feature:

    print(feature,train[feature].nunique())
for feature in year_feature:

    print(feature,train[feature].unique())
for feature in year_feature:

    if feature != 'YrSold':

        data = train.copy()

        # We will compare the difference between year variable with YrSold 

        data[feature] = data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalePrice')

        plt.show()
discrete_features = [feature for feature in numerical_features

                    if len(train[feature].unique())< 25 and feature

                    not in year_feature +['Id']]

print('Discrete Variable Count:{}'.format(len(discrete_features)))
train[discrete_features].head()
train[discrete_features].columns

train[discrete_features].head()
for feature in discrete_features:

    data = train.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
continuous_features = [feature for feature in numerical_features

                      if feature not in discrete_features + year_feature

                      +['Id']]

print('continuous features {}:'.format(len(continuous_features)))
train[continuous_features].head()
for feature in continuous_features:

    data = train.copy()

    data[feature].hist(bins=25)

    plt.xlabel(feature)

    plt.ylabel('count')

    plt.title(feature)

    plt.show()

    
categorical_features = train.select_dtypes(include=['object'])

categorical_features.columns
train.isna().sum()
Total = train.isnull().sum().sort_values(ascending = False)

percent = train.isnull().sum()/len(train)

missing_data = pd.concat([Total,percent],axis=1,keys=['Total','Percent'])

missing_data
col_with_missing = [col for col in train.columns

                   if train[col].isna().any()]

col_with_missing
import missingno as msno
msno.matrix(train)
msno.matrix(train.sample(100))
msno.bar(train)
msno.heatmap(train)
msno.dendrogram(train)
train.skew(),train.kurt()
sns.distplot(train.skew())
sns.distplot(train.kurt())
train['SalePrice'].hist()
target = np.log(train['SalePrice'])

target.skew()

plt.hist(target,color='orange')
correlation = numerical_features.corr()

print(correlation['SalePrice'].sort_values(ascending=False))
f, ax = plt.subplots(figsize=(15,15))

plt.title('Correlation of Numerical Features with SalePrice')

sns.heatmap(correlation,square=True,vmax=0.8)
var = 'OverallQual'

data = pd.concat([train['SalePrice'],train[var]],axis=1)

f,ax = plt.subplots(figsize=(12,8))

fig = sns.boxplot(x=var,y='SalePrice',data=data)



var = 'Neighborhood'

data = pd.concat([train.SalePrice,train[var]],axis=1)

f,ax = plt.subplots(figsize=(15,15))

sns.boxplot(x=var,y='SalePrice',data=data)

plt.xticks(rotation=45)
plt.figure(figsize=(15,10))

sns.countplot(x='Neighborhood',data=data)

plt.xticks(rotation=45)
var = 'SaleType'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 10))

fig = sns.boxplot(x='SaleType', y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

xt = plt.xticks(rotation=45)
var = 'SaleCondition'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 10))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

xt = plt.xticks(rotation=45)
features_with_na =[feature for feature in train.columns

                  if train[feature].isnull().sum()>1]

for feature in features_with_na:

    print(feature,np.round(train[feature].isnull().mean(), 4),' % of Missing Values')
total = numerical_features.isnull().sum().sort_values(ascending=False)

percent = total/len(numerical_features)

missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])

missing_data
missing_data.plot.bar()
total = categorical_features.isnull().sum().sort_values(ascending=False)

percent =total/len(categorical_features)

missing_data = pd.concat([total,percent],axis=1,keys=['Total','percent'])

missing_data
missing_values = categorical_features.isnull().sum(axis=0).reset_index()

missing_values.columns =['column_name','missing_count']

missing_values = missing_values.loc[missing_values['missing_count']>0]

missing_values = missing_values.sort_values(by='missing_count')



ind = np.arange(missing_values.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_values.missing_count.values, color='orange')

ax.set_yticks(ind)

ax.set_yticklabels(missing_values.column_name.values, rotation='horizontal')

ax.set_xlabel("Missing Observations Count")

ax.set_title("Missing Observations Count - Categorical Features")

plt.show()



for col in train.columns:

    if train[col].dtypes =='object':

        train[col] = train[col].fillna(train[col].mode().iloc[0])

        unique_category = len(train[col].unique())

        print("Feature '{column_name}' has '{unique_category}' unique categories".format(column_name = col,

                                                                                         unique_category=unique_category))
for col in test.columns:

    if test[col].dtypes =='object':

        test[col] = test[col].fillna(test[col].mode().iloc[0])

        unique_category = len(test[col].unique())

        print("Features in test set '{column_name}' has '{unique_category}' unique categories".format(column_name = col, unique_category=unique_category))
for feature in continuous_features:

    data = train.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature] = np.log(data[feature])

        data.boxplot(column=feature)

        plt.ylabel(feature)

        plt.title(feature)

        plt.show()
categorical_features.head()
for col in categorical_features:

    data = train.copy()

    data.groupby(col)['SalePrice'].median().plot.bar()

    plt.xlabel(col)

    plt.ylabel('SalePrice')

    plt.title(col)

    plt.show()