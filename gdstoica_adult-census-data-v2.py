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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from category_encoders.one_hot import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report    
df = pd.read_csv(os.path.join(dirname, filename))
df.info()
df.shape
df.head()
df.isnull().sum()
df.columns[df.isin(['?']).any()]  #search for values equal with '?' anywhere in the df
df['workclass'].value_counts()
df['occupation'].value_counts()
df['native.country'].value_counts()
#Replace '?' with np.nan and drop the rows containing nulls
df['occupation'].replace(to_replace='?', value=np.nan, inplace=True)
df['workclass'].replace(to_replace='?', value=np.nan, inplace=True)
df['native.country'].replace(to_replace='?', value=np.nan, inplace=True)
df.dropna(axis=0, subset=['occupation', 'workclass', 'native.country'], inplace=True)
sns.countplot(x='income', data=df)
df['income_enc'] = np.where(df['income']=='<=50K', 0, 1) #dummy encode (0/1 encoding for target variable)
df.drop(columns=['income'], inplace=True)
df['occupation'].value_counts()
occupation = { 'Exec-managerial': 'white_collar',
               'Adm-clerical': 'white_collar',
               'Tech-support': 'white_collar',
               'Craft-repair': 'blue_collar',
               'Machine-op-inspct': 'blue_collar',
               'Transport-moving': 'blue_collar',
               'Handlers-cleaners': 'blue_collar',
               'Farming-fishing': 'blue_collar',
               'Sales': 'pink_collar', 
               'Other-service': 'pink_collar',
               'Protective-serv': 'pink_collar', 
               'Priv-house-serv': 'pink_collar',
               'Prof-specialty': 'gold_collar',
               'Armed-Forces': 'gold_collar'}
df['occupation'] = df['occupation'].map(occupation)  
df['occupation'].value_counts()
df['workclass'].value_counts()
workclass = {'Private':'private', 'Self-emp-not-inc': 'no_income', 
             'Local-gov':'gov', 'State-gov': 'gov', 'Self-emp-inc': 'self_emp', 
             'Federal-gov': 'gov', 'Without-pay': 'no_income', 
             'Never-worked':'no_income' }
df['workclass'] = df['workclass'].map(workclass)  
df['workclass'].value_counts()
df['marital.status'].value_counts()
marital_status = {'Married-civ-spouse': 'married', 'Never-married':'single',  
                  'Divorced':'single', 'Separated':'single', 'Widowed':'single', 
                  'Married-spouse-absent':'married', 
                  'Married-AF-spouse':'married'}
df['marital.status'] = df['marital.status'].map(marital_status)  
df['relationship'].value_counts()
df['has_children'] = np.where(df['relationship'] == 'Own-child', 1, 0)
df.drop(columns=['relationship'], inplace=True)
df.head()
df['native.country'].value_counts()
usa_native = df['native.country']=='United-States'
len(df[usa_native]['native.country'])/len(df)  #91%
df['usa_native'] = np.where(df['native.country'] == 'United-States', 1, 0)
df.drop(columns=['native.country'], inplace=True)
df[['education', 'education.num']] 
#check if there is a relation between the 2 variables --> same information is provided by both 
#features --> we will drop the categorical one

pd.crosstab(df['education.num'],df['education'])
df.drop(columns=['education'], inplace=True)
df['race'].value_counts() #collapse the last two categories into one
df['race'] = df['race'].replace(to_replace='Amer-Indian-Eskimo', value='Other')
features = df.drop(columns=['income_enc'])
target = df['income_enc']
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.3)
trainset = pd.concat([xtrain, ytrain], axis=1)
trainset.describe()
sns.distplot(trainset['age'])
sns.distplot(trainset['fnlwgt']) #positive skew
trainset['fnlwgt'].describe()
sns.distplot(np.log(trainset['fnlwgt'])) #Transform fnlwgt to have a more centered distribution
df.loc[:, 'fnlwgt_log'] = df['fnlwgt'].apply(lambda x: np.log(x))
df.drop(columns=['fnlwgt'], inplace=True)
trainset['capital.gain'].hist()
trainset['capital.gain'].describe()
trainset['capital.gain'].nunique()
non_zero_mask = trainset['capital.gain']!=0
trainset[non_zero_mask]['capital.gain'].count()
trainset.groupby(['capital.gain']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
trainset['capital.loss'].describe()
trainset.groupby(['capital.loss']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
#the 2 features exclude one another
trainset[['capital.gain', 'capital.loss']]
m1 = trainset['capital.loss']>0
m2 = trainset['capital.gain']>0
trainset[m1&m2]
df['capital'] = 0 - df['capital.loss'] #changed on both trainset and testset
df['capital'] = df['capital']  + df['capital.gain']
df.drop(columns=['capital.gain', 'capital.loss'], inplace=True)
trainset.head()
df.head()
features = df.drop(columns=['income_enc']).copy(deep=True)
target = df['income_enc'].copy(deep=True)
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.3)
trainset = pd.concat([xtrain, ytrain], axis=1)
trainset.head()
sns.boxplot(trainset['age'])
def get_lower_and_upper_limits(df, column):
    """
    Returns lower and upper limits for the values taken by a column using IQR rule
    :param df:
    :param column:
    :return: lower and upper limits in this order
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    minimum = q1 - 1.5 * (q3 - q1)
    maximum = q3 + 1.5 * (q3 - q1)
    return minimum, maximum
get_lower_and_upper_limits(trainset, 'age') #age limits --> apply the limits from trainset on the entire df (both train & test)
trainset['age'].describe()
df.shape
age_max = 75.5 #only upper limit constraint since all the values are higher than the lower limit
constraint = (df['age'] > age_max)
df[~constraint].shape #not so many examples lost
df = df.drop(df[constraint].index) #drop rows
df.shape
sns.boxplot(trainset['hours.per.week'])
get_lower_and_upper_limits(trainset, 'hours.per.week')
hours_min = df['hours.per.week']<32.5
hours_max = df['hours.per.week']>52.5
df[hours_min].shape
df[hours_max].shape
def get_lower_and_upper_limits(df, column):
    """
    Returns lower and upper limits for the values taken by a column using IQR rule
    :param df:
    :param column:
    :return: lower and upper limits in this order
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    minimum = q1 - 5 * (q3 - q1)
    maximum = q3 + 5 * (q3 - q1)
    return minimum, maximum
get_lower_and_upper_limits(trainset, 'hours.per.week') #get limits on trainset
hours_min = df['hours.per.week']<15
hours_max = df['hours.per.week']>70
df[hours_min].shape
df[hours_max].shape
df = df.drop(df[hours_min].index) #apply limits on all dataset
df = df.drop(df[hours_max].index)
df.shape
xtrain['capital'].describe()
sns.boxplot(xtrain['capital'])
get_lower_and_upper_limits(xtrain, 'capital') #test with transforming this into positive/negative
# c1 = df['capital']<0
# c2 = df['capital']==0
# c3 = df['capital']>0

# df.loc[c1, 'capital_type'] = -1
# df.loc[c2, 'capital_type'] = 0
# df.loc[c3, 'capital_type'] = 1
# df.drop(columns=['capital'], inplace=True)
features = df.drop(columns=['income_enc']).copy(deep=True)
target = df['income_enc'].copy(deep=True)
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.3)
trainset = pd.concat([xtrain, ytrain], axis=1)
xtrain.head()
numerical_features = list(trainset.select_dtypes(include=np.number))
numerical_features.remove('income_enc')
f, ax = plt.subplots(figsize=(10, 8))
corr = trainset[numerical_features].corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, annot=True, ax=ax)
g = sns.FacetGrid(trainset, col='has_children')
g.map(plt.hist, 'age', bins=20)
g = sns.FacetGrid(trainset, col='income_enc')
g.map(plt.hist, 'age', bins=20)
plt.figure(figsize=(12,5))
plt.title("Box plot for income")
sns.boxplot(y="income_enc", x="age", data =  trainset, orient="h", palette = 'magma')
plt.figure(figsize=(12,5))
plt.title("Box plot for income")
sns.boxplot(y="income_enc", x="hours.per.week", data =  trainset, orient="h", palette = 'magma')
g = sns.FacetGrid(trainset, col='income_enc')
g.map(plt.hist, 'hours.per.week', bins=20)
trainset.info()
ohe = OneHotEncoder(verbose=0, cols=['workclass', 'marital.status', 'occupation', 'race', 'sex']  , drop_invariant=False, return_df=True, handle_missing='value', handle_unknown='value', use_cat_names=True)
ohe.fit(xtrain)
xtrain = ohe.transform(xtrain)
xtest = ohe.transform(xtest)
xtrain.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
rf = RandomForestClassifier(random_state=42, class_weight="balanced")
rf.fit(xtrain,ytrain)
ypred_test = rf.predict(xtest)
ypred_train = rf.predict(xtrain)
print(classification_report(ytrain, ypred_train))
print(classification_report(ytest, ypred_test))
rf = RandomForestClassifier(random_state=42, class_weight={0:1, 1:2.5}, max_depth=18, n_estimators=200, max_samples=0.8)
rf.fit(xtrain,ytrain)
ypred_test = rf.predict(xtest)
ypred_train = rf.predict(xtrain)
print(classification_report(ytrain, ypred_train))
print(classification_report(ytest, ypred_test))
cfm = confusion_matrix(ytest, ypred_test)
fig, ax = plt.subplots(figsize=(10,10))
sns.set(font_scale=1.4)
sns.heatmap(cfm/np.sum(cfm), annot=True, fmt='.2%', ax=ax)
pd.DataFrame(list(zip(xtrain.columns, rf.feature_importances_))).sort_values(by=1, ascending=False)
