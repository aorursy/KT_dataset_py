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



import missingno as msno



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#最大表示列数の指定（ここでは50列を指定）

pd.set_option('display.max_columns', 122)
app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
app_train.head()
app_train.info()
app_train.describe()
app_train.shape
app_train.isnull().any()
msno.matrix(app_train)
pd.options.display.float_format = '{: <10.2%}'.format

total = app_train.isnull().sum().sort_values(ascending=False)

percent = (app_train.isnull().sum()/app_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(sum(total > 0))
pd.reset_option('display.float_format')
app_train['TARGET'].value_counts()
sns.countplot(x='TARGET', data=app_train)
features = app_train.loc[:, app_train.columns != 'TARGET']

target = app_train.loc[:, app_train.columns == 'TARGET']



features_corr_1 = app_train.iloc[:, :40]

features_corr_2 = app_train.iloc[:, 40:80]

features_corr_3 = app_train.iloc[:, 80:121]



features_corr_1 = pd.concat([target, features_corr_1], axis=1)

features_corr_2 = pd.concat([target, features_corr_2], axis=1)

features_corr_3 = pd.concat([target, features_corr_3], axis=1)
plt.figure(figsize=(30, 24)) 

sns.heatmap(features_corr_1.corr(), annot=True, square=True, fmt='.2f', cmap='gist_rainbow', vmin=-1, vmax=1)

plt.show()
plt.figure(figsize=(30, 24)) 

sns.heatmap(features_corr_2.corr(), annot=True, square=True, fmt='.2f', cmap='gist_rainbow', vmin=-1, vmax=1)

plt.show()
plt.figure(figsize=(30, 24)) 

sns.heatmap(features_corr_3.corr(), annot=True, square=True, fmt='.2f', cmap='gist_rainbow', vmin=-1, vmax=1)

plt.show()
app_train['DAYS_BIRTH_YEAR'] = (app_train['DAYS_BIRTH'] // 365).abs()



age_ctgr_list = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

age_ctgr_name = ['20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-70']

 

app_train['DAYS_BIRTH_CTGR'] = pd.cut(app_train['DAYS_BIRTH_YEAR'], bins=age_ctgr_list, labels=age_ctgr_name)
sns.countplot(x='DAYS_BIRTH_CTGR', hue='TARGET', data=app_train)
sns.kdeplot(app_train['DAYS_BIRTH_YEAR'][app_train['TARGET'] == 0], shade=True, color="b", label='TARGET = 0')

sns.kdeplot(app_train['DAYS_BIRTH_YEAR'][app_train['TARGET'] == 1], shade=True, color="r", label='TARGET = 1')
app_train['AMT_INCOME_TOTAL_IN_DAYS_BIRTH'] = app_train['AMT_INCOME_TOTAL'] / app_train['DAYS_BIRTH'].abs()

app_train['AMT_INCOME_TOTAL_IN_AMT_CREDIT'] = app_train['AMT_INCOME_TOTAL'] / app_train['AMT_CREDIT']
app_train['AMT_INCOME_TOTAL_IN_DAYS_BIRTH'].describe()
income_analysis = pd.concat([target, app_train['AMT_INCOME_TOTAL_IN_DAYS_BIRTH'],app_train['AMT_INCOME_TOTAL_IN_AMT_CREDIT']], axis=1)
#外れ値の削除(改善余地あり)

for inx in income_analysis['AMT_INCOME_TOTAL_IN_DAYS_BIRTH'][income_analysis['AMT_INCOME_TOTAL_IN_DAYS_BIRTH'] > 26].index:

    income_analysis.drop(index=inx, inplace=True)
grid = sns.FacetGrid(income_analysis, col='TARGET', hue='TARGET', col_wrap=2)

grid.map(sns.distplot, 'AMT_INCOME_TOTAL_IN_DAYS_BIRTH', bins=10, kde=False)

plt.show()
sns.kdeplot(income_analysis['AMT_INCOME_TOTAL_IN_DAYS_BIRTH'][income_analysis['TARGET'] == 0], shade=True, color="b", label='TARGET = 0')

sns.kdeplot(income_analysis['AMT_INCOME_TOTAL_IN_DAYS_BIRTH'][income_analysis['TARGET'] == 1], shade=True, color="r", label='TARGET = 1')
#外れ値の削除(改善余地あり)

for inx in income_analysis['AMT_INCOME_TOTAL_IN_AMT_CREDIT'][income_analysis['AMT_INCOME_TOTAL_IN_AMT_CREDIT'] > 1].index:

    income_analysis.drop(index=inx, inplace=True)
grid = sns.FacetGrid(income_analysis, col='TARGET', hue='TARGET', col_wrap=2)

grid.map(sns.distplot, 'AMT_INCOME_TOTAL_IN_AMT_CREDIT', bins=10, kde=False)

plt.show()
sns.kdeplot(income_analysis['AMT_INCOME_TOTAL_IN_AMT_CREDIT'][income_analysis['TARGET'] == 0], shade=True, color="b", label='TARGET = 0')

sns.kdeplot(income_analysis['AMT_INCOME_TOTAL_IN_AMT_CREDIT'][income_analysis['TARGET'] == 1], shade=True, color="r", label='TARGET = 1')
pd.DataFrame(app_train.groupby(['OCCUPATION_TYPE', 'TARGET'])['SK_ID_CURR'].count())
sns.catplot(y='OCCUPATION_TYPE', hue='TARGET', data=app_train, kind="count", height=8, aspect=1)
pd.DataFrame(app_train.groupby(['CODE_GENDER', 'TARGET'])['SK_ID_CURR'].count())
sns.countplot(x='CODE_GENDER', hue='TARGET', data=app_train)
pd.DataFrame(app_train.groupby(['NAME_FAMILY_STATUS', 'TARGET'])['SK_ID_CURR'].count())
sns.countplot(y='NAME_FAMILY_STATUS', hue='TARGET', data=app_train)
pd.DataFrame(app_train.groupby(['CNT_CHILDREN', 'TARGET'])['SK_ID_CURR'].count())
sns.countplot(y='CNT_CHILDREN', hue='TARGET', data=app_train)
pd.DataFrame(app_train.groupby(['NAME_TYPE_SUITE', 'TARGET'])['SK_ID_CURR'].count())
sns.countplot(y='NAME_TYPE_SUITE', hue='TARGET', data=app_train)
pd.DataFrame(app_train.groupby(['FLAG_OWN_CAR', 'TARGET'])['SK_ID_CURR'].count())
sns.countplot(x='FLAG_OWN_CAR', hue='TARGET', data=app_train)
pd.DataFrame(app_train.groupby(['FLAG_OWN_REALTY', 'TARGET'])['SK_ID_CURR'].count())
sns.countplot(x='FLAG_OWN_REALTY', hue='TARGET', data=app_train)
pd.DataFrame(app_train.groupby(['NAME_HOUSING_TYPE', 'TARGET'])['SK_ID_CURR'].count())
sns.countplot(y='NAME_HOUSING_TYPE', hue='TARGET', data=app_train)
pd.DataFrame(app_train.groupby(['NAME_EDUCATION_TYPE', 'TARGET'])['SK_ID_CURR'].count())
sns.countplot(y='NAME_EDUCATION_TYPE', hue='TARGET', data=app_train)
app_train['AMT_CREDIT_PAYOFF_YEAR'] = app_train['AMT_CREDIT'] // app_train['AMT_ANNUITY']
pd.DataFrame(app_train.groupby(['AMT_CREDIT_PAYOFF_YEAR', 'TARGET'])['SK_ID_CURR'].count())
grid = sns.FacetGrid(app_train, col='TARGET', hue='TARGET', col_wrap=2)

grid.map(sns.distplot, 'AMT_CREDIT_PAYOFF_YEAR', bins=20, kde=False)

plt.show()
sns.kdeplot(app_train['AMT_CREDIT_PAYOFF_YEAR'][app_train['TARGET'] == 0], shade=True, color="b", label='TARGET = 0')

sns.kdeplot(app_train['AMT_CREDIT_PAYOFF_YEAR'][app_train['TARGET'] == 1], shade=True, color="r", label='TARGET = 1')