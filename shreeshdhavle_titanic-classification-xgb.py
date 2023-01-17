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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
# save PassengerId for final submission
passengerId = test.PassengerId

# merge train and test
data = train.append(test, ignore_index=True)

# create indexes to separate data later on
train_idx = len(train)
test_idx = len(data) - len(test)
data.head()
print(data.shape)
print(data.dtypes)
print(data['Survived'].value_counts())
data.describe()
data.isnull().sum()
#Importing Libraries
import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats

%matplotlib inline
df_con = data.copy()
# Droping all categorical features
cat_feat = ['PassengerId', 'Name', 'Ticket', 'Sex', 'Cabin', 'Embarked']
df_con.drop(cat_feat, axis=1, inplace=True)
print(df_con.head())
print(df_con.describe())
# Looking at the correlation matrix
df_con.corr()
# Looking at fare by different passenger class levels
df_con.groupby('Pclass')['Fare'].describe()
def describe_cont_feature(feature):
    print('\n*** Results for {} ***'.format(feature))
    print(df_con.groupby('Survived')[feature].describe())
    print(ttest(feature))
    
def ttest(feature):
    survived = df_con[df_con['Survived']==1][feature]
    not_survived = df_con[df_con['Survived']==0][feature]
    tstat, pval = stats.ttest_ind(survived, not_survived, equal_var=False)
    print('t-statistic: {:.1f}, p-value: {:.3}'.format(tstat, pval))
    
# Looking at the distribution of each feature at each level of the target variable
for feature in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
    describe_cont_feature(feature)
# Looking at the average value of each feature based on whether Age is missing
df_con.groupby(df_con['Age'].isnull()).mean()
# Plot overlaid histograms for continuous features
for i in ['Age', 'Fare']:
    died = list(df_con[df_con['Survived'] == 0][i].dropna())
    survived = list(df_con[df_con['Survived'] == 1][i].dropna())
    xmin = min(min(died), min(survived))
    xmax = max(max(died), max(survived))
    width = (xmax - xmin) / 40
    sns.distplot(died, color='r', kde=False, bins=np.arange(xmin, xmax, width))
    sns.distplot(survived, color='g', kde=False, bins=np.arange(xmin, xmax, width))
    plt.legend(['Did not survive', 'Survived'])
    plt.title('Overlaid histogram for {}'.format(i))
    plt.show()
# Generate categorical plots for ordinal features
for col in ['Pclass', 'SibSp', 'Parch']:
    sns.catplot(x=col, y='Survived', data=df_con, kind='point', aspect=2, )
    plt.ylim(0, 1)
# Creating new family size feature
df_con['Family_size'] = df_con['SibSp'] + df_con['Parch']
sns.catplot(x='Family_size', y='Survived', data=df_con, kind='point', aspect=2, )
plt.ylim(0, 1)
df_cat= data.copy()
df_cat.head()
# Droping all continuous features
cont_feat = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
df_cat.drop(cont_feat, axis=1, inplace=True)
df_cat.head()
# Checking if there are any missing values
df_cat.isnull().sum()
# Exploring the number of unique values for each feature
for col in df_cat.columns:
    print('{}: {} unique values'.format(col, df_cat[col].nunique()))
# Checking survival rate by gender
df_cat.groupby('Sex').mean()
# Checking survival rate by the port departed from
df_cat.groupby('Embarked').mean()
# Checking if Cabin are missing at random
df_cat.groupby(df_cat['Cabin'].isnull()).mean()
# Looking at unique values for the Ticket feature
df_cat['Ticket'].value_counts()
# Inspecting names Columns
df_cat['Name'].unique()[:10]
# Create a title feature by parsing passenger name
df_cat['Title'] = df_cat['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df_cat.head()
# Looking at survival rate by title
df_cat.pivot_table('Survived', index=['Title', 'Sex'], aggfunc=['count', 'mean'])
# Map of aggregated titles:
titles_dict = {'Capt': 'Officer',
               'Col': 'Officer',
               'Major': 'Officer',
               'Jonkheer': 'Royalty',
               'Don': 'Royalty',
               'Sir': 'Royalty',
               'Dr': 'Officer',
               'Rev': 'Officer',
               'the Countess': 'Royalty',
               'Dona': 'Royalty',
               'Mme': 'Mrs',
               'Mlle': 'Miss',
               'Ms': 'Miss',
               'Mr': 'Mr',
               'Mrs': 'Mrs',
               'Miss': 'Miss',
               'Master': 'Master',
               'Lady': 'Royalty'}

df_cat['Title'] = df_cat['Title'].map(titles_dict)
df_cat['Title'].head()
df_cat['Title'].unique()
df_cat['Cabin_ind'] = np.where(df_cat['Cabin'].isnull(), 0, 1)
df_cat.head()
df_cat['Embarked'].describe()
common_value = 'S'
df_cat['Embarked'] = df_cat['Embarked'].fillna(common_value)
df_cat.isnull().sum()
# Generate categorical plots for features
for col in ['Title', 'Sex', 'Cabin_ind', 'Embarked']:
    sns.catplot(x=col, y='Survived', data=df_cat, kind='point', aspect=2, )
    plt.ylim(0, 1)
# Split embarked by whether the passenger had a cabin
df_cat.pivot_table('Survived', index='Cabin_ind', columns='Embarked', aggfunc='count')
data.isnull().sum()
data['Embarked'].describe()
# Fill in missing values for the Embarked feature
common_value = 'S'
data['Embarked_Cleaned'] = data['Embarked'].fillna(common_value)
data.isnull().sum()
age_Pclass_sex =data.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_Pclass_sex[sex][pclass]))
        
print('Median age of all passengers: {}'.format(data['Age'].median()))
data['Age_Cleaned']= data.groupby(['Sex','Pclass'])['Age'].apply(lambda x:x.fillna(x.median()))
data.isnull().sum()
# Plot histogram for continuous feature to see if a transformation is needed
for feature in ['Age_Cleaned', 'Fare']:
    sns.distplot(data[feature], kde=False)
    plt.title('Histogram for {}'.format(feature))
    plt.show()
# Generating QQ plots
for i in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    data_t = data['Fare']**(1/i)
    qqplot(data_t, line='s')
    plt.title("Transformation: 1/{}".format(str(i)))
# Box-Cox transformation
for i in [3, 4, 5, 6, 7]:
    data_t = data['Fare']**(1/i)
    n, bins, patches = plt.hist(data_t, 50, density=True)
    mu = np.mean(data_t)
    sigma = np.std(data_t)
    plt.plot(bins, scipy.stats.norm.pdf(bins, mu, sigma))
    plt.title("Transformation: 1/{}".format(str(i)))
    plt.show()
# Create the new transformed feature
data['Fare_clean'] = data['Fare'].apply(lambda x: x**(1/5))
data.head()
data.isnull().sum()
# fill NaN with median fare
data.Fare_clean = data.Fare_clean.fillna(data.Fare_clean.median())
data.isnull().sum()
data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
data.head()
# Map of aggregated titles:
titles_dict = {'Capt': 'Officer',
               'Col': 'Officer',
               'Major': 'Officer',
               'Jonkheer': 'Royalty',
               'Don': 'Royalty',
               'Sir': 'Royalty',
               'Dr': 'Officer',
               'Rev': 'Officer',
               'the Countess': 'Royalty',
               'Dona': 'Royalty',
               'Mme': 'Mrs',
               'Mlle': 'Miss',
               'Ms': 'Miss',
               'Mr': 'Mr',
               'Mrs': 'Mrs',
               'Miss': 'Miss',
               'Master': 'Master',
               'Lady': 'Royalty'}

data['Title'] = data['Title'].map(titles_dict)
data['Title'].unique()
data['Cabin_Cleaned'] = np.where(data['Cabin'].isnull(), 0, 1)
# Creating new family size feature
data['Family_size'] = data['SibSp'] + data['Parch']+ 1
data.head()
data.isnull().sum()
data.drop( ['Ticket', 'PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)
data.head()
data.dtypes
# Convert the male and female groups to integer form
data.Sex = data.Sex.map({"male": 0, "female":1})

# create dummy variables for categorical features
title_dummies = pd.get_dummies(data.Title, prefix="Title")
embarked_dummies = pd.get_dummies(data.Embarked_Cleaned, prefix="Embarked")

# concatenate dummy columns with main dataset
data_Cleaned = pd.concat([data, title_dummies, embarked_dummies], axis=1)
# drop categorical fields
data_Cleaned.drop(['Title', 'Embarked_Cleaned'], axis=1, inplace=True)
data_Cleaned.head()
data_Cleaned.isnull().sum()
data_Cleaned.dtypes

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# create train and test data
train = data_Cleaned[ :train_idx]
test = data_Cleaned[test_idx: ]

# convert Survived back to int
train.Survived = train.Survived.astype(int)

# create X and y for data and target values 
X = train.drop('Survived', axis=1).values 
y = train.Survived.values

# create array for test set
X_test = test.drop('Survived', axis=1).values
train.shape
test.shape
rf = RandomForestClassifier()
# create param grid object 
forest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 60, 10)],
)
# build and fit model 
forest_cv = GridSearchCV(estimator=rf, param_grid=forest_params, cv=5) 
forest_cv.fit(X, y)
print("Best score: {}".format(forest_cv.best_score_))
print("Optimal params: {}".format(forest_cv.best_estimator_))
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
xgb =XGBClassifier()
# create param grid object 
xgb_params = dict(     
    max_depth = [n for n in range(2, 9)],     
    learning_rate = [0.01, 0.05, 0.1, 0.5],    
    n_estimators = [n for n in range(10, 300, 10)],
)
xgb_cv = GridSearchCV(estimator=xgb, param_grid=xgb_params, cv=5) 
xgb_cv.fit(X, y)
print("Best score: {}".format(xgb_cv.best_score_))
print("Optimal params: {}".format(xgb_cv.best_estimator_))
# prediction on test set
xgb_pred = xgb_cv.predict(X_test)
# dataframe with predictions
kaggle = pd.DataFrame({'PassengerId': passengerId, 'Survived': xgb_pred})
# save to csv
kaggle.to_csv('./titanic_pred.csv', index=False)