# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import missingno as msno # missing data visualizations 

import scipy.stats as stats

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#drop the survived column from train, then combine train and test 

combine = pd.concat([train.drop('Survived', axis=1), test])

train
# data type

train.info()
# data quality report

# Generates descriptive statistics that summarize the central tendency, dispersion and shape of 

# a dataset’s distribution, excluding NaN values.

train.describe() # include=None, only numeric columns
# describe the non-numeric (usually categorical) columns

# For object data (e.g. strings or timestamps), the result’s index will include count, unique, top, and freq. The top is the most common value. 

# The freq is the most common value’s frequency. 

train.describe(include=[np.object])

# Series.valud_counts Returns object containing counts of unique values.

train['Survived'].value_counts()
# method 1: sum up the missing values for each column 

# DataFrame.isull(): Return a boolean same-sized object indicating if the values are NA.

# then .sum() summarizes each column

train.isnull().sum()
# method 2: visualize the missing values

# width_ratios: The ratio of the width of the matrix to the width of the sparkline.

msno.matrix(train, fontsize=25, width_ratios=(10,1)) 

msno.bar(train, fontsize=25)
# which features have missing values (nan)?

train.columns[train.isnull().sum() > 0]
msno.bar(test, fontsize=25)
# Only 2 of Embaked are missing. We can remove these two records or replace the value with this feature's mode.

combine.fillna({'Fare': combine['Fare'].median(), 'Embarked': combine['Embarked'].mode()[0]}, inplace=True)

combine.isnull().sum()
# most of 'Cabin' are missing, just remove it

train.drop('Cabin', axis=1, inplace=True)

test.drop('Cabin', axis=1, inplace=True)

combine.drop('Cabin', axis=1, inplace=True)
# check missing values now: only Age has missing values

combine.isnull().sum()
# overall distribution of age in the training set excluding NaN

# this is a combination of histogram and kernel density estimation in seaborn

# a shortcut is to use the 'hist' of pandas

sns.distplot(train['Age'].dropna(), rug=True)
sns.FacetGrid(train, col='Survived', size=6).map(sns.distplot, 'Age')

# according to 'Survived', we create multiple grids for each possible value
sns.boxplot('Survived', 'Age', data=train)
sns.distplot(train['Age'].dropna(), rug=True)

plt.figure(figsize=(18, 5))

# a bar plot shows only the mean (or other estimator) value (specified by the estimator parameter)

sns.barplot('Age', 'Survived', data=train) 
train.head()
# for the three categorial features

# we usually use boxplot to reflect the relation between categorical and continuous variables

sns.boxplot('Pclass', 'Age', data=train)

plt.figure()

sns.boxplot('Sex', 'Age', data=train)

plt.figure()

sns.boxplot('Embarked', 'Age', data=train)
sns.boxplot('Pclass', 'Age', hue='Sex', data=train)
g = sns.FacetGrid(data=train, row = 'Sex', col='Pclass', size=5)

g.map(sns.distplot, 'Age');
# impute Age: here we many combine the train and test set together for feature imputation 

# https://www.kaggle.com/questions-and-answers/37491

n_train = len(train)

n_test = len(test)

combine = pd.concat([train, test], axis=0, ignore_index=True)  # If True, do not use the index values along the concatenation axis.

for pc in range(1, 4):

    for sex in ['female', 'male']:

        filter = (combine['Pclass'] == pc) & (combine['Sex'] == sex)

        m = combine.loc[filter, 'Age'].mean()

        s = combine.loc[filter, 'Age'].std()

        nan_filter = filter & (combine['Age'].isnull()) # identify the NaN Age in this group

        combine.loc[nan_filter, 'Age'] = np.random.normal(m, s, nan_filter.sum())

        # we can also use np.count_nonzero() to find the number of True's in nan_filter
combine.isnull().sum()
# Embark has only two missings, just fill it with the mode

combine['Embarked'].fillna(combine['Embarked'].mode()[0], inplace=True)
# one Fare is missing

combine[combine['Fare'].isnull()]
# intuitively, Fare depends on Pclass. We fill it with the average of Pclass=3

combine.loc[combine['Fare'].isnull(), 'Fare'] = combine.loc[combine['Pclass'] == 3, 'Fare'].mean()
# check nan now

# Dataframe.values Numpy representation of NDFrame

# ndarray.any(): Returns True if any of the elements of a evaluate to True.

combine.isnull().values.any()

msno.bar(combine, fontsize=25)
combine.head(10)
import re

# anything, some_characters.

# for the remaining characters, we don't care. match.group(0) will be the full match of the regexp

#"Braund, Mr. Owen Harris" --> group(0) ---> 'Braund, Mr.'

# group(1) --> 'Mr'

# However, this regexp fails for passenger760, Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)

# that is, there may be spaces in the title

# add a new column

combine['Title'] = combine['Name'].apply(lambda name: re.match(r'.+, ([a-zA-z ]+)\.', name).group(1))

combine['Title'].value_counts()
combine['Title'].replace(['Mlle', 'Lady'], 'Miss', inplace=True)

combine['Title'].replace([ 'Mme', 'Ms', 'Dona', 'the Countess'], 'Mrs', inplace=True)

# for the others, all replaced by 'Mr'

combine['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Jonkheer', 'Sir', 'Don', 'Capt'], 'Mr', inplace=True)

combine['Title'].value_counts()

# after we finish the above data preprocess for both training and test set, we isolate the training set 

train = combine[0:n_train]

# What's the relation between *Title* and *Survival*?

# let's check the survival rate for each Title

mean_survival_each_title = train['Survived'].groupby(train['Title']).mean()

print(mean_survival_each_title)

# the above computation can also be done with sns.barplot

# a bar plot shows only the mean (or other estimator) value (specified by the estimator parameter)

sns.barplot('Title', 'Survived', data=train)
sns.barplot('Embarked', 'Survived', data=train)

# it seems that *Embarked* has an influence, though not significantly.
# pandas.get_dummies: one-hot encoding

# the same encoding scheme should be applied to both training and test set, or combine, for convenience

combine = pd.get_dummies(combine, columns=['Sex', 'Embarked', 'Title'])

combine.head(5)
#NOTE: we need the passengerId for the test set to submit our predication result

test_passengerId = combine.loc[n_train:, 'PassengerId']

combine.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

combine.head(5)
# now check the whole dataset

combine.info()
train = combine[0:n_train]

corr = train.corr('spearman')

plt.figure(figsize=(13, 13))

sns.heatmap(corr, square=True, annot=True)
# exclude the column for y and convert it into numpy array  for sklearn algorithms

train = combine[0:n_train]

test = combine[n_train:]

train_X = train.drop('Survived', axis=1).values 

train_y = train['Survived'].values

test_X = test.drop('Survived', axis=1).values 
print(train_X.shape, train_y.shape, test_X.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(500, min_samples_split=10, oob_score=True)

rf.fit(train_X, train_y)
# the out-of-bag estimate of the random forest

rf.oob_score_
predict_train_y = rf.predict(train_X)

print(accuracy_score(train_y, predict_train_y))
predict_test_y = rf.predict(test_X).astype(int) # default is float, which will give you 0 score after submission

result = pd.DataFrame({'PassengerId': test_passengerId, 'Survived': predict_test_y})

result.to_csv('submission.csv', index=False)
from sklearn.model_selection import RandomizedSearchCV

import scipy.stats as stats

params = {'n_estimators': stats.randint(low=300, high=700),  # a random integer between 300 and 700

          # If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split

          # uniform(0.1, 0.4) returns a random continous variable between 0.1 and 0.1+0.4=0.5

          'max_features': stats.uniform(loc=0.1, scale=0.4),

          # If a list is given, it is sampled uniformly.

          'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

rscv = RandomizedSearchCV(rf, param_distributions=params, n_iter=30, scoring='accuracy', n_jobs=-1, cv=5)

rscv.fit(train_X, train_y)
# cv_results_ : dict of numpy (masked) ndarrays. We import it into DataFrame for display.

pd.DataFrame(rscv.cv_results_).sort_values(by='mean_test_score', ascending=False)
# access the best result with best_params_ or best_estimator_ (the estimator using the best_params_)

rscv.best_params_
def generate_submission(file, model):

    predict_test_y = model.predict(test_X).astype(int) # default is float, which will give you 0 score after submission

    result = pd.DataFrame({'PassengerId': test_passengerId, 'Survived': predict_test_y})

    result.to_csv(file + '.csv', index=False)
generate_submission('rf_rscv_5.1.1b', rscv.best_estimator_)
params = {'n_estimators': stats.randint(low=600, high=1000),  # a random integer between 600 and 1000

          # If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split

          # uniform(0.2, 0.4) returns a random continous variable between 0.2 and 0.2+0.4=0.6

          'max_features': stats.uniform(loc=0.2, scale=0.4),

          # If a list is given, it is sampled uniformly.

          'min_samples_split': stats.randint(low=8, high=20)}

rscv = RandomizedSearchCV(rf, param_distributions=params, n_iter=30, scoring='accuracy', n_jobs=-1, cv=5)

rscv.fit(train_X, train_y)

pd.DataFrame(rscv.cv_results_).sort_values(by='mean_test_score', ascending=False)
rscv.best_params_
generate_submission('rf_rscv_5.1.1b', rscv.best_estimator_)