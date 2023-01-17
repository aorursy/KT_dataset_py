# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Set random state for reproducibility

np.random.seed(42)



import os

# Any results you write to the current directory are saved as output.

# Look at data directory:

print(os.listdir('/kaggle/input/bank-marketing-prediction/'))
df_train = pd.read_csv('/kaggle/input/bank-marketing-prediction/train.csv')

df_test = pd.read_csv('/kaggle/input/bank-marketing-prediction/test.csv')
df_train.info()
df_test.info()
import seaborn as sns
sns.countplot('y', data=df_train);
sns.boxplot(y='age', x='y', data=df_train);
sns.boxplot(y='duration', x='y', data=df_train);
sns.jointplot(x='age', y='duration', data=df_train[(df_train['duration'] < 1000) & (df_train['age'] < 60)], kind='hex');
sns.barplot(y='y', x='month', data=df_train);
# List of features that I want to remove

remove_features = ['SampleId', 'y']#, 'day', 'month']
# Get the list of categorical features

categorical_features = list(df_train.dtypes[df_train.dtypes == 'object'].index)

categorical_features = [c for c in categorical_features if not c in remove_features]



categorical_features
# Get the list of numerical features

numerical_features = list(df_train.dtypes[df_train.dtypes != 'object'].index)

numerical_features = [c for c in numerical_features if not c in remove_features]



numerical_features
# Plot the distributions of all numerical features

sns.pairplot(data=df_train[:500], vars=numerical_features, hue='y', diag_kind='hist');
import matplotlib.pyplot as plt



# Plot the distributions of all numerical features and their logarithms

for col in numerical_features:

    # Create 2 subplots

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot distribution of the parameter

    sns.distplot(df_train[col], kde=False, ax=axes[0])

    # Take the logarithm of the parameter

    log_col = np.log(df_train[col] - df_train[col].min() + 1)

    log_col.name = 'log({})'.format(col)

    # Plot distribution of the log of the parameter

    sns.distplot(log_col, kde=False, ax=axes[1])

    plt.show()
# Plot all the same distributions using boxplots

for col in numerical_features:

    # Create 2 subplots

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot distribution of the parameter

    sns.boxplot(y=col, x='y', data=df_train, ax=axes[0])

    # Take the logarithm of the parameter

    log_col = np.log(df_train[col] - df_train[col].min() + 1)

    log_col.name = 'log({})'.format(col)

    # Plot distribution of the log of the parameter

    sns.boxplot(y=log_col, x='y', data=df_train, ax=axes[1])

    plt.show()
columns_log = ['balance', 'duration', 'campaign', 'pdays', 'previous']
# Drop columns we don't need

X_train = df_train.drop(remove_features, axis=1)

y_train = df_train['y']



X_test = df_test.drop(remove_features[:-1], axis=1) # not including 'y'



X_train.head()
# One-hot encoding of the categorical features



# Merge train and test parts

X = pd.concat([X_train, X_test], axis=0)

X = pd.get_dummies(X, categorical_features, drop_first=False)



# Split them again

X_train = X.iloc[:len(X_train)]

X_test = X.iloc[len(X_train):]



X_train.head()
# Logarithm of some values

for col in columns_log:

    for X in [X_train, X_test]:

        X[col + '_log'] = np.log(X[col] - X_train[col].min() + 1)

# Drop old values

for col in columns_log:

    for X in [X_train, X_test]:

        X.drop(col, axis=1, inplace=True)
# Plot the correlation matrix between features

plt.figure(figsize=(10, 10))

sns.heatmap(X_train.corr());
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_validate



# Scale features

scaler = StandardScaler()

scaler.fit(X_train)



X_train_sc = scaler.transform(X_train)

X_test_sc = scaler.transform(X_test)



# Create model

lr = LogisticRegression()



# Measure model performance

lr_f1 = cross_validate(lr, X_train_sc, y_train, cv=5, scoring='f1')['test_score'].mean()

print('f1 score: {:.03f}'.format(lr_f1))



# Fit linear model

lr.fit(X_train_sc, y_train)



predictions_lr = lr.predict(X_test_sc)

predictions_lr.shape
# Make submission table

sub_lr = pd.read_csv('/kaggle/input/bank-marketing-prediction/sample_submission.csv')

sub_lr['y'] = predictions_lr



# Save as file

sub_lr.to_csv('submission_lr.csv', index=False)



sub_lr.head()
from lightgbm import LGBMClassifier



# Create model

lgb = LGBMClassifier()

# I use default parameters. Better hyperparameters can be found using GridSearchCV, RandomSearchCV or bayes optimisation



# Measure model performance

lgb_f1 = cross_validate(lgb, X_train, y_train, cv=5, scoring='f1')['test_score'].mean()

print('f1 score: {:.03f}'.format(lgb_f1))



# Fit gradient boosting model

lgb.fit(X_train, y_train)



predictions_lgb = lgb.predict(X_test)

predictions_lgb.shape
from lightgbm import plot_importance



plot_importance(lgb, max_num_features=-1, height=0.5, grid=False, figsize=(5,15));
# Make submission table

sub_lgb = pd.read_csv('/kaggle/input/bank-marketing-prediction/sample_submission.csv')

sub_lgb['y'] = predictions_lgb



# Save as file

sub_lgb.to_csv('submission_lgb.csv', index=False)



sub_lgb.head()