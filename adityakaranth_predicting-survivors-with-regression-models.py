# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')

matplotlib.rcParams['figure.figsize'] = (10,6)



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn import metrics

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_train.info()
# Get numeric columns

print('Numeric cols :',df_train.select_dtypes(include=[np.number]).columns.values)
# Get Non-numeric columns

print('Categorical cols :',df_train.select_dtypes(exclude=[np.number]).columns.values)
df_train.describe(include='all')
# Check for missing data

sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# Number of missing values

print(df_train.isnull().sum().sort_values(ascending=False))
# Number of missing %

print(round((df_train.isnull().mean()*100).sort_values(ascending=False),2))
df_train['Age'].dropna().plot.hist()
sns.countplot(x='Survived', hue='Sex',data=df_train)
sns.countplot(x='Pclass', hue='Survived',data=df_train)
# Correlation plot

sns.heatmap(df_train.corr(), annot=True, cmap='coolwarm')
df_train_missingAge = df_train[df_train['Age'].isnull()]

print('No of data points with Age missing = {}\n'.format(len(df_train_missingAge)))

print(df_train_missingAge)
df_train_availableAge = df_train[df_train['Age'].notnull()]

print('No of data points with Age NOT missing = {}\n'.format(len(df_train_availableAge)))

print(df_train_availableAge)
# All numerical cols of known Age data points as training data

X_train1 = df_train_availableAge[['Pclass', 'Fare', 'SibSp','Parch']] 



# Age of known data points as Target

y_train1 = df_train_availableAge['Age']



# All numerical cols of missing Age data points as training data

X_test1 = df_train_missingAge[['Pclass', 'Fare', 'SibSp','Parch']]
lr1 = LinearRegression()

lr1.fit(X_train1, y_train1)
predicted_age1 = lr1.predict(X_test1)

predicted_age1 = np.round(predicted_age1,1)

print(predicted_age1)
final_df_train = df_train.copy() #Make a copy of df_train

final_df_train.loc[final_df_train['Age'].isnull(), 'Age'] = predicted_age1

final_df_train.head(20) #df_train with filled Age 
# After filling Age column, we check misiing values again

print(round((final_df_train.isnull().mean()*100).sort_values(ascending=False),2))
# Lot of data in Cabin is missing hence Cabin is not useful, so let's drop

final_df_train.drop('Cabin', axis=1, inplace=True) 
# Dropping 'Name' and 'Ticket' 

final_df_train.drop(['Name', 'Ticket'], axis=1, inplace=True)
print(final_df_train)
Sex1 = pd.get_dummies(final_df_train['Sex'], drop_first=True)

print(Sex1)
Embarked1 = pd.get_dummies(final_df_train['Embarked'], drop_first=True)

print(Embarked1)
final_df_train.drop(['Sex', 'Embarked'], axis=1, inplace=True)
final_df_train = pd.concat([final_df_train, Sex1, Embarked1], axis=1)
print(final_df_train)
df_test.info()
# Get numeric columns

print('Numeric cols :',df_test.select_dtypes(include=[np.number]).columns.values)
# Get Non-numeric columns

print('Categorical cols :',df_test.select_dtypes(exclude=[np.number]).columns.values)
df_test.describe(include='all')
# Check for missing data

sns.heatmap(df_test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# Missing values 

print((df_test.isnull().sum()).sort_values(ascending=False))
# Missing values in %

print((df_test.isnull().mean()*100).sort_values(ascending=False))
df_test['Age'].dropna().plot.hist()
sns.countplot(x='Pclass', hue='Sex',data=df_test)
sns.heatmap(df_test.corr(), annot=True, cmap='coolwarm')
# As only 0.2% Fare values are missing, we can fill those with mean

df_test.loc[df_test['Fare'].isnull(), 'Fare'] = df_test['Fare'].mean()
# We can verify the same

df_test['Fare'].isnull().sum()
df_test_missingAge = df_test[df_test['Age'].isnull()]

print('No of data points with Age missing = {}\n'.format(len(df_test_missingAge)))

print(df_test_missingAge)
df_test_availableAge = df_test[df_test['Age'].notnull()]

print('No of data points with Age NOT missing = {}\n'.format(len(df_test_availableAge)))

print(df_test_availableAge)
# All numerical cols of known Age data points as training data

X_train2 = df_test_availableAge[['Pclass', 'Fare', 'SibSp','Parch']] 



# Age of known data points as Target

y_train2 = df_test_availableAge['Age']



# All numerical cols of missing Age data points as training data

X_test2 = df_test_missingAge[['Pclass', 'Fare', 'SibSp','Parch']]
lr2 = LinearRegression()

lr2.fit(X_train2, y_train2)
predicted_age2 = lr2.predict(X_test2)

predicted_age2 = np.round(predicted_age2,1)

print(predicted_age2)
final_df_test = df_test.copy() #Make a copy of df_test

final_df_test.loc[final_df_test['Age'].isnull(), 'Age'] = predicted_age2

final_df_test.head(20) #df_test with filled Age 
# After filling Age column, we check misiing values again

print(round((final_df_test.isnull().mean()*100).sort_values(ascending=False),2))
# Lot of data in Cabin is missing hence Cabin is not useful, so let's drop

final_df_test.drop('Cabin', axis=1, inplace=True) 
# Let's check again if any data is missing

sns.heatmap(final_df_test.isnull(), yticklabels=False, cbar=False, cmap='viridis') 
# Dropping 'Name' and 'Ticket'

final_df_test.drop(['Name', 'Ticket'], axis=1, inplace=True)
print(final_df_test)
Sex2 = pd.get_dummies(final_df_test['Sex'], drop_first=True)

print(Sex2)
Embarked2 = pd.get_dummies(final_df_test['Embarked'], drop_first=True)

print(Embarked2)
final_df_test.drop(['Sex', 'Embarked'], axis=1, inplace=True)
final_df_test = pd.concat([final_df_test, Sex2, Embarked2], axis=1)
print(final_df_test)
X_train = final_df_train.drop(['Survived','PassengerId'], axis=1)

y_train = final_df_train['Survived'] 

X_test = final_df_test.drop('PassengerId', axis=1)
log = LogisticRegression(max_iter=1000)

param_grid = {'C':[0.1, 1, 10, 100, 1000]}

grid = GridSearchCV(log, param_grid, verbose=3)

grid.fit(X_train, y_train)
print(grid.best_estimator_)
predicted = grid.predict(X_test)

print(predicted)
predicted = pd.DataFrame({'PassengerId':final_df_test['PassengerId'], 'Survived': predicted})

print(predicted)
# Output data file obtained from Kaggle 'gender_submission.csv' as df

# This df obtained from kaggle is not Actual file but an example, anyway let's compare here

print(df)
# Plotting the Actual and Predicted data

fig, ax = plt.subplots(1,2, figsize=(12,8))

sns.countplot(df['Survived'], ax=ax[0])

sns.countplot(predicted['Survived'], ax=ax[1])

ax[0].set(xlabel='Survived', title='Actual')

ax[1].set(xlabel='Survived', title='Predicted')

plt.tight_layout()
# Classification Report

print(metrics.classification_report(df['Survived'],predicted['Survived']))
# Confusion Matrix

print(metrics.confusion_matrix(df['Survived'],predicted['Survived']))
# Accuracy

print(metrics.accuracy_score(df['Survived'],predicted['Survived'])*100)