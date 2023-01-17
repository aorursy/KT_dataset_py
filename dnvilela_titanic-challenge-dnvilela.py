# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from xgboost import XGBClassifier as XGB
from sklearn.model_selection import train_test_split as TTS
%matplotlib inline
# Loading data

df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
# Cheking train data

df_train.head()
# Cheking test data

df_test.head()
# Cheking the null or blank values

pd.DataFrame({'Train': df_train.drop('Survived', axis = 1).isna().sum(), 'Test': df_test.isna().sum()})
# First, it is necessary to identify each data set so that they can be segmented after treatment.

df_train['Set'] = 'Train'
df_test['Set'] = 'Test'

# Now, blend.

df = pd.concat([df_train.drop('Survived', axis = 1), df_test])
df.head()
df_train.dtypes
plt.figure(figsize=(10,5))
sb.distplot(df_train[df_train['Survived'] == 0]['Age'].dropna(), color='red')
sb.distplot(df_train[df_train['Survived'] == 1]['Age'].dropna(), color='blue')
plt.ylabel('Passenger count')
plt.title('Age distribution by survived')
plt.legend(['Died', 'Survived'])
plt.show()
plt.figure(figsize=(10,5))
sb.distplot(df_train[df_train['Survived'] == 0]['Fare'].dropna(), color='red')
sb.distplot(df_train[df_train['Survived'] == 1]['Fare'].dropna(), color='blue')
plt.ylabel('Passenger count')
plt.title('Fare distribution by survived')
plt.legend(['Died', 'Survived'])
plt.show()
sb.catplot(x = 'Sex', y = 'Survived', data = df_train, kind = 'bar', color='lightblue')
plt.title('Number of suvivors by sex')
plt.show()
sb.catplot(x = 'Parch', y = 'Survived', data = df_train, kind = 'bar', color='lightblue')
plt.title('Number of suvivors by parch')
plt.show()
sb.catplot(x = 'SibSp', y = 'Survived', data = df_train, kind = 'bar', color='lightblue')
plt.title('Number of suvivors by SibSp')
plt.show()
sb.catplot(x = 'Pclass', y = 'Survived', data = df_train, kind = 'bar', color='lightblue')
plt.title('Number of suvivors by Pclass')
plt.show()
sb.catplot(x = 'Embarked', y = 'Survived', data = df_train, kind = 'bar', color='lightblue')
plt.title('Number of suvivors by Embarked')
plt.show()
# Extract the title

df['Title'] = [s.split(', ')[1].split('.')[0] for s in df['Name']]
# Crate a dictionary of age means by Title

age_dict = df.groupby('Title')['Age'].mean().astype(int).to_dict()
# Replace nan values with the age mean

df['Age'] = [age_dict[t] if pd.isna(a) else a for a, t in zip(df['Age'], df['Title'])]
# Age category

df['Age_Class'] = pd.cut(df['Age'].astype(int), 8, labels=range(8))
df['Age_Class'] = df['Age_Class'].astype(int)
# Family size

df['Family'] = df['SibSp'] + df['Parch'] + 1
# Change de sex values

df['Sex'].replace({'male':1, 'female': 0}, inplace = True)
# Change de embarked values

df['Embarked'].replace({'C':0, 'Q': 1, 'S': 2}, inplace = True)
df['Embarked'].fillna(df['Embarked'].mean(), inplace = True)
# Fare values

df['Fare'].fillna(df['Fare'].mean(), inplace = True)
# Titles

title_dict = {k:i for i,k in enumerate(df.Title.unique())}

df['Title'].replace(title_dict, inplace = True)
# Passenger have a cabin?


df['Cabin'] = [c[0] if not(pd.isna(c)) else 'X' for c in df['Cabin']]

cabin_dict = {k:i for i,k in enumerate(df.Cabin.unique())}

df['Cabin'].replace(cabin_dict, inplace = True)
# Fare by person

df['Cost'] = df['Fare'] / df['Family']
# Is single?

df['Single'] = np.where(df['Family'] == 1, 1, 0)
# Drop the columns and few null values

df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)
df.head()
df.isna().sum()
# Split the data in train and test

df_train_n = pd.concat([df[df['Set'] == 'Train'].drop('Set', axis = 1), df_train['Survived']], axis = 1)
df_test_n = df[df['Set'] == 'Test'].drop(['Set'], axis = 1)
# Getting the vectors

X_train, X_test, y_train, y_test = TTS(df_train_n.drop('Survived', axis = 1), df_train_n['Survived'], test_size=0.33, random_state=0)
# Fit the model on train data

model = XGB(n_estimators = 100).fit(X_train, y_train)
model.score(X_train, y_train)
# Checking the fatures importance

pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
# Predicting 

y_pred_k = model.predict(df_test_n)
# Rounding the probabilities

y_pred_k = np.where(y_pred_k >= 0.5, 1, 0)
# Creating the dataframe to export

df_k = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_pred_k})
df_k.head()
# Exporting the data

df_k.to_csv('Titanic_prediction.csv', sep = ',', index = None)