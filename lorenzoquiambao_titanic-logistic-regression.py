# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_columns = None
# pd.options.display.max_rows = None
raw_train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
raw_train_data
# NaN values
# Change: Drop Cabin column


train_nan_sum = raw_train_data.isna().sum()
train_data_count = raw_train_data['PassengerId'].count()
train_nan_perc = train_nan_sum/train_data_count

print(train_nan_sum)

fig = plt.figure(figsize = (15,5))
plt.bar(raw_train_data.columns.values, train_nan_perc)
plt.xlabel('Train Data Columns')
plt.ylabel('Percentage of NaN values')
plt.title('Percentage of NaN values per Train Data Columns')
plt.show()
# Age
# Change: 

# raw_train_data['Age'].describe()

survived_x_age = raw_train_data[raw_train_data['Survived'] == 1]
died_x_age = raw_train_data[raw_train_data['Survived'] == 0]

plt.hist(survived_x_age['Age'], alpha = 0.5, bins = 50)
plt.hist(died_x_age['Age'], alpha = 0.5, bins = 50)
plt.legend(['Survived', 'Died'])
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Number of passengers survived/died based on age')
plt.show()

update_age = raw_train_data['Age'].fillna(-0.5)
update_age = pd.cut(update_age, [-1, 0, 18, 100], labels = ['Missing', 'Child', 'Adult'])

plt.bar(update_age.unique(), update_age.value_counts())
plt.show()
def group_age(age):
    if age < 0 or age == 'NaN':
        return 'Missing'
    elif age < 5:
        return 'Infant'
    elif age < 12:
        return 'Child'
    elif age < 18:
        return 'Teenager'
    elif age < 35:
        return 'Young Adult'
    elif age < 60:
        return 'Adult'
    else:
        return 'Senior'

# or you can do this
def process_age(df, cut_points, label_names):
    df['Age'] = df['Age'].fillna(-0.5)
    df['Age_categories'] = pd.cut(df, cut_points, labels = label_names)
    return df

update_age_2 = raw_train_data['Age'].fillna(-0.5)
update_age_2 = update_age_2.apply(group_age)
plt.bar(update_age_2.unique(), update_age_2.value_counts())
plt.show()
# Pclass, Sex, Embarked
# Insight: Pclass,1 has more survivors, Female sex has more survivors, Passengers came from Cherbourg has more survivors

selected_features = ['Pclass', 'Sex', 'Embarked']

for feature in selected_features:
    pivot = raw_train_data.pivot_table(index = feature, values = 'Survived')
    print(raw_train_data[feature].value_counts())
    pivot.plot.bar()
# Pclass
# Keep: All

plt.bar(raw_train_data['Pclass'].unique(), raw_train_data['Pclass'].value_counts())
plt.xticks(raw_train_data['Pclass'].unique())
plt.xlabel('Pclass - Ticket Class')
plt.ylabel('Count')
plt.title('Number of Passengers per Ticket Class')
plt.show()
# Names
# Change: Split to 3 Columns ['Title', 'Surname', 'Firstname']
# Change: Simplify 'Title' to ['Mr', 'Mrs', 'Miss']
# Change: 

import re

surnames = []
titles = []

for pid, name in zip(raw_train_data['PassengerId'], raw_train_data['Name']):
    name_split = re.split(', |\. ', name)
    surname = name_split[0]
    title = name_split[1]
    
    first_name = re.split(' ', name_split[2])[0]
    if first_name[0] == '(':
        first_name = first_name[1:]
    print(pid, name_split, first_name)
    
    surnames.append(surname)
    titles.append(title)
    
titles_unique = pd.unique(titles)
titles_unique[-2] = 'Countess'
titles_unique[0], titles_unique[1] = titles_unique[1], titles_unique[0]

surnames_unique = pd.unique(surnames)

# print(f'unique titles: {titles_unique} count: {len(titles_unique)}')
# print(f'unique surnames: {surnames_unique} count: {len(surnames_unique)}')


def get_title(full_str, sub_str_list):
    for sub_str in sub_str_list:
        if full_str.find(sub_str) != -1:
            return sub_str
    print(full_str)
    return np.nan

def get_surname(name):
    name_split = re.split(', |\. ', name)
    return name_split[0]

def get_firstname(name):
    name_split = re.split(', |\. ', name)
    first_name = re.split(' ', name_split[2])[0]
    if first_name[0] != '(':
        return first_name
    return first_name[1:]

for pid, name in zip(raw_train_data['PassengerId'], raw_train_data['Name']):
    get_title(name, titles_unique)
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix = column_name)
    df = pd.concat([df, dummies], axis = 1)
    return df

def create_dummies_mult_cols(df, column_name_list):
    for column_name in column_name_list:
        df = create_dummies(df, column_name)
    return df
df = raw_train_data.copy()
df.info()
df = df.drop(['PassengerId'], axis = 1)
df = df.drop(['Cabin'], axis = 1)
df['Age'] = df['Age'].fillna(-0.5)

df['Age_categories'] = df['Age'].apply(group_age)
# or
# cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
# label_names = ['Missing', 'Infant', 'Child', 'Teenager', 'Young Adult', 'Adult', Senior]
# df = process_age(df, cut_points, label_names)

df['Title'] = df['Name'].map(lambda x: get_title(x, titles_unique))
df['Surname'] = df['Name'].apply(get_surname)
df['Firstname'] = df['Name'].apply(get_firstname)
df = df.drop(['Name'], axis = 1)

cols_to_dummies = ['Pclass', 'Sex', 'Age_categories', 'Embarked']
df = create_dummies_mult_cols(df, cols_to_dummies)
df = df.drop(cols_to_dummies, axis = 1)
df = df.drop(['Age'], axis = 1)

df = df.drop(['Ticket'], axis = 1)

cols_reordered = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Title', 'Firstname', 'Surname', 
                  'Sex_male', 'Sex_female', 'Age_categories_Infant', 'Age_categories_Child', 'Age_categories_Teenager', 
                  'Age_categories_Young Adult', 'Age_categories_Adult', 'Age_categories_Senior', 
                  'Age_categories_Missing', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S','Survived']
df = df[cols_reordered]

df.head(30)
df.isna().sum()
df_clean = df.copy()
df_clean
df_clean.to_csv('train_preprocessed.csv', index = False)
df_prep = pd.read_csv('train_preprocessed.csv')
df_prep.head()
df_prep.describe()
df_prep_unscaled = df_prep.iloc[:, :-1]
df_prep_unscaled
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy = True, with_mean = True, with_std = True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean = None
        self.var = None
        
    def fit(self, X, y = None):
        self.scaler.fit(X[self.columns], y)
        self.mean = np.mean(X[self.columns])
        self.var = np.var(X[self.columns])
        return self
    
    def transform(self, X, y = None, copy = None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns = self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis = 1)[init_col_order]
df_prep_unscaled.columns.values
columns_to_scale = ['SibSp', 'Parch', 'Fare']
titanic_scaler = CustomScaler(columns_to_scale)
titanic_scaler.fit(df_prep_unscaled)
df_prep_scaled = titanic_scaler.transform(df_prep_unscaled)
df_prep_scaled
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr_columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Sex_female', 'Age_categories_Infant',
              'Age_categories_Child', 'Age_categories_Teenager', 'Age_categories_Young Adult', 
              'Age_categories_Adult','Age_categories_Senior', 'Age_categories_Missing', 'SibSp',
              'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
lr_input = df_prep_scaled[lr_columns]
lr_target = df_prep.iloc[:, -1]
lr.fit(lr_input, lr_target)
lr.score(lr_input, lr_target)
lr.intercept_
summary_table = pd.DataFrame(columns = ['Feature name'], data = lr_columns)
summary_table['Coef_'] = np.transpose(lr.coef_)
summary_table['Odds_ratio'] = np.exp(summary_table['Coef_'])
summary_table.sort_values(by = ['Odds_ratio'], ascending = False)
from sklearn.model_selection import train_test_split
df_input = lr_input
df_target = lr_target
train_X, test_X, train_Y, test_Y = train_test_split(df_input, df_target, train_size = 0.8, random_state = 1)
print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)
lr = LogisticRegression()
lr.fit(train_X, train_Y)
predictions = lr.predict(test_X)
predict_proba = lr.predict_proba(test_X)
lr.score(train_X, train_Y)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_Y, predictions)
accuracy
from sklearn.model_selection import cross_val_score

lr = LogisticRegression()
scores = cross_val_score(lr, df_input, df_target, cv = 10)
scores
np.mean(scores)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test[test['Fare'].isnull()]
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
test.info()
def clean_data(df):
    df = df.drop(['PassengerId'], axis = 1)
    df = df.drop(['Cabin'], axis = 1)
    df['Age'] = df['Age'].fillna(-0.5)

    df['Age_categories'] = df['Age'].apply(group_age)
    # or
    # cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
    # label_names = ['Missing', 'Infant', 'Child', 'Teenager', 'Young Adult', 'Adult', Senior]
    # df = process_age(df, cut_points, label_names)

    df['Title'] = df['Name'].map(lambda x: get_title(x, titles_unique))
    df['Surname'] = df['Name'].apply(get_surname)
    df['Firstname'] = df['Name'].apply(get_firstname)
    df = df.drop(['Name'], axis = 1)

    cols_to_dummies = ['Pclass', 'Sex', 'Age_categories', 'Embarked']
    df = create_dummies_mult_cols(df, cols_to_dummies)
    df = df.drop(cols_to_dummies, axis = 1)
    df = df.drop(['Age'], axis = 1)

    df = df.drop(['Ticket'], axis = 1)

    cols_reordered = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Title', 'Firstname', 'Surname', 
                      'Sex_male', 'Sex_female', 'Age_categories_Infant', 'Age_categories_Child', 'Age_categories_Teenager', 
                      'Age_categories_Young Adult', 'Age_categories_Adult', 'Age_categories_Senior', 
                      'Age_categories_Missing', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
    df = df[cols_reordered]
    
    return df
test_clean = clean_data(test)
test_clean
columns_to_scale = ['SibSp', 'Parch', 'Fare']
test_scaler = CustomScaler(columns_to_scale)
test_scaler.fit(test_clean)
test_clean_scaled = test_scaler.transform(test_clean)
test_clean_scaled
lr = LogisticRegression()

lr.fit(df_input, df_target)

lr_columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Sex_female', 'Age_categories_Infant',
              'Age_categories_Child', 'Age_categories_Teenager', 'Age_categories_Young Adult', 
              'Age_categories_Adult','Age_categories_Senior', 'Age_categories_Missing', 'SibSp',
              'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
test_predictions = lr.predict(test_clean_scaled[lr_columns])
test_predictions
test_id = test['PassengerId']
submission_df = pd.DataFrame({'PassengerId': test_id, 'Survived': test_predictions})
submission_df
submission_df.to_csv('titanic_submission', index = False)
