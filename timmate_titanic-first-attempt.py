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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data.head()
test_data.head()
#data_about_me = [10001, 3, 'X, Mr. Y', 'male', 21, 0, 0, '301922', test_data.Fare.median(), 'BE73', test_data.Embarked.mode()[0]]

# df_about_me = pd.DataFrame([data_about_me], columns=test_data.columns)

# test_data = test_data.append(df_about_me, ignore_index=True)

# test_data.tail()
train_data.isnull().sum()
test_data.isnull().sum()
# Drop unhelpful features like "Cabin", etc.

features_to_drop = ['Cabin', 'Ticket', 'PassengerId']

train_data = train_data.drop(features_to_drop, axis=1)

test_data_id = test_data.PassengerId

test_data = test_data.drop(features_to_drop, axis=1)



# Fill gaps for missing values with a median value (for quantitative features) or a mode (for qualitative features)

all_data = train_data.drop(['Survived'], axis=1).append(test_data)



train_data = train_data.fillna({'Age': all_data['Age'].median(), 

                                'Embarked': all_data['Embarked'].mode()[0]})



test_data = test_data.fillna({'Age': all_data['Age'].median(),

                             'Fare': all_data['Fare'].median()})
train_data.isnull().sum()
test_data.isnull().sum()
train_data.head()
# Extract titles from names

def extract_title(name):

    title = (name.split(', ')[1]).split('. ')[0]

    

    if title not in ['Mr', 'Mrs', 'Miss', 'Master']:

        return 'Rare'

    else:

        return title

    

train_data['Title'] = train_data.Name.apply(extract_title)

test_data['Title'] = test_data.Name.apply(extract_title)

train_data.head()
# Create 'Family_Size' feature by adding values of 'SibSp' feature to 'Parch' feature + 1

train_data['Family_Size'] = train_data.SibSp + train_data.Parch + 1

test_data['Family_Size'] = test_data.SibSp + test_data.Parch + 1

train_data.head()
# Create 'Is_Alone' feature (1 if 'Family_Size' == 1, 0 otherwise)

train_data['Is_Alone'] = (train_data.Family_Size == 1).astype(int)

test_data['Is_Alone'] = (test_data.Family_Size == 1).astype(int)

train_data.head()
# # Add 'Name_Length' feature

# train_data['Name_Length'] = train_data.Name.apply(len)

# test_data['Name_Length'] = test_data.Name.apply(len)

# train_data.head()
# # Check if there is a correlation between name length and survival rate

# import matplotlib.pyplot as plt



# plt.subplot()

# plt.hist([train_data[train_data['Survived'] == 1]['Name_Length'], 

#          train_data[train_data['Survived'] == 0]['Name_Length']], 

#          stacked=True, color = ['g','r'], label = ['Survived','Dead'], alpha=0.5)

# plt.title('Name Length Histogram by Survival')

# plt.xlabel('Name Length (chars)')

# plt.ylabel('# of Passengers')

# plt.legend()
# train_data.corr()
# Drop features that we don't need anymore

features_to_drop = ['Name', 'SibSp']

train_data.drop(features_to_drop, axis=1, inplace=True, errors='ignore')

test_data.drop(features_to_drop, axis=1, inplace=True, errors='ignore')

train_data.head()
# Encode 'Sex' feature (1 - male, 0 - female)

encode_sex = lambda sex: 1 if sex == 'male' else 0

train_data['Sex_Code'] = train_data['Sex'].apply(encode_sex)

test_data['Sex_Code'] = test_data['Sex'].apply(encode_sex)

train_data.head()
# Divide age to 5 groups (use pd.cut for equal-spaced intervals)

train_data['Age_Code'] = pd.cut(train_data.Age, 5, retbins=True, labels=False)[0]

test_data['Age_Code'] = pd.cut(test_data.Age, 5, retbins=True, labels=False)[0]

train_data.head()
# Divide 'Fare' feature to 5 groups (use pd.qcut for equal-filled bins)

train_data['Fare_Code'] = pd.qcut(train_data.Fare, 5, retbins=True, labels=False)[0]

test_data['Fare_Code'] = pd.qcut(test_data.Fare, 5, retbins=True, labels=False)[0]

train_data.head()
# Alternative way to encode 'Embarked' and 'Title' features (dummy encoding)

train_data = pd.get_dummies(train_data.drop(['Sex'], axis=1, errors='ignore'))

test_data = pd.get_dummies(test_data.drop(['Sex'], axis=1, errors='ignore'))

train_data.head()
train_data.shape, test_data.shape
test_data.head()
# # Encode 'Embarked' feature

# from sklearn.preprocessing import LabelEncoder



# train_data

# train_data['Embarked_Code'] = LabelEncoder().fit_transform(train_data.Embarked)

# test_data['Embarked_Code'] = LabelEncoder().fit_transform(test_data.Embarked)



# train_data.head()
# # Encode 'Title' feature

# train_data['Title_Code'] = LabelEncoder().fit_transform(train_data.Title)

# test_data['Title_Code'] = LabelEncoder().fit_transform(test_data.Title)

# train_data.head()
# Drop features that we don't need anymore

features_to_drop = ['Sex', 'Age', 'Fare', 'Embarked', 'Title']

train_data.drop(features_to_drop, axis=1, inplace=True, errors='ignore')

test_data.drop(features_to_drop, axis=1, inplace=True, errors='ignore')

train_data.head()



# features_to_drop = ['Sex', 'Embarked', 'Title']

# train_data.drop(features_to_drop, axis=1, inplace=True, errors='ignore')

# Training a decision tree classifier (DTC)from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV



X = train_data.drop(['Survived'], axis=1)

y = train_data.Survived



clf = DecisionTreeClassifier()

param_grid = {'max_depth': range(3, 8, 1),

             'min_samples_split': range(2, 5, 1),

             'min_samples_leaf': range(3, 8, 1),

             'criterion': ['entropy']}



search = GridSearchCV(clf, param_grid, verbose=True, n_jobs=-1)

_ = search.fit(X, y)



print('DTC mean CV score: %0.4f' % cross_val_score(search.best_estimator_, X, y).mean())



# predictions = model.predict(X_test)
search.best_params_
# # Training a random forest

# clf = RandomForestClassifier()



# y = train_data.Survived



# param_grid = {'n_estimators': range(50, 101, 50),

#               'max_depth': range(3, 8, 2), 

#               'criterion': ['entropy'], 

#               'min_samples_leaf': range(3, 16, 4), 

#               'min_samples_split': range(5, 26, 4)

#              }



# search = GridSearchCV(clf, param_grid, verbose=True, n_jobs=-1)

# _ = search.fit(X, y)



# print('RFC mean CV score: %0.4f' % cross_val_score(search.best_estimator_, X, y).mean())
search.best_params_
# Training an SVC

from sklearn.svm import SVC

svc = SVC()



y = train_data.Survived



param_grid = {'C': np.arange(0.1, 10.1, 0.05),

             #'kernel': ['linear', 'poly', 'rbf', 'sigmoid']#, 'precomputed']

             }



search = GridSearchCV(svc, param_grid, verbose=True, n_jobs=-1)

_ = search.fit(X, y)



print('SVC mean CV score: %0.4f' % cross_val_score(search.best_estimator_, X, y).mean())
search.best_params_
test_data.tail()
# Get predictions from the best model (SVM)

clf = search.best_estimator_

X_test = test_data

predictions = clf.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data_id,

              'Survived': predictions

             })





output.to_csv('my_submission5.csv', index=False)

print('Done')
output.tail()