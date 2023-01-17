# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer #use KNN for missing values

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv') #training data

test_data = pd.read_csv('/kaggle/input/titanic/test.csv') #test data
df.head() # check the first 5 rows in training data
test_data.head()
df['Ticket'].value_counts()
plt.figure(figsize=(5,5))

df['Sex'].value_counts().plot(kind='bar')
plt.figure(figsize=(10,10))

sns.barplot(x=df['Age'].value_counts().index, y=df['Age'].value_counts().values) # check distribution of age
plt.figure(figsize=(5,5))

df['Survived'].value_counts(normalize=True).plot(kind='bar')
df['Embarked'].value_counts().plot(kind='bar') # check distribution of Embarked. We notice S is more dominant over other values
df.isnull().sum() # check total null values for each column
df.isnull().sum().sum() # total number of null values 
df.drop(columns='Cabin', inplace=True) # too many missing values

test_data.drop(columns='Cabin', inplace=True) #since we are dropping drom training set, makes no sense to have in test data
imputed = KNNImputer()
def map_embarked_to_int(embarked_df):

    return embarked_df.map({'S':0, 'C':1, 'Q':2})
def map_sex_to_int(sex_df):

    return sex_df.map({'male':0, 'female':1})
def fix_missing_value_using_knn(age_df):

    age_array = age_df.values.reshape(-1, 1)

    return imputed.fit_transform(age_array)
df['Age'] = fix_missing_value_using_knn(df['Age']) # replace missing values of Age using KNN 

test_data['Age'] = fix_missing_value_using_knn(test_data['Age']) # just like we did for the training data
df.isnull().sum()
df['Embarked'].fillna('S', inplace=True) # since 'S' dominate the Embarked column
df['Embarked'] = map_embarked_to_int(df['Embarked']) # convert category to number (Could have used The LabelEncoder)

test_data['Embarked'] = map_embarked_to_int(test_data['Embarked'])
df.isnull().sum().sum() ## no more null values
test_data.isnull().sum()
test_data['Fare'].fillna(method='ffill', inplace=True) # replace Null value using the ffill method. Dont think this will make any huge difference
test_data.isnull().sum().sum() # no more null values in test data
df['Family'] = df['SibSp'] + df['Parch'] + 1 # family-> siblings + parents/children + you 

test_data['Family'] = test_data['SibSp'] + test_data['Parch'] + 1 # we do the same for test data
df.drop(columns=['Parch', 'SibSp'], inplace=True) # we drop Parch and SibSp columns since we dont need it anymore

test_data.drop(columns=['Parch', 'SibSp'], inplace=True)
df.set_index('PassengerId', inplace=True) # PassengerId is unique and wont be used in the training data

test_data.set_index('PassengerId', inplace=True)
df['Sex'] = map_sex_to_int(df['Sex']) # convert categorical Values to int

test_data['Sex'] = map_sex_to_int(test_data['Sex'])
df.drop(columns='Name', inplace=True)

test_data.drop(columns='Name', inplace=True) # we wont use Name in training set
df.drop(columns='Ticket', inplace=True) # we dont need ticket number

test_data.drop(columns='Ticket', inplace=True)
df['Age'] = pd.qcut(df['Age'], q=3, labels=[0, 1, 2]) #change Age to category

test_data['Age'] = pd.qcut(test_data['Age'], q=3, labels=[0, 1, 2])
target = df['Survived'] # separate survived column from training data set
train = df.drop(columns=['Survived'])
train
# Split the data into training and testing sets

train_X, test_X, train_Y, test_Y = train_test_split(train, target, test_size = 0.2, random_state = 1)
param_grid = { 

    'n_estimators': [200, 500, 1000],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6],

    'criterion' :['gini', 'entropy']

}

rf_gsv = GridSearchCV(RandomForestClassifier(),param_grid=param_grid, cv=5)

rf_gsv.fit(train_X, train_Y)

random_forest = rf_gsv.best_estimator_
params = {

         "solver" : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

         "C" : [0.3, 0.5, 1.0]

         }

lr_gsv = GridSearchCV(LogisticRegression(), param_grid=params, cv=5)

lr_gsv.fit(train_X, train_Y)

logistic_regression = lr_gsv.best_estimator_
for column_name, importance in zip(train_X.columns, random_forest.feature_importances_):

    print(f'Variable : {column_name}, \t importance : {importance}')
params = {"C": [0.1, 0.3, 0.5, 0.7, 1.0]}

svm_gs = GridSearchCV(svm.SVC(), param_grid=params, cv=3)

svm_gs.fit(train_X, train_Y)

svm_clf = svm_gs.best_estimator_
params = {"n_neighbors": [3, 5, 8, 10],

         "weights": ['uniform', 'distance'], 

         "leaf_size": [5, 10, 20, 30, 50],

         "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']}

gs_knn = GridSearchCV(KNeighborsClassifier(), param_grid=params, cv=5)

gs_knn.fit(train_X, train_Y)

knn = gs_knn.best_estimator_

print(knn)
result_svm = svm_clf.predict(test_X)

result_knn = knn.predict(test_X)

result_random_forest = random_forest.predict(test_X)

result_logistic_regression = logistic_regression.predict(test_X)
print(f'Random Forest Accuracy: {accuracy_score(test_Y, result_random_forest)}')

print(f'Logistic Regression Accuracy: {accuracy_score(test_Y, result_logistic_regression)}')

print(f'SVM Accuracy: {accuracy_score(test_Y, result_svm)}')

print(f'K Nearest Neighbors Accuracy: {accuracy_score(test_Y, result_knn)}')
survived_logistic_regression = logistic_regression.predict(test_data)

survived_random_forest = random_forest.predict(test_data)

survived_svm = svm_clf.predict(test_data)

survived_knn = knn.predict(test_data)
test_copy = test_data.copy()

test_copy['Survived'] = survived_random_forest

test_copy = test_copy['Survived']

test_copy.to_csv('/kaggle/working/submission01.csv')
test_copy = test_data.copy()

test_copy['Survived'] = survived_logistic_regression

test_copy = test_copy['Survived']

test_copy.to_csv('/kaggle/working/submission02.csv')
test_copy = test_data.copy()

test_copy['Survived'] = survived_svm

test_copy = test_copy['Survived']

test_copy.to_csv('/kaggle/working/submission03.csv')
test_copy = test_data.copy()

test_copy['Survived'] = survived_knn

test_copy = test_copy['Survived']

test_copy.to_csv('/kaggle/working/submission04.csv')