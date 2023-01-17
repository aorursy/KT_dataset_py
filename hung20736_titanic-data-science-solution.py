import pandas as pd 

import numpy as np

import random as rnd

import re

import os



#visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron, SGDClassifier
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname,filename))
pd.set_option('max_rows',10)
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data

test_data.select_dtypes(include='object').describe()
test_data
test_data.describe()
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data
pd.set_option('max_rows',20)

train_data.dtypes
train_data.describe()
train_data.select_dtypes(include='object').describe()
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
train_data[['Parch','Survived']].groupby(['Parch'],as_index=False).mean()
train_data[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean()
train_data[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean()
train_data['Ticket'].unique()
# We can categorize Ticket into: ticket with prefix: PC, STON, CA, C.A, A/, SOTON
test_data = test_data.drop(columns=['Cabin','PassengerId'], axis= 1)

train_data = train_data.drop(columns=['Cabin','PassengerId'], axis=1)
full_data = [train_data, test_data]
train_data
train_data.describe()
train_data.select_dtypes(include='object').describe()
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train_data.select_dtypes(include='object').describe()
train_data['Age'][np.isnan(train_data['Age'])]
for dataset in full_data:

    age_mean = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_missing = sum(dataset['Age'].isnull())

    age_random_generate =  np.random.randint(age_mean - age_std, age_mean + age_std, age_missing)

    dataset['Age'][dataset['Age'].isnull()] = age_random_generate
test_data.select_dtypes(include='object').describe()
test_data[test_data['Fare'].isnull()]
test_data['Fare'][test_data['Fare'].isnull()] = test_data['Fare'].mean()
original = train_data

original
train_data['CategoricalAge'] = pd.cut(train_data['Age'], 5)

test_data['CategoricalAge'] = pd.cut(test_data['Age'], 5)

train_data['CategoricalFare'] = pd.qcut(train_data['Fare'], 4)

test_data['CategoricalFare'] = pd.qcut(test_data['Fare'], 4)
train_data[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()
train_data[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()
train_data
train_data = train_data.drop(['CategoricalAge', 'CategoricalFare'],axis =1)

test_data = test_data.drop(['CategoricalAge', 'CategoricalFare'],axis =1)

train_data
def get_title(name):

    match = re.search("([a-zA-z]+)\.", name)

    if match:

        return match.group(1)

    return ''
print(get_title('Braund, Mr. Owen Harris'))
train_data['Title'] = train_data['Name'].apply(get_title)

test_data['Title'] = test_data['Name'].apply(get_title)
train_data
train_data[['SibSp','Survived']].groupby(['SibSp']).mean()
train_data[['Parch','Survived']].groupby(['Parch']).mean()
train_data['HaveMoreThan_3_par_ch'] =  train_data['Parch'] > 3 

train_data['HaveMoreThan_2_sibsp'] = train_data['SibSp'] > 2 
train_data[['HaveMoreThan_3_par_ch','Survived']].groupby(['HaveMoreThan_3_par_ch']).mean()
train_data[['HaveMoreThan_2_sibsp','Survived']].groupby(['HaveMoreThan_2_sibsp']).mean()
test_data['HaveMoreThan_3_par_ch'] = test_data['Parch'] > 3

test_data['HaveMoreThan_2_sibsp'] = test_data['SibSp'] > 2 

test_data
train_data_remain_ticket = train_data

train_data = train_data.drop(columns=['SibSp','Parch','Ticket','Name'], axis = 1)

train_data
test_data_remain_ticket = test_data

test_data = test_data.drop(columns=['SibSp','Parch','Ticket','Name'], axis = 1)

test_data
pd.set_option('max_rows',50)

(pd.crosstab(train_data['Title'], train_data['Sex']))
pd.set_option('max_rows',11)

full_data = [train_data, test_data]
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
full_data = [train_data, test_data]
for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']      

    

    #Mapping HaveMoreThan_3_par_ch and HaveMoreThan_2_sibsp

    dataset['HaveMoreThan_3_par_ch'] = dataset['HaveMoreThan_3_par_ch'].map({False:0, True:1}).astype(int)

    dataset['HaveMoreThan_2_sibsp'] = dataset['HaveMoreThan_2_sibsp'].map({False:0, True:1}).astype(int)
train_data
X = train_data.drop(['Survived'],axis=1)

y = train_data['Survived']

X
one_hot_encoded_categorical_data = pd.get_dummies(X[['Embarked','Title']])
one_hot_encoded_data = one_hot_encoded_categorical_data.join(X.drop(['Embarked','Title'],axis=1))

one_hot_encoded_data
one_hot_encoded_categorical_data_test = pd.get_dummies(test_data[['Embarked','Title']])

one_hot_encoded_data_test = one_hot_encoded_categorical_data_test.join(test_data.drop(['Embarked','Title'],axis=1))

one_hot_encoded_data_test
full_data = [train_data, test_data]

train_data
for dataset in full_data:

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    
train_data
manmap_train_data = train_data.drop(['Survived'],axis=1)

y = train_data['Survived']

manmap_train_data
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
xg_pipeline = Pipeline([('xgbcl', XGBClassifier(random_state=1) )])

xgb_model = XGBClassifier()
X_train, X_val, y_train, y_val = train_test_split(one_hot_encoded_data, y, test_size=0.2)
param_grid={

    'n_estimators' : [5, 10, 50],

    'learning_rate' : [ 0.1, 0.3, 1],

#     'n_jobs' : 2

}



fit_params={

    'eval_set':[(X_val, y_val)],

    'early_stopping_rounds': 10,

    'verbose': False

}
searchCV = GridSearchCV(xgb_model,param_grid, cv=10,  scoring='accuracy')

searchCV.fit(one_hot_encoded_data, y, **fit_params) 
best_params = searchCV.best_params_

best_params
X_train, X_val, y_train, y_val = train_test_split(manmap_train_data, y, test_size=0.2)
fit_params={

    'eval_set':[(X_val, y_val)],

    'early_stopping_rounds': 10,

    'verbose':False

}
searchCV.fit(X_train, y_train, **fit_params) 
best_params2 = searchCV.best_params_

best_params2
help(GridSearchCV)
my_model = XGBClassifier( random_state=1)

my_model.set_params(**best_params)
X = one_hot_encoded_data 

X_test = one_hot_encoded_data_test

X_test
my_model.fit(X,y)
my_model.predict(X_test)
predicted_survived_people = list(my_model.predict(X_test))

orignal_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

PassengerId_submit = list(orignal_test_data['PassengerId'])

submit = pd.DataFrame({'PassengerId':PassengerId_submit, 'Survived': predicted_survived_people})

submit = submit.set_index('PassengerId')

submit.to_csv('my_submit3.csv')
X = manmap_train_data 

X_test = test_data

test_data
my_model.set_params(**best_params2)
my_model.fit(X,y)
predicted_survived_people = list(my_model.predict(X_test))

orignal_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

PassengerId_submit = list(orignal_test_data['PassengerId'])

submit = pd.DataFrame({'PassengerId':PassengerId_submit, 'Survived': predicted_survived_people})

submit = submit.set_index('PassengerId')

submit.to_csv('my_submit4.csv')
score: 0.78468
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X,y)
predicted_survived_people = list(rf_model.predict(X_test))

orignal_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

PassengerId_submit = list(orignal_test_data['PassengerId'])

submit = pd.DataFrame({'PassengerId':PassengerId_submit, 'Survived': predicted_survived_people})

submit = submit.set_index('PassengerId')

submit.to_csv('my_submit5.csv')
score: 0.76555