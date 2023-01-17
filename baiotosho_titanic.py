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
#family_size

#ticket_type

#titles



#family_size

train_data['family_size'] = train_data['SibSp'] + train_data['Parch'] + 1





#titles



def extract_titles(name):

    start = name.index(',')

    stop = name.index('.')

    return name[start + 1: stop].strip()



train_data['titles'] = train_data['Name'].apply(lambda x: extract_titles(x))





# ticket_type

from collections import Counter

ctr = Counter(train_data.Ticket.values)

def simplify_ticket(k):

    if ctr[k] > 1:

        return 'Shared'

    else:

        return 'Single'

        

train_data['ticket_type'] = train_data['Ticket'].apply(lambda x: simplify_ticket(x))





#fare_per_person

def fare_per_person(row):

    if row['ticket_type'] == 'Single':

        return row['Fare']

    else:

        return row['Fare'] / row['family_size']





train_data['fare_per_person'] = train_data.apply(lambda row: fare_per_person(row), axis = 1)

# averge age dictionary

from collections import Counter

titles_count = Counter(train_data.titles)

title_mean_age = dict()

default_age = train_data.Age.mean()



for title in titles_count:

    title_mean = train_data[train_data.titles == title].Age.mean()

    if np.isnan(title_mean):

        title_mean_age[title] = default_age

    else:

        title_mean_age[title] = np.round(title_mean, 2)





def fill_age_by_title(row):

    if np.isnan(row['Age']):

        if row['titles'] in titles_count:

            return title_mean_age[row['titles']]

        else:

            return default_age

    else:

        return row['Age']

    

train_data.Age = train_data.apply(lambda row: fill_age_by_title(row), axis = 1) 
column_expr = ['Sex', 'Embarked', 'titles', 'ticket_type']

num_cols = list(train_data.select_dtypes(include = [np.number]).drop(columns = ['Survived'])) #'PassengerId' not dropped

X = pd.concat([train_data.select_dtypes(include = [np.number]).drop(columns = ['Survived']),

               train_data[column_expr]], axis = 1) 

y = train_data.Survived
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder



rf_clf = RandomForestClassifier(n_estimators = 100, max_depth=5, random_state = 42)



pipeline_num = Pipeline([

    ('simple_imputer', SimpleImputer(strategy = 'mean')),

    ('standard_scaler', StandardScaler())

]) 



pipeline_cat = Pipeline([

    ('simple_imputer',SimpleImputer(strategy = 'most_frequent')),

    ('onehotencode', OneHotEncoder(handle_unknown="ignore"))

])



full_transformer_prepare = ColumnTransformer([

    ('num_transformer', pipeline_num, num_cols),

    ('cat_transformer', pipeline_cat, column_expr)

])





full_transformer_prepare_predict = Pipeline([

    ('full_prep', full_transformer_prepare),

    ('predict', rf_clf)

])
full_transformer_prepare_predict.fit(X, y)
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')



# Feature Engineering

test_data['family_size'] = test_data['SibSp'] + test_data['Parch'] + 1

test_data['titles'] = test_data['Name'].apply(lambda x: extract_titles(x))

ctr = Counter(test_data.Ticket.values)

test_data['ticket_type'] = test_data['Ticket'].apply(lambda x: simplify_ticket(x))



test_data['fare_per_person'] = test_data.apply(lambda row: fare_per_person(row), axis = 1)

#End Feature Engineering



# fill age manually

titles_count = Counter(train_data.titles)

title_mean_age = dict()

default_age = test_data.Age.mean()



for title in titles_count:

    title_mean = test_data[test_data.titles == title].Age.mean()

    if np.isnan(title_mean):

        title_mean_age[title] = default_age

    else:

        title_mean_age[title] = np.round(title_mean, 2)



test_data.Age = test_data.apply(lambda row: fill_age_by_title(row), axis = 1) 

#end fill age





X_test = pd.concat([test_data.select_dtypes(include = [np.number]),

               test_data[column_expr]], axis = 1) 



predictions =  full_transformer_prepare_predict.predict(X_test)





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':predictions})



output.to_csv('submission_21.csv', index = False)

print("Your submission was successfully saved!")
