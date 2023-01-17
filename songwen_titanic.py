import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('../input/train.csv', header=0) 
# female = 0, Male = 1

train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:

    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values



Ports = list(enumerate(np.unique(train_df['Embarked'])))    

Ports_dict = { name : i for i, name in Ports }              

train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     



median_age = train_df['Age'].dropna().median()

if len(train_df.Age[ train_df.Age.isnull() ]) > 0:

    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age



median_fare = train_df['Fare'].dropna().median()

if len(train_df.Fare[ train_df.Fare.isnull() ]) > 0:

    train_df.loc[ (train_df.Fare.isnull()), 'Fare' ] = median_fare

    

train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 



# TEST DATA

test_df = pd.read_csv('../input/test.csv', header=0)        # Load the test file into a dataframe



id_list = test_df['PassengerId']

# I need to do the same with the test data now, so that the columns are the same as the training data

# I need to convert all strings to integer classifiers:

# female = 0, Male = 1

test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:

    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values



Ports = list(enumerate(np.unique(test_df['Embarked'])))    

Ports_dict_test = { name : i for i, name in Ports }              

test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict_test[x]).astype(int)     



median_age = test_df['Age'].dropna().median()

if len(test_df.Age[ test_df.Age.isnull() ]) > 0:

    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age



median_fare = test_df['Fare'].dropna().median()

if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:

    test_df.loc[ (test_df.Fare.isnull()), 'Fare' ] = median_fare



test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 



train_df.info()
# search for the best parameters of random forest

def parameter_evaluate(data):

    clf_ev = RandomForestClassifier()

    x, y = data.drop(['Survived'], axis=1), data['Survived']

    parameters = {'n_estimators': [100, 300], 'max_features': [3, 4, 5, 'auto'],

                  'min_samples_leaf': [9, 10, 12], 'random_state': [7]}

    grid_search = GridSearchCV(estimator=clf_ev, param_grid=parameters, cv=10, scoring='accuracy')

    print("parameters:")

    # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    grid_search.fit(x, y)

    print("Best score: %0.3f" % grid_search.best_score_)

    print("Best parameters set:")

    bsp = grid_search.best_estimator_.get_params()  # the dict of parameters with best score

    for param_name in sorted(bsp.keys()):

        print("\t%s: %r" % (param_name, bsp[param_name]))

    return bsp



#parameters = parameter_evaluate(train_df)  

# we don't need to search everytime after getting best parameters

#parameters = {'n_estimators': 300, 'max_features': 3, 'min_samples_leaf': 9, 'random_state': 7}

parameters = {'n_estimators': 100, 'max_features': 5, 'min_samples_leaf': 10, 'random_state': 7}

rf = RandomForestClassifier(**parameters)

rf.fit(train_df.drop(['Survived'], axis=1), train_df['Survived'])
#train_df

#test_df.info()

results = rf.predict(test_df)

output = pd.DataFrame({'PassengerId': id_list, "Survived": results})

output.to_csv('prediction_1.csv', index=False)
from sklearn.linear_model import LinearRegression, LogisticRegression
lr = LogisticRegression(random_state=7)

lr.fit(train_df.drop(['Survived'], axis=1), train_df['Survived'])
results=lr.predict(test_df)

output = pd.DataFrame({'PassengerId': id_list, "Survived": results})

output.to_csv('prediction_2.csv', index=False)