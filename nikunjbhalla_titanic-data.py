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
titanic_data  = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_data.head()



X_test = pd.read_csv('/kaggle/input/titanic/test.csv')
#Assumption : Name, ticket number and fare will not be of much importance to the outcome

columns_to_drop = ['Name','Ticket','Fare']

titanic_data = titanic_data.drop(columns_to_drop, axis=1)
titanic_data.info()
#Age, Cabin and Embarked columns have missing values

titanic_data['Embarked'].value_counts()
#As value of Cabin is present in only 204 of 889 rows, it can be droppeed too, 

#imputation wont be of value as more than half  of rows doesnt have the value and data may get biased

titanic_data = titanic_data.drop(['Cabin'], axis=1)
#Age is a numerical, while embarked is a categorical type of columns.

#Need to impute both of them separately to handle missing values in daa
y = titanic_data.Survived

X = titanic_data

X.drop(['Survived'], axis=1, inplace=True)



from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
X_train.head()
categorical_cols = ['Sex','Embarked']



numerical_cols = ['Age']



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

def get_score(n_estimators):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', RandomForestClassifier(n_estimators=n_estimators,

                                                              random_state=0))

                             ])

    scores = cross_val_score(my_pipeline, X_train, y_train,

                              cv=5)

    print(n_estimators,  ' : ',scores.mean())

    return scores.mean()
results = dict()

for est in range(50,500,50):

    results[est] = get_score(est)
best_estimator = min(results, key=results.get)

print('best_estimator is', best_estimator, 'with MAE =', results[best_estimator])
import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(results.keys(), results.values())

plt.show()
#from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', RandomForestClassifier(n_estimators=best_estimator, max_features=3))

                             ])
my_pipeline.fit(X_train, y_train)



preds = my_pipeline.predict(X_valid)
from sklearn.metrics import accuracy_score  

scores_classification = accuracy_score(y_valid, preds)
scores_classification
X_train.columns
X_test.columns
X_test = X_test.drop(['Name','Ticket','Fare','Cabin'], axis=1)
X_test['Survived'] = my_pipeline.predict(X_test)
submit = X_test[['PassengerId','Survived']]

print(len(submit))

submit.to_csv("../working/submit.csv", index=False)