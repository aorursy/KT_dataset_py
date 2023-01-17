# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

cols = train_data.columns

# to make this notebook's output identical at every run

np.random.seed(42)
train_data.head()
# Drop non-relevant columns

train_data = train_data.drop(['Name','Ticket'], axis = 1)

test_data = test_data.drop(['Name','Ticket'], axis = 1)
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
best_score = float("inf")

best_model = None



X = train_data.drop('Survived',axis = 1)

Y = train_data.Survived
# Too missing data in Cabin cols

# Drop non-relevant columns

X = X.drop(['Cabin'], axis = 1)

test_data = test_data.drop(['Cabin'], axis = 1)
X.info()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer



def Model(model):

    num_pipeline = Pipeline(steps=[

            ('imputer', SimpleImputer(strategy="median")),

            ('scaler', StandardScaler()),

            ])



    cat_pipeline = Pipeline(steps=[

        ('imputer',SimpleImputer(strategy="most_frequent")),

        ('one_hot_encoder', OneHotEncoder(sparse=False)),

        ])



    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

    categorical_features = X.select_dtypes(include=['object']).columns



    preprocessor = ColumnTransformer([

    ("num_pipeline", num_pipeline, numeric_features),

    ("cat_pipeline", cat_pipeline, categorical_features),

    ])



    return Pipeline(steps=[

                          ('preprocessor', preprocessor),

                          ('model', model)])









X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.20, random_state=1)

print(len(X_train), "train +", len(X_valid), "valid +", len(test_data), "test")
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier



model_name = ["Logistic regression", 

              "Naive Bayes",

              "Stochastic Gradient Descent ", 

              "K-Nearest Neighbours",

              "Decision Tree",

              "Random Forest",

              "Support Vector Machine",

              "XGBoost"]



models = [

    LogisticRegression(random_state=0),

    GaussianNB(),

    SGDClassifier(max_iter=1000, tol=1e-3),

    KNeighborsClassifier(n_neighbors=2),

    DecisionTreeClassifier(random_state=0),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    SVC(gamma='auto'),

    XGBClassifier()]
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold



i = 0

for modele in models:

    model = Model(modele)

#     1 -Train test split 

#     model.fit(X_train, Y_train)   

#     prediction_train = model.predict(X_train)

#     prediction_valid = model.predict(X_valid)



    

    print('Model ',model_name[i])

#     2- Cross validation

    skfold = StratifiedKFold(n_splits=10)

    scores = cross_val_score(model, X, Y, cv=skfold)

    train_accuracy = round(scores.mean()* 100, 2) #, scores.std() * 2

    print("Accuracy: %.2f%%" % (train_accuracy))

    

#     predictions = cross_val_predict(model, X_valid, Y_valid, cv=5)

#     print("Validation: ", round(model.score(X_valid, Y_valid) * 100, 2)) #accuracy_score(Y_valid, prediction_valid))

    

    print('-'*40)

    i+=1

    
from sklearn.model_selection import GridSearchCV



model_name = ["Support Vector Machine"]

# Set the parameters by cross-validation

# defining parameter range 

param_grid = {'model__C': [0.1, 1, 10, 100],  

              'model__gamma': [1, 0.1, 0.01, 0.001], 

              'model__kernel': ['rbf','poly', 'linear']}  



clf = GridSearchCV(

        Model(SVC()), 

        param_grid, 

        scoring='accuracy', 

        cv=5,#skfold,

        verbose = 3

    )

clf.fit(X, Y)
best_result = clf.best_score_

print("Best accuracy:",round(best_result*100,2),'%')



print("Best parameters set found on development set:")

print()

print(clf.best_params_)
from sklearn.model_selection import GridSearchCV

import xgboost as xgb



# Set the parameters by cross-validation

# defining parameter range 

param_grid = {

        'model__min_child_weight': [1, 5, 10],

        'model__gamma': [0.5, 1, 1.5, 2, 5],

        'model__subsample': [0.6, 0.8, 1.0],

        'model__colsample_bytree': [0.6, 0.8, 1.0],

        'model__max_depth': [3, 4, 5]

             }  



clf = GridSearchCV(

        Model(xgb.XGBClassifier()), 

        param_grid, 

        scoring='accuracy', 

        cv=5,#skfold,

        verbose = 3,

        refit=True

    )

clf.fit(X, Y)
best_result = clf.best_score_

print("Best accuracy:",round(best_result*100,2),'%')



print("Best parameters set found on development set:")

print()

print(clf.best_params_)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



# Set the parameters by cross-validation

# defining parameter range 

param_grid = {

        'model__n_estimators': [120, 140],

        'model__max_depth': [30,50],

        'model__min_samples_split': [2,3],

        'model__min_samples_split': [3,5],

        'model__class_weight': [{0:1, 1:1},{0:1,1:5},{0:1,1:3},'balanced']

             }  



clf = GridSearchCV(

        Model(RandomForestClassifier()), 

        param_grid, 

        scoring='accuracy', 

        cv=5,#skfold,

        verbose = 3,

        refit=True

    )

clf.fit(X, Y)
best_result = clf.best_score_

print("Best accuracy:",round(best_result*100,2),'%')



print("Best parameters set found on development set:")

print()

print(clf.best_params_)
best_model = Model(xgb.XGBClassifier(colsample_bytree= 0.8, gamma= 0.5, max_depth= 5, min_child_weight= 5, subsample= 0.8))

best_model.fit(X, Y)



predicted_classes = best_model.predict(test_data)



output = pd.DataFrame({'PassengerId': test_data.index,

                       'Survived': predicted_classes})



# you could use any filename. We choose submission here

output.to_csv('submission.csv', index=False)
