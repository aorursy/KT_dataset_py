import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.model_selection import cross_val_score, KFold



from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head(10)
train.isna().sum(axis=0)
test.isna().sum(axis=0)
X_train = train.drop('Survived', axis = 1)

y_train = train.iloc[:,1]

X_test = test
num_features = ['Age']

cat_features = ['Sex', 'Pclass']
num_transformer = Pipeline(

steps=[

    ('imputer', SimpleImputer(strategy = 'mean')),

    ('scaler', MinMaxScaler())

    ]

)



cat_transformer = Pipeline(

steps=[

    ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))

    ]

)



preprocessor = ColumnTransformer(

    transformers = [

        ('num', num_transformer, num_features),

        ('cat', cat_transformer, cat_features)

    ]

)
lr_pipe = Pipeline(

steps = [

    ('preprocessor', preprocessor),

    ('classifier', LogisticRegression(solver = 'lbfgs'))    

]

)
lr_pipe.fit(X_train, y_train)

print(lr_pipe.score(X_train, y_train))
test_pred = lr_pipe.predict(X_test)

print(test_pred[:10])
from sklearn.model_selection import cross_val_score

cv_results = cross_val_score(lr_pipe, X_train,y_train, cv=10, scoring='accuracy')

print('Results by fold:\n', cv_results)

print()

print('Mean CV Accuracy:', np.mean(cv_results))
np.random.seed(1)

dt_pipe = Pipeline(

    steps = [

        ('preprocessor', preprocessor),

        ('classifier', DecisionTreeClassifier(max_depth=6))

    ]

)



cv_results = cross_val_score(dt_pipe, X_train, y_train, cv=10, scoring='accuracy')



print('Mean CV Accuracy:', np.mean(cv_results))
dt_pipe.set_params(classifier__max_depth = 12)

cv_results = cross_val_score(dt_pipe, X_train, y_train, cv = 10, scoring = 'accuracy')

print('Mean CV Accuracy:', np.mean(cv_results))
dt_pipe.fit(X_train, y_train)

print('Training Accuracy:', dt_pipe.score(X_train, y_train))
test_pred = dt_pipe.predict(X_test)

submission = pd.DataFrame({

     'PassengerID': test.PassengerId,

     'Survived': test_pred

    

})

submission.to_csv('my_submission.csv', index = False)

submission.head()