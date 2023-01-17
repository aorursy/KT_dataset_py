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
# read in training and test datasets
X_full = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
X_test_full = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

# preview training dataframe
X_full.head()
# preview test dataframe
X_test_full.head()
# explore high-level data characeristics in training dataset
X_full.info()
# high level info test dataset
X_test_full.info()
# Remove irrelevant columns
X = X_full.copy()
X_test = X_test_full.copy()
cols_to_drop = ['Name', 'Ticket', 'Cabin']
X.drop(cols_to_drop, axis=1, inplace=True)
X_test.drop(cols_to_drop, axis=1, inplace=True)
# check training data
X.info()
# check test data
X_test.info()
# evaluate cardinality of categorical columns
cat_cols = [col for col in X.columns if X[col].dtype=='object']
print('Cardinality of Remaining Categorical Data')
for col in cat_cols:
    print('{}: {}'.format(col, X[col].nunique()))
# import modules
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score

# drop any missing target values rows
X.dropna(subset=['Survived'], inplace=True)

# separate target from predictors
y = X.Survived
X.drop(['Survived'], axis=1, inplace=True)

# split training dataset into training and validation sets for model testing
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8)

# Identify numerical columns and categorical columns to preprocess accordingly
numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

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
# we need classification models to fit, validate and compare
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# xgboost classifier (with grid search for hyperparameter optimization) 
clf = XGBClassifier()
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }
model_XGBC = GridSearchCV(clf,
                    parameters, n_jobs=4,
                    scoring=make_scorer(roc_auc_score),
                    cv=5)

# gradient boosting classifier
model_GBC = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05, random_state=0)

# Ada boost classifier
model_ABC = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)

# random forest classifier
model_RF = RandomForestClassifier(n_estimators=1000, random_state=0)

# logistic regression classifier
model_LR = LogisticRegression(max_iter=1000, random_state=0)

# list of models for comparison in next step
models = [model_ABC, model_GBC, model_RF, model_LR]
# define function to evaluate the F1 score of the model and return overall pipeline and score
def eval_model(model, preproc=preprocessor, X_cv=X, y_cv=y, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                 ])

    # Evaluate the model by cross validation scoring and averaging the scores
#     cv_score = np.mean(cross_val_score(my_pipeline, X, y, cv=5, scoring=make_scorer(f1_score)))
#     cv_score = np.mean(cross_val_score(my_pipeline, X, y, cv=5, scoring=make_scorer(roc_auc_score)))
    
    # fit model
    my_pipeline.fit(X_t, y_t)
    
    # validate and score
    preds = my_pipeline.predict(X_v)
    score = roc_auc_score(y_v, preds)
    
    return my_pipeline, score

# Compare models
pipelines = []
for i, model in enumerate(models):
    model_pipeline, score = eval_model(model)
    pipelines.append(model_pipeline)
    print('ROC-AUC Score of Model {} - {}: {}'.format(i, model.__class__.__name__, score))
# Build final pipeline using the best scoring model above
# final_model = models[0]
# # final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
# #                                   ('model', final_model)
# #                                  ])
# final_pipeline = pipelines[0]

# # Fit model
# # final_pipeline.fit(X, y)
# print('Selected model is {}'.format(final_model.__class__.__name__))

# # predict using final pipeline (w final model) on test dataset
# predictions = final_pipeline.predict(X_test)

# # prepare submission CSV file for competition
# output = pd.DataFrame({'PassengerId': X_test.index, 'Survived': predictions})
# output.to_csv('my_submission.csv', index=False)
# print("Your submission was successfully saved!")
# simple method suggested by Titanic Tutorial (to check if this actually does better than 'gender_submission.csv')
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(X_full[features])
X_test = pd.get_dummies(X_test_full[features])

# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model_XGBC.fit(X, y)
predictions = model_XGBC.predict(X_test)

output = pd.DataFrame({'PassengerId': X_test_full.index, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")