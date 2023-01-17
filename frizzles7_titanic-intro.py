import numpy as np
import pandas as pd
import os

titanic_full_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')
titanic_full_train.head(10)
titanic_full_train.describe(include='all')
fields = ['Survived', 'Age', 'Fare', 'Parch', 'SibSp', 'Pclass']
titanic_full_train[fields].hist(bins=25, figsize=(20,15))
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.impute import SimpleImputer

titanic_y = titanic_full_train.Survived
clf = GradientBoostingClassifier()
titanic_X_colns = ['PassengerId', 'Age', 'Fare',]
titanic_X = titanic_full_train[titanic_X_colns]
my_imputer = SimpleImputer()
imputed_titanic_X = my_imputer.fit_transform(titanic_X)

clf.fit(imputed_titanic_X, titanic_y)
titanic_plots = plot_partial_dependence(clf, features=[1,2], X=imputed_titanic_X, 
                                        feature_names=titanic_X_colns, grid_resolution=10)
y = titanic_full_train.Survived
X = titanic_full_train.drop(['Survived'], axis=1)
X = X.drop(['Name', 'Ticket', 'Cabin'], axis=1)
#use one-hot encoding for categoricals (Sex, Embarked) using get_dummies
OHE_X = pd.get_dummies(X)
#review the data we now have
OHE_X.describe(include='all')
#keep track of what was imputed
X_plus = OHE_X.copy()

cols_with_missing = (col for col in OHE_X.columns 
                                 if OHE_X[col].isnull().any())
for col in cols_with_missing:
    X_plus[col + '_was_missing'] = X_plus[col].isnull()

X_plus = my_imputer.fit_transform(X_plus)
#split the data into training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_plus,
                                                    y, 
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify = y)
#try random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
#try XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb.score(X_test, y_test)
#try logistic regression

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg.fit(X_train, y_train)
lg.score(X_test, y_test)
#try randomized search cv

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': range(8, 20),
    'max_depth': range(6, 10),
    'learning_rate': [.4, .45, .5, .55, .6],
    'colsample_bytree': [.6, .7, .8, .9, 1]
}

gbm = XGBClassifier(n_estimators=10)

xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 
                                    estimator = gbm, scoring = "accuracy", 
                                    verbose = 1, n_iter = 50, cv = 4)

X = np.concatenate([X_train, X_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)
xgb_random.fit(X, y)

print("Best parameters found: ", xgb_random.best_params_)
print("Best accuracy found: ", xgb_random.best_score_)
# Treat the test data in the same way as training data:
# use OHE, then impute and track imputed values
OHE_X_submit = pd.get_dummies(titanic_test)
OHE_X, OHE_X_submit = OHE_X.align(OHE_X_submit,
                                  join='inner', 
                                  axis=1)

OHE_X_submit_plus = OHE_X_submit.copy()
cols_with_missing = (col for col in OHE_X.columns 
                                 if OHE_X[col].isnull().any())
for col in cols_with_missing:
    OHE_X_submit_plus[col + '_was_missing'] = OHE_X_submit_plus[col].isnull()
OHE_X_submit_plus = my_imputer.transform(OHE_X_submit_plus)
# Use the model to make predictions
xgb_pred = xgb_random.predict(OHE_X_submit_plus)
submission = pd.concat([titanic_test.PassengerId, pd.DataFrame(xgb_pred)], axis = 'columns')
submission.columns = ["PassengerId", "Survived"]
submission.to_csv('titanic_submit.csv', header=True, index=False)