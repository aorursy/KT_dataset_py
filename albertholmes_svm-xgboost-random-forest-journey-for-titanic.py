__author__ = 'Albert: https://www.kaggle.com/albertholmes'

# File xgb_valid.py

# Use the K-Fold corss validation to do experiment based on titanic data



# Import modules

import numpy as np

import pandas as pd

import xgboost as xgb

import gc

from sklearn.base import TransformerMixin

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier as RF

from matplotlib import pyplot as plt



%matplotlib inline



# Mean Square Error function

mse = lambda actual, pred: np.mean((actual - pred) ** 2)
class DataFrameImputer(TransformerMixin):

    """

    TransformerMixin is an interface that you can create your own

    transformer or models.

    The .fit_transform method that calls .fit and .transform methods,

    you should define the two methods by yourself.

    """

    def fit(self, X, y=None):

        """

        The pandas.Series.value_counts method returns the object

        containing counts of unique values.

        The resulting object will be in descending order so that

        the first element is the most frequently-occurring element.



        np.dtype('O'): The 'O' means the Python objects

        """

        d = [X[c].value_counts().index[0] if X[c].dtype == np.dtype('O')

             else X[c].median() for c in X]



        self.fill = pd.Series(d, index=X.columns)

        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)



# Garbage collection

gc.enable()
# Read the data

train = pd.read_csv('../input/train.csv')



# Get target value

target = train['Survived'].values

del train['Survived']



# Prepare the k-fold cross validation

skf = StratifiedKFold(n_splits=10)

"""

print(skf.get_n_splits(train, target))

"""

# Add new features

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1



feature_names = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']

nonnumeric_fea = ['Sex', 'Embarked']

categorical_fea = ['Pclass', 'Sex', 'Embarked']



# Impute the missing values

imputed = DataFrameImputer().fit_transform(train[feature_names])
"""

Preprocessing the nonnumeric feature

Encode labels with value between 0 and (n_classes - 1)

"""

le = LabelEncoder()

for feature in nonnumeric_fea:

    imputed[feature] = le.fit_transform(imputed[feature])



"""

Use one hot encoder to get new feature from categorical feature

"""

enc = OneHotEncoder()

chosen_features = imputed[categorical_fea]

new_fea = enc.fit_transform(chosen_features).toarray()

for feature in categorical_fea:

    del imputed[feature]



train = imputed.values

train = np.concatenate((train, new_fea), axis=1)
xgb_errors = []

svc_errors = []

rf_errors = []

"""

stack_errors = []

"""

# K-Fold cross validation

for train_idx, valid_idx in skf.split(train, target):

    ensemble_feas = []



    # print('TRAIN:', train_idx, 'VALID:', valid_idx)

    tra, val = train[train_idx], train[valid_idx]

    target_tra, target_val = target[train_idx], target[valid_idx]



    # XGBoost: Train and predict

    gbm = xgb.XGBClassifier(max_depth=5, learning_rate=0.05,

                            n_estimators=300).fit(tra, target_tra)

    pred = gbm.predict(val)

    error = mse(target_val, pred)

    print('XGBoost MSE: %.3f' % (error))



    xgb_errors.append(error)

    ensemble_feas.append(pred)



    # SVC: Train and predict

    clf = SVC(C=200.0, gamma=0.002)

    clf.fit(tra, target_tra)



    pred = clf.predict(val)

    error = mse(target_val, pred)

    print('SVC MSE: %.3f' % (error))



    svc_errors.append(error)

    ensemble_feas.append(pred)



    # Random Forest: Train and predict

    clf = RF(n_estimators=100, max_depth=6)

    clf.fit(tra, target_tra)



    pred = clf.predict(val)

    error = mse(target_val, pred)

    print('RANDOM FOREST MSE: %.3f' % (error))



    rf_errors.append(error)

    ensemble_feas.append(pred)



    # Stacking

    """

    stack_feas = np.array(ensemble_feas).T

    pred = np.array([1 if sum(stack_feas[idx]) > 1 else 0

            for idx in range(stack_feas.shape[0])])

    error = mse(target_val, pred)

    print('Stacking MSE: %.3f' % (error))



    stack_errors.append(error)

    """



xgb_final_error = sum(xgb_errors) / len(xgb_errors)

print('XGBoost FINAL MSE: %.3f' % (xgb_final_error))



svc_final_error = sum(svc_errors) / len(svc_errors)

print('SVC FINAL MSE: %.3f' % (svc_final_error))



rf_final_error = sum(rf_errors) / len(rf_errors)

print('RANDOM FOREST FINAL MES: %.3f' % (rf_final_error))
xgb.plot_importance(gbm)

plt.show()