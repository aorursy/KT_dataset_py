import numpy as np

import pandas as pd

import sklearn

import xgboost

import lightgbm



from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold





pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_columns', None)
raw_train_data = pd.read_csv('../input/train.csv')

raw_train_data
(((raw_train_data['Fare'] + 1)**(1/9) - 1) * 9).plot(kind='hist');
def make_train_features(data):

    #data = data.set_index('PassengerId')

    data['Rels'] = data['SibSp'] + data['Parch'] + 1

    features = [pd.get_dummies(data['Sex']), 

                data['Pclass'],

                data['Age'],

                data['SibSp'],

                data['Parch'],

                data['Rels'],

                (((data['Fare'] + 1)**(1/9) - 1) * 9),

                pd.get_dummies(data['Cabin'].str[0]),

                pd.get_dummies(data['Embarked'], prefix='Embarked', prefix_sep='_')]

    features = pd.concat(features, axis=1)

    features.fillna(value=30, inplace=True)

    return features 
train_features = make_train_features(raw_train_data)

display(train_features.head(10))
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression



lgb = lightgbm.LGBMClassifier(boosting_type='gbdt', 

                               num_leaves=7, 

                               max_bin=200,

                               min_data_in_leaf=20,

                               max_depth=-1, 

                               learning_rate=0.5, 

                               n_estimators=200, 

                               objective='binary', 

                               class_weight=None, 

                               min_split_gain=0.001, 

                               min_child_weight=0.01, 

                               subsample=0.2, 

                               subsample_freq=1, 

                               colsample_bytree=0.2, 

                               reg_alpha=1.2, 

                               reg_lambda=5, 

                               random_state=1, 

                               n_jobs=-1, 

                               silent=False)

xgb = xgboost.XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=1000, colsample_bytree=0.2)

rf = RandomForestClassifier(n_estimators=64)

vc = VotingClassifier(estimators=[('lgb', lgb), ('xgb', xgb), ('rf', rf)], voting='hard')



skf = KFold(n_splits=10, shuffle=False)



skf_results = cross_validate(

    lgb, 

    X=train_features, 

    y=raw_train_data['Survived'], 

    cv=skf,

    n_jobs=-1,

    return_train_score=False, 

    verbose=False)



scores = skf_results['test_score']



print("Accuracy: %0.3f" % (scores.mean()))
def make_test_features(data, train_features):

    features = make_train_features(data)

    features_cols = pd.DataFrame(data=None, columns=train_features.columns)

    # Remove columns that weren't in the train set. 

    features = pd.concat([features_cols, features], join='inner', sort=False)

    # Then add missing columns and fill them with zeros.

    features = pd.concat([features_cols, features], sort=False).fillna(value=0)

    return features  
raw_test_data = pd.read_csv('../input/test.csv')

test_features = make_test_features(raw_test_data, train_features)

test_features.head(10)
final_model = lgb.fit(train_features, y=raw_train_data['Survived'])

submission = pd.DataFrame({'PassengerId': raw_test_data['PassengerId'], 'Survived': final_model.predict(test_features)})

submission.to_csv('submission.csv', index=False)