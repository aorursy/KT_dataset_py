from sklearn.metrics import roc_auc_score

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import xgboost as xgb

import matplotlib.pyplot as plt

from mlxtend.feature_selection import ColumnSelector

from sklearn.base import BaseEstimator, ClassifierMixin
# You need to fix this

train = pd.read_csv("/kaggle/input/hackathon-features/train.csv")

print("train shape", train.shape)

test = pd.read_csv("/kaggle/input/hackathon-features/test.csv")

print("test shape", test.shape)



y = train['target']

X = train.drop(columns = ['target','id'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
class MandatoryClassifier(BaseEstimator, ClassifierMixin):

    

    def __init__(self):

        self.model = xgb.XGBClassifier(

            objective='binary:logistic',

            gamma=0.1,

            subsample=0.7,

            n_estimators=200

        )

        

    def fit(self, X, y):

        assert X.shape[1] <= 80, "sorry, only max 80 features allowed"

        assert X.shape[1] <= 80, "sorry, only max 80 features allowed"

        

        self.model = self.model.fit(X,y)

        return self

        

    def predict(self):

        raise AssertionError("Use predict_proba in this competition")

        

    def predict_proba(self, X):

        return self.model.predict_proba(X)

        
def get_rf_feat_importances(X,y):

    rf = RandomForestClassifier(n_estimators=20, random_state = 42)

    rf.fit(X, y)

    df = pd.DataFrame(

        {'feature': X.columns, 'importance':rf.feature_importances_})

    df = df.sort_values(by=['importance'], ascending=False)

    return df



top100 = get_rf_feat_importances(X_train, y_train)
model = make_pipeline(

    ColumnSelector(cols=list(top100.feature[:80])),

    MandatoryClassifier())



model = model.fit(X_train, y_train)
preds = model.predict_proba(X_test[top100['feature']])[:,1]

roc_auc_score(y_test, preds)
preds = model.predict_proba(test)[:,1]

plt.hist(preds,bins=100)

plt.title('Test.csv predictions')

plt.show()
sub = pd.read_csv("/kaggle/input/hackathon-features/sample_submission.csv")

sub['Predicted'] = preds

sub.to_csv("submission.csv", index=False) 