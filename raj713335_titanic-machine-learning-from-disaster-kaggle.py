import os

os.chdir('kaggle/input')
TITANIC_PATH = os.path.join("input", "titanic")

print(os.getcwd())

print(os.path)

import pandas as pd

def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)
train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")
train_data.head()
train_data.info()
train_data.describe()
train_data["Survived"].value_counts()
train_data["Pclass"].value_counts()
train_data["Sex"].value_counts()
train_data["Embarked"].value_counts()
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        #("select_numeric", DataFrameSelector(["AgeBucket", "RelativesOnboard", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])
num_pipeline.fit_transform(train_data)
# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)
from sklearn.preprocessing import OneHotEncoder
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
cat_pipeline.fit_transform(train_data)
from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
X_train = preprocess_pipeline.fit_transform(train_data)
X_train
y_train = train_data["Survived"]
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)
X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)
from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forset_model=forest_clf.fit(X_train, y_train)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()
train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()

test_data["AgeBucket"] = test_data["Age"] // 15 * 15

train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()


test_data["RelativesOnboard"] = test_data["SibSp"] + test_data["Parch"]
# test_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()
train_data
num_pipeline = Pipeline([
        #("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("select_numeric", DataFrameSelector(["AgeBucket", "RelativesOnboard", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])


cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
X_train = preprocess_pipeline.fit_transform(train_data)
X_test=preprocess_pipeline.fit_transform(test_data)
test_data
# preds=forset_model.predict(X_test)
submission = pd.DataFrame({'PassengerId':test_data['PassengerId'], 'Survived':preds})
submission.to_csv('submission.csv', index=False)
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
import seaborn as sns
from collections import Counter



# dtrain = lgb.Dataset(train[feature_cols], label=train['Stay'])
# dvalid = lgb.Dataset(valid[feature_cols], label=valid['Stay'])

#param = {'num_leaves': 64, 'objective': 'multiclass'}
params = {}
params['learning_rate'] = 0.05
params['max_depth'] = 18
params['n_estimators'] = 3000
params['objective'] = 'binary'
params['boosting_type'] = 'gbdt'
params['subsample'] = 0.7
params['random_state'] = 42
params['colsample_bytree']=0.7
params['min_data_in_leaf'] = 55
params['reg_alpha'] = 1.7
params['reg_lambda'] = 1.11
params['class_weight']: {0: 0.62, 1: 0.38}



clf = lgb.LGBMClassifier()


lightgbm_scores = cross_val_score(clf, X_train, y_train, cv=10)
# clf.fit(train[feature_cols], train['Stay'], early_stopping_rounds=100, eval_set=[(valid[feature_cols], valid['Stay']),
#         (test[feature_cols], test['Stay'])], eval_metric='multi_error', verbose=True)

# eval_score = accuracy_score(test['Stay'], clf.predict(test[feature_cols]))

# print('Eval ACC: {}'.format(eval_score))

lightgbm_scores
valid_fraction = 0.20
valid_size = int(len(X_train) * valid_fraction)

train = X_train[:-2 * valid_size]
valid = X_train[-2 * valid_size:-valid_size]
test = X_train[-valid_size:]


train_lb = y_train[:-2 * valid_size]
valid_lb = y_train[-2 * valid_size:-valid_size]
test_lb = y_train[-valid_size:]
clf.fit(train, train_lb, early_stopping_rounds=100, eval_set=[(valid, valid_lb),
        (test, test_lb)], eval_metric='multi_error', verbose=True)



eval_score = accuracy_score(test_lb, clf.predict(test))

print('Eval ACC: {}'.format(eval_score))
from sklearn.model_selection import GridSearchCV

best_iter = clf.best_iteration_
params['n_estimators'] = best_iter
print(params)

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

clf = lgb.LGBMClassifier(**params)

#clf.fit(X_train, y_train, eval_metric='multi_error', verbose=False)

grid_search = GridSearchCV(clf, param_grid, cv=5, verbose=3)
grid_search.fit(X_train,y_train)

# eval_score_auc = roc_auc_score(df_train[label_col], clf.predict(df_train[feature_cols]))
eval_score_acc = accuracy_score(y_train, grid_search.predict(X_train))

print('ACC: {}'.format(eval_score_acc))
preds=grid_search.predict(X_test)


submission = pd.DataFrame({'PassengerId':test_data['PassengerId'], 'Survived':preds})
submission.to_csv('submission.csv', index=False)
import xgboost

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score




# Define the model
my_model_3 = XGBRegressor(n_estimators=1000, learning_rate=0.15)

# Fit the model
my_model_3.fit(train, train_lb,
             early_stopping_rounds=5,
               eval_set=[(valid, valid_lb),
        (test, test_lb)],
             verbose=False) # Your code here

# Get predictions
#predictions_3 = my_model_3.predict(X_valid)

eval_score = accuracy_score(test_lb, my_model_3.predict(test))

print('Eval ACC: {}'.format(eval_score))
