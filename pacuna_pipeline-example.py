import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, Imputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn_pandas import CategoricalImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
train = pd.read_csv("../input/census.csv")
# numerical
num_cols = ['age', 'education-num', 'capital-gain',
            'capital-loss', 'hours-per-week']

# categorical
cat_cols = ['workclass', 'education_level', 
            'marital-status', 'occupation', 
            'relationship', 'race', 
            'sex', 'native-country']

# need log transform
log_transform_cols = ['capital-loss', 'capital-gain']
# select the categorical columsn
def get_cat_cols(X):
    return X[cat_cols]

# select the numerical columns
def get_num_cols(X):
    return X[num_cols]

# select the columns that need log transform
def get_log_transform_cols(X):
    return X[log_transform_cols]

# one-hot encode the categorical variables
def get_dummies(X):
    return pd.get_dummies(X)

# imputer for empty values in categorical variables.
# note: this is not optimal since we are not using the strategy from train in the test
# sample. Not sure how to accomplish that.
def cat_imputer(X):
    return X.apply(lambda col: CategoricalImputer().fit_transform(col)) 
# log transform
log_transform_pipeline = Pipeline([
 ('get_log_transform_cols', FunctionTransformer(get_log_transform_cols, validate=False)),
 ('imputer', SimpleImputer(strategy='mean')),   
 ('log_transform', FunctionTransformer(np.log1p))
])

# for all the numerical cols fill null values with the mean of the column
# and then apply scaling
num_cols_pipeline = Pipeline([
 ('get_num_cols', FunctionTransformer(get_num_cols, validate=False)),
 ('imputer', SimpleImputer(strategy='mean')),
 ('min_max_scaler', MinMaxScaler())
])

# for all the categorical cols, apply the categorical imputer function
# from the sklearn_pandas library and then one-hot encode using the pandas
# get_dummies function
cat_cols_pipeline = Pipeline([
 ('get_cat_cols', FunctionTransformer(get_cat_cols, validate=False)),
 ('imputer', FunctionTransformer(cat_imputer, validate=False)),
 ('get_dummies', FunctionTransformer(get_dummies, validate=False))
])
steps_ = FeatureUnion([
    ('log_transform', log_transform_pipeline),
    ('num_cols', num_cols_pipeline),
    ('cat_cols', cat_cols_pipeline)
])

# this full pipeline will apply the 3 previous steps
full_pipeline = Pipeline([('steps_', steps_)])
# binarize the target variable
y = train['income'].map({'<=50K': 0, '>50K': 1})

# transform the entire training set.
# this pipeline will be fitted to the training set
# and the test set (for submission) only need to be transformed (not fitted)
X = full_pipeline.fit_transform(train)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = AdaBoostClassifier(n_estimators=300)
clf.fit(X_train, y_train)
probs_train = clf.predict_proba(X_train)[:, 1]
probs_test = clf.predict_proba(X_test)[:, 1]
print("score train: {}".format(roc_auc_score(y_train, probs_train)))
print("score test: {}".format(roc_auc_score(y_test, probs_test)))
test = pd.read_csv("../input/test_census.csv")

# use the pipeline to transform
X_sub = full_pipeline.transform(test)

# rename the first column to id
test['id'] = test.iloc[:,0] 

# make predictions
test['income'] = clf.predict_proba(X_sub)[:, 1]

# generate output file
test[['id', 'income']].to_csv("submission.csv", index=False)