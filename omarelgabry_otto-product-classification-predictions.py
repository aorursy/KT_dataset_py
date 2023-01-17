%matplotlib inline
# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import os
os.system("ls ../input")

# get otto & test csv files as a DataFrame
otto_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# preview the data
otto_df.head()
# otto_df.info()
print("----------------------------")
# test_df.info()
# product features

# Plot summation for every product feature
sum_otto = otto_df.sum()
sum_otto.drop(['target', 'id']).order().plot(kind='barh', figsize=(15,20))
# target

# map each class to numerical value from 0 to 8(i.e. 9 classes)
range_of_classes = range(1, 10)
map_values_dic   = {}

for n in range_of_classes:
    map_values_dic['Class_{}'.format(n)] = n - 1

otto_df['target'] = otto_df['target'].map(map_values_dic)

# Plot
sns.countplot(x='target', data=otto_df)
# define training and testing sets

X_train = otto_df.drop(["id", "target"],axis=1)
Y_train = otto_df["target"].copy()
X_test  = test_df.drop("id",axis=1).copy()
# Xgboost 

# Normal way
params = {"objective": "multi:softprob", "num_class": 9}

T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)

gbm = xgb.train(params, T_train_xgb, 20)
Y_pred = gbm.predict(X_test_xgb)

# Using Validation Set
# X_train, X_valid, Y_train, Y_valid = cross_validation.train_test_split(X_train, Y_train, test_size=0.01)

# params = {"objective": "multi:softprob", "num_class": 38}

# T_train_xgb = xgb.DMatrix(X_train, Y_train)
# T_valid_xgb = xgb.DMatrix(X_valid, Y_valid)
# X_test_xgb  = xgb.DMatrix(X_test)
# watchlist = [(T_valid_xgb, 'eval'), (T_train_xgb, 'train')]

# gbm = xgb.train(params, T_train_xgb, 20, evals=watchlist, early_stopping_rounds=10)
# Y_pred = gbm.predict(X_test_xgb)
# Create submission

submission = pd.DataFrame({ "id": test_df["id"]})

i = 0

# Create column name based on target values(see sample_submission.csv)
for num in range_of_classes:
    col_name = str("Class_{}".format(num))
    submission[col_name] = Y_pred[:,i]
    i = i + 1
    
submission.to_csv('otto.csv', index=False)
submission.head()
# target
# It shows the arrangement of the average of predictions for every target(class) in submission

submission.drop(["id"], axis=1).mean().order().plot(kind='barh', figsize=(10,4))