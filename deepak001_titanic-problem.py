# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load the tarin data and test data

train_df = pd.read_csv(r'/kaggle/input/titanic/train.csv')

test_df = pd.read_csv(r'/kaggle/input/titanic/test.csv')
# Do a quick check on the data

train_df.head()
# Check if the data is skews or not



train_df.Survived.value_counts()
train_df['Survived'].plot.hist()
train_df['Age'].plot.hist()
# Check the data types of tarin data

train_df.dtypes
# Get the columns

train_df.columns
features =['PassengerId','Pclass', 'Name', 'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

targets = ['Survived']
# Get names of columns with missing values

cols_with_missing = [col for col in train_df.columns

                     if train_df[col].isnull().any()]



cols_with_missing
categorical_columns = ['Pclass', 'Sex','Cabin', 'Embarked']

non_categorical_columns = ['Fare','Age', 'SibSp','Parch']
train_df_categorical_cols_imputed = train_df[categorical_columns].apply(lambda x: x.fillna(x.value_counts().index[0]))

test_df_categorical_cols_imputed = test_df[categorical_columns].apply(lambda x: x.fillna(x.value_counts().index[0]))



# df_most_common_imputed
train_df_non_categorical_cols_imputes =train_df[non_categorical_columns].fillna(train_df[non_categorical_columns].mean())

test_df_non_categorical_cols_imputes =test_df[non_categorical_columns].fillna(test_df[non_categorical_columns].mean())
X_train = pd.concat([train_df_categorical_cols_imputed, train_df_non_categorical_cols_imputes], axis=1)

X_test = pd.concat([test_df_categorical_cols_imputed, test_df_non_categorical_cols_imputes], axis=1)
cols_with_missing_train = [col for col in X_train.columns

                     if X_train[col].isnull().any()]



cols_with_missing_train 
cols_with_missing_test = [col for col in X_test.columns

                     if X_test[col].isnull().any()]



cols_with_missing_test
from sklearn.preprocessing import OneHotEncoder



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_columns]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[categorical_columns]))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_test.index = X_test.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(categorical_columns, axis=1)

num_X_test = X_test.drop(categorical_columns, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
# num_val_samples = int(len(features) * 0.2)

train_features = OH_X_train

train_targets = train_df.Survived

test_features = OH_X_test

# val_targets = targets[-num_val_samples:]



print("Number of training samples:", len(train_features))

print("Number of validation samples:", len(test_features))


counts =train_df.Survived.value_counts()

print(

    "Number of positive samples in training data: {} ({:.2f}% of total)".format(

        counts[1], 100 * float(counts[1]) / len(train_targets)

    )

)



weight_for_0 = 1.0 / counts[0]

weight_for_1 = 1.0 / counts[1]
def display_scores(scores):

    print("Scores: ",scores)

    print("Mean:",scores.mean())

    print("Standard deviation:",scores.std())
mean = np.mean(train_features, axis=0)

train_features -= mean

test_features -= mean

std = np.std(train_features, axis=0)

train_features /= std

test_features /= std
from sklearn.model_selection import cross_val_score

from sklearn.metrics import balanced_accuracy_score,make_scorer

balance_accuracy = make_scorer(balanced_accuracy_score)
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

'''

param_grid = [{'class_weight':['balanced'],

'kernel' : ['sigmoid','poly'], # 'linear', 'poly',

'degree' : [4,5,6],

'gamma' : ['auto'],

'random_state':[1,2],

}]



svc_clf= SVC()



grid_search = GridSearchCV(svc_clf,param_grid,cv=5,

                          scoring=balance_accuracy,

                          return_train_score=True)



grid_search.fit(train_features,train_targets)'''
# grid_search.best_params_
# grid_search.best_estimator_
svc_clf = SVC(class_weight='balanced', degree=4, gamma='auto', kernel='sigmoid',

    random_state=1)

svc_clf.fit(train_features, train_targets)

scores = cross_val_score(svc_clf, train_features, train_targets, scoring=balance_accuracy,cv=10)
display_scores(scores)
# from sklearn.model_selection import GridSearchCV



# param_grid = [{'n_estimators':[3,10,30,100, 200],'max_features':[2,4,6,8],'max_leaf_nodes':[5,10,15,20,25]},

#              {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]}]



# forest_model = RandomForestClassifier(random_state=1)



# grid_search = GridSearchCV(forest_model,param_grid,cv=5,

#                           scoring=balance_accuracy,

#                           return_train_score=True)



# grid_search.fit(train_features,train_targets)
# grid_search.best_params_
# grid_search.best_estimator_
# from sklearn.model_selection import GridSearchCV



# param_grid = [{'n_estimators':[3,10,30,100, 200],'max_features':[2,4,6,8],'max_leaf_nodes':[100,200,300]},

#              {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]}]



# forest_model = RandomForestClassifier(random_state=1)



# grid_search = GridSearchCV(forest_model,param_grid,cv=5,

#                           scoring=balance_accuracy,

#                           return_train_score=True)



# grid_search.fit(train_features,train_targets)
# grid_search.best_params_
# grid_search.best_estimator_
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestClassifier(max_features=6, max_leaf_nodes=60, n_estimators=30,

                       random_state=1)

forest_model.fit(train_features, train_targets)

random_forest_pred = forest_model.predict(test_features)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import balanced_accuracy_score,make_scorer

balance_accuracy = make_scorer(balanced_accuracy_score)



scores = cross_val_score(forest_model, train_features, train_targets, scoring=balance_accuracy,cv=10)
display_scores(scores)
# random_forest_pred

predictions = pd.DataFrame(random_forest_pred,columns=['Survived'])
# predictions.head()

final_submission = pd.concat([test_df['PassengerId'],predictions['Survived']],axis=1)

final_submission.to_csv('gender_submission_forest_new3.csv',index=False)
# from sklearn.model_selection import GridSearchCV

# from xgboost import XGBClassifier



# param_grid = [{'learning_rate':[0.5,0.2,0.3,0.2,0.1,0.01],'num_boost_round': [100,200,300,400,500,100,2000],'n_jobs':[1,2,3,4],

# 'min_child_weight':[10,12,13,14,15],'colsample_bytree':[0.2,0.5,0.8,1],'max_depth':[10,12,16,18],'eta':[0.3,0.5]}]



# # 'learning_rate':[0.5,0.2,0.3,0.2,0.1,0.01],'num_boost_round': [100,200,300,400,500,100,2000],'n_jobs':[1,2,3,4],

# # 'min_child_weight':[10,12,13,14,15],colsample_bytree:[0.2,0.5,0.8,1],max_depth:[10,12,16,18],eta:0.3

            



# XGBoost_model = XGBClassifier(random_state=1)



# grid_search = GridSearchCV(XGBoost_model,param_grid,cv=5,

#                           scoring=balance_accuracy,

#                           return_train_score=True)



# grid_search.fit(train_features,train_targets)
import xgboost as xgb



xgb_clf = xgb.XGBClassifier(colsample_bynode=1, colsample_bytree=0.8, eta=0.5, 

              gpu_id=-1, importance_type='gain', interaction_constraints='',

              learning_rate=0.05, max_delta_step=0, max_depth=16,

              min_child_weight=10, n_jobs=1,n_estimators=500,

              num_parallel_tree=1, objective='binary:logistic', random_state=1,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

              tree_method='exact', validate_parameters=1, verbosity=None)





xgb_clf = xgb_clf.fit(train_features,train_targets)

prediction_xgb=xgb_clf.predict(test_features)

#now we pass the testing data to the trained algorithm

predictions = pd.DataFrame(prediction_xgb,columns=['Survived'])



# predictions.head()

final_submission = pd.concat([test_df['PassengerId'],predictions['Survived']],axis=1)

final_submission.to_csv('gender_submission_new_xgb.csv',index=False)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import balanced_accuracy_score,make_scorer

balance_accuracy = make_scorer(balanced_accuracy_score)



scores = cross_val_score(xgb_clf, train_features, train_targets, scoring=balance_accuracy,cv=10)
display_scores(scores)
from sklearn.ensemble import VotingClassifier



voting_clf = VotingClassifier(

    estimators = [('xg',xgb_clf),('rf',forest_model),('svc',svc_clf)],

    voting='hard'

)

voting_clf.fit(train_features,train_targets)
scores = cross_val_score(voting_clf, train_features, train_targets, scoring=balance_accuracy,cv=10)
display_scores(scores)
prediction_voting=voting_clf.predict(test_features)

prediction_voting = pd.DataFrame(prediction_voting,columns=['Survived'])

final_submission = pd.concat([test_df['PassengerId'],predictions['Survived']],axis=1)

final_submission.to_csv('gender_submission_voting.csv',index=False)