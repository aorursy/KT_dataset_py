%%time

import os

import numpy as np 

import pandas as pd



from sklearn import ensemble

import xgboost as xgb



import matplotlib.pyplot as plt



from sklearn import preprocessing



from sklearn.model_selection import RandomizedSearchCV



from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import Pipeline



from sklearn.metrics import roc_auc_score



import warnings

warnings.filterwarnings('ignore')
%%time

#load data

test_data = pd.read_csv('../input/test.csv')

train_data = pd.read_csv('../input/train.csv')
train_data = train_data.sample(n=5000)

train_data.shape
# How data looks

train_data.head()
%time

# Any nulls?

(train_data.isnull().sum()).any() > 0
%time

# Drop columns what contains 0 only

dropable_cols = []

for i in train_data.columns:

    if (train_data[i] == 0).all():

        dropable_cols.append(i)

        

train_data.drop(dropable_cols, axis=1, inplace=True)

test_data.drop(dropable_cols, axis=1, inplace=True)

print("Data shape after droping rows: ")

print("Train data shape: ",train_data.shape, "Test data shape: ", test_data.shape)

%%time

# Removing dublicated columns

columns_to_drop = []

columns = train_data.columns

for i in range(len(columns) - 1):

    # assign and check column equality 

    column_to_check = train_data[columns[i]]

    for c in range(i+1, len(columns)):

        if np.array_equal(column_to_check, train_data[columns[c]].values):

            columns_to_drop.append(columns[c])

train_data.drop(columns_to_drop, axis=1, inplace=True)

test_data.drop(columns_to_drop, axis=1, inplace=True)

print("Data after cleaning")

print("Train data shape: ",train_data.shape, "Test data shape: ", test_data.shape)
### #############################33333###

tr = train_data.copy()

tr.corr() > 0.95

# What is in target column

train_data.TARGET.value_counts()

#if training is done only on negative rewiews(1), and if doest apply to neg model is true maybe it is better for model?
%%time

#Spliting to Train, test, valid

df_train = train_data[:3000] 

df_test = train_data[3000:4000]

# Model will not see valid data set 

df_valid = train_data[4000:]

print(df_train.shape, df_test.shape, df_valid.shape)
%time

# Training data

X_train = df_train.drop(['ID', 'TARGET'], axis=1)

y_train = df_train.TARGET

# Test data

X_test = df_test.drop(['ID', 'TARGET'], axis=1)

y_test = df_test.TARGET

# Validation data

X_valid = df_valid.drop(['ID', 'TARGET'], axis=1)

y_valid = df_valid.TARGET

# submision data

data_for_sub = test_data.drop(['ID'], axis=1)
xgb = xgb.XGBRFRegressor()

tree = ensemble.RandomForestRegressor()

ada = ensemble.AdaBoostRegressor()

grad = ensemble.GradientBoostingRegressor()
import scipy as sp



def get_scores_and_params(pipeline, params):

   



    search = RandomizedSearchCV(pipeline, 

    params, cv=3, n_iter=5, scoring="roc_auc",

                                    n_jobs=-1,

                                    verbose=2)

    search.fit(X_train, y_train)

    return search.best_score_, search.best_params_
pipelines = [Pipeline([('xgb', xgb)]), 

             Pipeline([('tree', tree)]), 

             Pipeline([('ada', ada)]),

             Pipeline([('grad', grad)])]

getd = [

{'xgb__max_depth': sp.stats.randint(1, 11),

'xgb__n_estimators': [100, 200, 500, 1000],

'xgb__colsample_bytree': [0.5,0.6,0.7,0.8]}

        

,{'tree__n_estimators': [100, 200, 500, 1000],

  'tree__min_samples_split':[2, 4, 8, 10],

  'tree__min_samples_leaf': [1, 2, 3, 4]}



,{'ada__learning_rate': [0.3, 0.4,0.5,0.7,1],

  'ada__n_estimators': [10, 50, 100, 500]}

    

,{'grad__learning_rate': [0.1,0.2,0.5,1],

  'grad__max_depth': [3,5,7],

  'grad__n_estimators': [1, 2, 3, 4]}

]

warnings.filterwarnings('ignore')



for i in range(len(pipelines)):

    print(get_scores_and_params(pipelines[i], getd[i]))

    
#%time

## # Feature selection

# Classifier runs faster

clf = ensemble.AdaBoostRegressor(n_estimators=500, learning_rate=0.5)

selector = clf.fit(X_train, y_train)



# plot most important features

feat_imp = pd.Series(clf.feature_importances_, index = X_train.columns.values).sort_values(ascending=False)

feat_imp[:40].plot(kind='bar', title='Feature Importances according to AdaBoostRegressor', figsize=(12, 8))

plt.ylabel('Feature Importance Score')

plt.subplots_adjust(bottom=0.3)

plt.show()
# features to fit model

features = feat_imp[:40].index

print(features)
# mount new selected features

X_train = X_train[features]

X_test = X_test[features]

X_valid = X_valid[features]
X_train

df = df[df.line_race != 0]
# plot first 5 most important features

for i in features[0:5]:

    x = train_data[train_data[i] != 0]

    #train_data[i].value_counts().mean()

    x = train_data[i].value_counts().head().index

    y = train_data[i].value_counts().head()

    

    plt.figure()

    plt.scatter(y, x)



    plt.xlabel(i)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)

val_pred = clf.predict(X_valid)

roc_auc_score(y_test, preds), roc_auc_score(y_valid, val_pred)
pros = clf.predict(data_for_sub[features])
# submission

sub = pd.DataFrame()

sub['ID'] = test_data['ID']

sub['target'] = pros

sub.to_csv('submission.csv',index=False)

# distribution of values

test = pd.read_csv('submission.csv')

test.head()