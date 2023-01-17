import pandas as pd

import numpy as np

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import lightgbm as lgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 
path_train = '../input/health-insurance-cross-sell-prediction/train.csv'



train = pd.read_csv(path_train, sep=',', index_col=['id'])



#encoding categorical features

va = {'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0}

gen = {'Male' : 0, 'Female' : 1}

vg = {'Yes' : 1, 'No' : 0}

train['Vehicle_Age'] = train['Vehicle_Age'].map(va)

train['Gender'] = train['Gender'].map(gen)

train['Vehicle_Damage'] = train['Vehicle_Damage'].map(vg)



train.tail()
plt.figure(figsize=(13, 7))



sns.distplot(a=train.Annual_Premium, kde=False)
sns.distplot(a=train.Vintage, kde=False)
train.shape
#Removing outliers

train = train.query('Annual_Premium <= 100000')

train.shape
num_feat = ['Age', 'Vintage', 'Annual_Premium']



cat_feat = [

    'Gender', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',

    'Driving_License', 'Policy_Sales_Channel', 'Region_Code'

]
#Just scaling num_cols

scl = StandardScaler()



num_scl = pd.DataFrame(scl.fit_transform(train[num_feat]))

num_scl.index = train[num_feat].index

num_scl.columns = train[num_feat].columns

X_ = pd.concat([num_scl, train[cat_feat]], axis=1)

X_.head()
y = train.Response

X_.shape, y.shape
grid_param = {

    'num_leaves': [60, 70, 80],

    'min_child_weight': [0.1, 0.5, 1, 1.5, 2],

    'feature_fraction': [0.1, 0.5, 1, 1.5, 2],

    'bagging_fraction': [0.1, 0.5, 1, 1.5, 2],

    'max_depth': [6, 7, 8],

    'learning_rate': [0.9, 0.1, 0.12, 0.15],

    'reg_alpha': [0.5, 0.9, 1.2, 1.8],

    'reg_lambda': [0.5, 0.9, 1.2, 1.8,],

    'num_iterations': [90, 100, 110]

}



model = lgb.LGBMClassifier(random_state=22)



grid_fold = KFold(n_splits=5, shuffle=True, random_state=12)



grid_search = RandomizedSearchCV(model,

                                 param_distributions=grid_param,

                                 scoring='roc_auc',

                                 cv=grid_fold,

                                 n_jobs=-1,

                                 verbose=1,

                                 random_state=112)



grid_result = grid_search.fit(X_, y)

print(grid_result.best_score_, grid_result.best_params_)
params = {

    'reg_lambda': 1.8,

    'reg_alpha': 0.9,

    'num_leaves': 80,

    'min_child_weight': 1,

    'max_depth': 6,

    'learning_rate': 0.12,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'objective': 'binary',

    "boosting_type": "gbdt",

    "bagging_seed": 23,

    "metric": 'auc',

    "verbosity": -1

}
#split to folds and training lightgbm



n_folds = 5

fold = KFold()

splits = fold.split(X_, y)

columns = X_.columns

oof = np.zeros(X_.shape[0])

score = 0

y_oof = np.zeros(X_.shape[0])

feature_importances = pd.DataFrame()

feature_importances['feature'] = columns
for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = X_[columns].iloc[train_index], X_[columns].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label = y_train)

    dvalid = lgb.Dataset(X_valid, label = y_valid)

    

    clf = lgb.train(params, dtrain, valid_sets=[dtrain, dvalid], 

                    verbose_eval=100)

    

    

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    score += roc_auc_score(y_valid, y_pred_valid) / n_folds

    

print(f"\nMean AUC = {score}")

print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")
feature_importances['average'] = feature_importances[[

    f'fold_{fold_n + 1}' for fold_n in range(fold.n_splits)

]].mean(axis=1)



plt.figure(figsize=(14, 7))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(9), x='average', y='feature');

plt.title('TOP feature importance over {} folds average'.format(fold.n_splits))