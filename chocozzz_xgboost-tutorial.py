import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

all = pd.concat([ train, test ],sort=False)
all.head()
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
all["Embarked"] = le.fit_transform(all["Embarked"].fillna('0'))
all["Sex"] = le.fit_transform(all["Sex"].fillna('3'))
all.head()
# split the data back into train and test
df_train = all.loc[all['Survived'].isin([np.nan]) == False]
df_test  = all.loc[all['Survived'].isin([np.nan]) == True]
print(df_train.shape)
print(df_test.shape)
df_train.head()
df_test.head()
'''
# XGBoost model + parameter tuning with GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

feature_names = ['Sex','Embarked','Pclass','Survived','Age','SibSp','Parch','Fare']

xgb = XGBRegressor()
params={
    'max_depth': [2,3,4,5], 
    'subsample': [0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree': [0.5,0.6,0.7,0.8],
    'n_estimators': [1000,2000,3000],
    'reg_alpha': [0.01, 0.02, 0.03, 0.04]
}

grs = GridSearchCV(xgb, param_grid=params, cv = 10, n_jobs=4, verbose=2)
grs.fit(np.array(df_train[feature_names]), np.array(df_train['Survived']))

print("Best parameters " + str(grs.best_params_))
gpd = pd.DataFrame(grs.cv_results_)
print("Estimated accuracy of this model for unseen data: {0:1.4f}".format(gpd['mean_test_score'][grs.best_index_]))
# TODO: why is this so bad?
'''
train_y = df_train['Survived']; train_x = df_train.drop('Survived',axis=1)
excluded_feats = ['PassengerId','Ticket','Cabin','Name']
features = [f_ for f_ in train_x.columns if f_ not in excluded_feats]
features
from sklearn.model_selection import KFold
folds = KFold(n_splits=4, shuffle=True, random_state=546789)
oof_preds = np.zeros(train_y.shape[0])
sub_preds = np.zeros(df_test.shape[0])
import gc
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_x)):

    trn_x, trn_y = train_x[features].iloc[trn_idx], train_y.iloc[trn_idx]

    val_x, val_y = train_x[features].iloc[val_idx], train_y.iloc[val_idx]

    

    clf = XGBClassifier(

        objective = 'binary:logistic', 

        booster = "gbtree",

        eval_metric = 'auc', 

        nthread = 4,

        eta = 0.05,

        gamma = 0,

        max_depth = 2, 

        subsample = 0.6, 

        colsample_bytree = 0.8, 

        colsample_bylevel = 0.675,

        min_child_weight = 22,

        alpha = 0,

        random_state = 42, 

        nrounds = 2000,
        
        n_estimators=3000

    )



    clf.fit(trn_x, trn_y, eval_set= [(trn_x, trn_y), (val_x, val_y)], verbose=10, early_stopping_rounds=100)

    

    oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]

    sub_preds += clf.predict_proba(df_test[features])[:, 1] / folds.n_splits

    

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))

    del clf, trn_x, trn_y, val_x, val_y

    gc.collect()

    

print('Full AUC score %.6f' % roc_auc_score(train_y, oof_preds))   



test['Survived'] = sub_preds



test[['PassengerId', 'Survived']].to_csv('xgb_submission_esi.csv', index=False, float_format='%.8f')