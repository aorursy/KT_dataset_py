import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
train  = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
id_test = test['id'].values
train_target = train['target'].values

#train_test = train.drop(['target', 'id'], axis=0)
train = train.drop(['target', 'id'], axis = 1)
test = test.drop(['id'], axis = 1)
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
print(col_to_drop)
train = train.drop(col_to_drop, axis=1)
test = test.drop(col_to_drop, axis=1)
train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)
cat_features = [a for a in train.columns if a.endswith('cat')]
print(cat_features)
for column in cat_features:
    temp = pd.get_dummies(pd.Series(train[column]))
    train = pd.concat([train, temp], axis=1)
    train = train.drop([column], axis=1)
for column in cat_features:
    temp = pd.get_dummies(pd.Series(test[column]))
    test = pd.concat([test, temp], axis = 1)
    test = test.drop([column], axis=1)
print(train.shape)
print(test.shape)
class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models
        
    def fit_predict(self, X, y, T):
        increase = True
        print(X.shape)
        if increase:
            pos = pd.Series(y == 1)
            y = pd.Series(y)
            X = pd.concat([X, X.loc[pos]], axis = 0)
            y = pd.concat([y, y.loc[pos]], axis = 0)
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X = X.iloc[idx]
            y = y.iloc[idx]
        print(X.shape)
        print(T.shape)
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        
        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle = True, random_state=17).split(X, y))
        
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        
        
        
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], self.n_splits))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                

                print("fit %s fold %d " %(str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_holdout)[:, 1]
                
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:, 1]
            S_test[:, i] = S_test_i.mean(axis=1)
        result = cross_val_score(self.stacker, S_train, y, cv=3)
        print("Stacker score : %.5f "%(result.mean()))
        self.stacker.fit(S_train, y)
        
        res = self.stacker.predict_proba(S_test)[:, 1]
        return res
        
lgb_params = {
    'learning_rate' : 0.02,
    'n_estimators' : 650,
    'max_bin' : 10,
    'subsample' : 0.8,
    'subsample_freq' : 10,
    'colsample_bytree' : 0.8,
    'min_child_samples' : 500,
    'seed' : 99
}

lgb_params2 = {
    'n_estimators' : 1090,
    'learning_rate' : 0.02,
    'colsample_bytree' : 0.3,
    'subsample' : 0.7,
    'subsample_freq' : 2,
    'num_leaves' : 16,
    'seed' : 99
}

lgb_params3 = {
    'n_estimators' : 110,
    'max_depth' : 4,
    'learning_rate' : 0.02,
    'seed' : 99
}

lgb_model = LGBMClassifier(**lgb_params)
lgb_model2 = LGBMClassifier(**lgb_params2)
lgb_model3 = LGBMClassifier(**lgb_params3)

log_model = LogisticRegression()
stack = Ensemble(n_splits = 3, stacker = log_model, base_models = (lgb_model, lgb_model2))

y_pred = stack.fit_predict(train, train_target, test)
y_pred
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.head(20)
sub.to_csv('submission.csv', index=False)