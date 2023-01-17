import gc
import numpy as np
import pandas as pd
import lightgbm as lgbm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
%matplotlib inline
train = pd.read_csv('../input/train.csv')
train['target'] = np.log1p(train['target'])
test = pd.read_csv('../input/test.csv')
topfeatsbygp = ['58e2e02e6', 'f190486d6', '6eef030c1', '9fd594eec', 'b30e932ba',
                '58232a6fb', 'f74e8f13d', '1c71183bb', 'f514fdb2e', '70feb1494',
                'ced6a7e91', 'fb49e4212', '26fc93eb7', 'db3839ab0', 'fc99f9426',
                'bb1113dbb', '15ace8c9f', '1702b5bf0', '20aa07010', '2288333b4',
                '64e483341', 'cfc1ce276', 'c47340d97', '324921c7b', '1d9078f84',
                '491b9ee45']
testcolumns = list(['ID']) + list(topfeatsbygp)
traincolumns = list(topfeatsbygp)+list(['target']) 
test = test[testcolumns]
train = train[traincolumns]
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    "learning_rate": 0.1,
    "num_leaves": 180,
    "feature_fraction": 0.50,
    "bagging_fraction": 0.50,
    'bagging_freq': 4,
    "max_depth": -1,
    "reg_alpha": 0.3,
    "reg_lambda": 0.1,
    "min_child_weight":10,
    'zero_as_missing':True
}
lgtrain = lgbm.Dataset(train[topfeatsbygp],train.target)
lgb_cv = lgbm.cv(
    params = lgbm_params,
    train_set = lgtrain,
    num_boost_round=2000,
    stratified=False,
    nfold = 5,
    verbose_eval=50,
    seed = 23,
    early_stopping_rounds=50)

optimal_rounds = np.argmin(lgb_cv['rmse-mean'])
best_cv_score = min(lgb_cv['rmse-mean'])
del lgtrain
gc.collect()
print(optimal_rounds)
print(best_cv_score)
lgtrain = lgbm.Dataset(train[topfeatsbygp],train.target)
clf = lgbm.train(lgbm_params,
                 lgtrain,
                 num_boost_round = optimal_rounds + 1,
                 verbose_eval=50)
fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = topfeatsbygp
fold_importance_df["importance"] = clf.feature_importance()
plt.figure(figsize=(8,10))
sns.barplot(x="importance", y="feature", 
            data=fold_importance_df.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features')
plt.tight_layout()
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):
    print(n_fold)
    trn_x, trn_y = train[topfeatsbygp].iloc[trn_idx], train.iloc[trn_idx]['target']
    val_x, val_y = train[topfeatsbygp].iloc[val_idx], train.iloc[val_idx]['target']
    
       
    clf = lgbm.train(lgbm_params,
                     lgbm.Dataset(trn_x,trn_y),
                     num_boost_round = optimal_rounds + 1,
                     verbose_eval=200)
    
    oof_preds[val_idx] = clf.predict(val_x, num_iteration=optimal_rounds + 1)
    sub_preds += clf.predict(test[topfeatsbygp], num_iteration=optimal_rounds + 1) / folds.n_splits
    
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
sub = pd.DataFrame({'ID':test.ID, 'target':np.expm1(sub_preds)})
sub.to_csv('featuresbygp.csv',index=False)
