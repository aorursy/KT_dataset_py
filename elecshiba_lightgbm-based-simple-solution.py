import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

from functools import partial
from sklearn.model_selection import StratifiedKFold

import os
print("Files under ../input folder:", os.listdir("../input"))
df_train = pd.read_csv("../input/train_call_history.csv")
df_test  = pd.read_csv("../input/test_call_history.csv")
df_train.info()
columns = df_train.columns
percent_missing = df_train.isnull().sum() * 100 / len(df_train)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df.sort_values("percent_missing", ascending=False, inplace=True)
missing_value_df.head()
thresh = 0.5 * 100
drop_columns = list(missing_value_df[missing_value_df["percent_missing"] > thresh].index)
df_train.drop(drop_columns, axis=1, inplace=True)
df_test.drop(drop_columns, axis=1, inplace=True)
print(f"Dropped {len(drop_columns)} columns.")
df_train.head(2)
drop_columns = [
    "company_code","sogyotoshitsuki","establishment","industry_code1", "industry_code2", "industry_code3",
    "atsukaihin_code_1", "atsukaihin_code_2", "atsukaihin_code_3","eto_meisho"
]
df_train.drop(drop_columns, axis=1, inplace=True)
df_test.drop(drop_columns, axis=1, inplace=True)
print(f"Dropped {len(drop_columns)} columns.")
df_train.select_dtypes(exclude=['int', 'float']).info()
for col in df_train.select_dtypes(exclude=['int', 'float']).columns:
    df_train[col] = df_train[col].astype('category')
    df_test[col] = df_test[col].astype('category')
def lgbm_modeling_cross_validation(params,
                                   full_train, 
                                   y, 
                                   nr_fold=5, 
                                   random_state=1):


    clfs = []
    importances = pd.DataFrame()
    folds = StratifiedKFold(n_splits=nr_fold, 
                            shuffle=True, 
                            random_state=random_state)
    
    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]
        
#         ### Added from here
#         trn_xa, trn_y, val_xa, val_y=smoteAdataset(trn_x.values, trn_y.values, val_x.values, val_y.values)
#         trn_x=pd.DataFrame(data=trn_xa, columns=trn_x.columns)
#         val_x=pd.DataFrame(data=val_xa, columns=val_x.columns)
#         ### to here
    
        clf = LGBMClassifier(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric="auc",
            verbose=100,
            early_stopping_rounds=50
        )
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
    
        imp_df = pd.DataFrame({
                'feature': full_train.columns,
                'gain': clf.feature_importances_,
                'fold': [fold_ + 1] * len(full_train.columns),
                })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    return clfs
y = df_train["result"]
df_train.drop("result", axis=1, inplace=True)
lgb_params = {
        'device': 'cpu', 
        'objective': 'binary', 
        'boosting_type': 'gbdt', 
        'n_jobs': -1, 
        'max_depth': 5, 
        'n_estimators': 1000, 
        'max_cat_to_onehot': 4, 
}

eval_func = partial(lgbm_modeling_cross_validation, 
                    full_train=df_train, 
                    y=y, 
#                     classes=classes, 
#                     class_weights=class_weights, 
                    nr_fold=7, 
                    random_state=7)

clfs = eval_func(lgb_params)
preds_ = None
for clf in clfs:
    if preds_ is None:
        preds_ = clf.predict_proba(df_test) / len(clfs)
    else:
        preds_ += clf.predict_proba(df_test) / len(clfs)
df_submission = pd.DataFrame({'id': df_test['id'], 'result': preds_[:,1]})
df_submission.to_csv('lightgbm_submission.csv', index = False)
