import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

path = "/kaggle/input/lish-moa/"
train_df = pd.read_csv(path + "train_features.csv", index_col = "sig_id")
test_df = pd.read_csv(path + "test_features.csv", index_col = "sig_id")
subm_df = pd.read_csv(path + "sample_submission.csv")
tr_scored = pd.read_csv(path + "train_targets_scored.csv", index_col = "sig_id")
tr_nonscored = pd.read_csv(path + "train_targets_nonscored.csv", index_col = "sig_id")
def make_numeric(df):
    df["cp_type"] = df["cp_type"].replace("trt_cp",0)
    df["cp_type"] = df["cp_type"].replace("ctl_vehicle",1)
    df["cp_dose"] = df["cp_dose"].replace("D1",0)
    df["cp_dose"] = df["cp_dose"].replace("D2",1)
    return df
train_df = make_numeric(train_df)
test_df = make_numeric(test_df)
num_fold = 5
def get_lgbm_model(X_train, y_train, X_val, y_val, fold, columns):
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = {
        "metric":"binary_logloss",
        "learning_rate":0.01
    }  
    
    model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_train, y_train), (X_val, y_val)], 
        verbose=10, 
        early_stopping_rounds=10
    )
    
    return model

def get_lgbm_pred(X, y, test):
    print("get_lgbm_pred ")

    pred = []
    pred_val = np.zeros((len(X)))
            
    #X_train, X_val, y_train, y_val = train_test_split(X_scaled, y.fillna(0).values, test_size=0.2, shuffle=True, random_state=42)
    kf = KFold(n_splits=num_fold, random_state=None, shuffle=False)
    fold = 0
    score = 0
    for train_index, test_index in kf.split(X, y):
        fold += 1
        print("fold ", fold)
    
        X_train = X.iloc[train_index, :]
        X_val = X.iloc[test_index, :]
        y_train = y[train_index]
        y_val = y[test_index]
        
        model = get_lgbm_model(X_train.values, y_train, X_val.values, y_val, fold, X_train.columns)


        if fold ==1:
            pred.append(model.predict(test))
        else:
            pred += model.predict(test)

        
        pred_val[test_index] = model.predict(X_val)            
        score = score + np.sqrt(mean_squared_error(y_val, pred_val[test_index]))
        print("score ", str(score/fold))
                
    return pred[0]/num_fold, pred_val
n=0
for col in subm_df.columns[1:]:
    n += 1
    print(n, col)
    subm_df[col], pred_tr = get_lgbm_pred(train_df, tr_scored[col], test_df)
subm_df.to_csv("submission.csv", index=False)