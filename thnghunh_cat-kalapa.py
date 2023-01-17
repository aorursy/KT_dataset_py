from itertools import combinations

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
import warnings

from catboost import CatBoostClassifier
from unidecode import unidecode
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from itertools import combinations
from datetime import datetime
from contextlib import contextmanager

import warnings 
warnings.filterwarnings('ignore')
lgbm_param = {'boosting_type': 'gbdt', 'colsample_bytree': 0.6602479798930369, 
              'is_unbalance': False, 'learning_rate': 0.00746275526696824, 
              'max_depth': 15, 'metric': 'auc', 'min_child_samples': 25, 
              'num_leaves': 60, 'objective': 'binary', 'reg_alpha': 0.4693391197064131, 
              'reg_lambda': 0.16175478669541327, 'subsample_for_bin': 60000}


NUM_BOOST_ROUND= 10000

DROP = ['currentLocationLocationId','currentLocationLatitude','currentLocationLongitude','homeTownLocationId'] +\
       [f"Field_{c}" for c in [11, 14, 15, 16, 17, 18, 24,25, 26, 30, 31, 32, 33, 34,35, 37,40,45, 46, 48,49, 52, 56, 57, 68]]
DATE = ["Field_{}".format(i) for i in [5, 6, 7, 8, 9, 11, 15, 25, 32, 33, 34, 35, 40]]
DATETIME = ["Field_{}".format(i) for i in [1, 2, 43, 44]]
DROP += DATE
DROP += DATETIME
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
def gini(y_true, y_score):
    return roc_auc_score(y_true, y_score)*2 - 1

def lgb_gini(y_pred, dataset_true):
    y_true = dataset_true.get_label()
    return 'gini', gini(y_true, y_pred), True
def subtract_date(date1,date2, df):
    df[date1] = pd.to_datetime(df[date1], infer_datetime_format=True)
    df[date2] = pd.to_datetime(df[date2], infer_datetime_format=True)
    df[date1+date2] = (df[date2] - df[date1]).dt.days
    
def process_ngaySinh(s):
    if s != s:
        return np.nan
    try:
        s = int(s)
    except ValueError:
        s = s.split(" ")[0]
        
    return datetime.strptime(str(s)[:6], "%Y%m")

def datetime_normalize(s):
    if s != s:
        return np.nan
    
    s = s.split(".")[0]
    if s[-1] == "Z":
        s = s[:-1]
        
    date, time = s.split("T")
    datetime_obj = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    return datetime_obj

def date_normalize(s):
    if s != s:
        return np.nan
    
    try:
        datetime_obj = datetime.strptime(s, "%m/%d/%Y")
    except:
        datetime_obj = datetime.strptime(s, "%Y-%m-%d")
        
    return datetime_obj

def process_datetime_cols(df):
    cat_cols = []
    for col in DATETIME:
        df[col] = df[col].apply(datetime_normalize)
        
    for col in DATE:
        if col == "Field_34":
            continue
        df[col] = df[col].apply(date_normalize)

    df["Field_34"] = df["Field_34"].apply(process_ngaySinh)
    df["ngaySinh"] = df["ngaySinh"].apply(process_ngaySinh)
    
    cat_cols += DATE + DATETIME
    for col in DATE + DATETIME:
        df[col] = df[col].dt.strftime('%d-%m-%Y')
    
    for cat in ['F', 'E', 'C', 'G', 'A']:
        subtract_date(f'{cat}_startDate', f'{cat}_endDate', df)
    print(df.shape) 
    return df

def str_normalize(s):
    s = str(s).strip().lower()
    s = re.sub(' +', " ", s)
    return s
def create_newcate(df):
    cate_columns = ['brief','Field_79','Field_80','data.basic_info.locale']
    for i in cate_columns:
        df['Field_82' + '_' + i] = df['Field_82'].astype(str) + '_' + df[i].astype(str)
    for i in cate_columns[1:]:
        df['brief' + '_' + i] = df['brief'].astype(str) + '_' + df[i].astype(str)
    return df
def process_ordinal(df):        
    df["subscriberCount"].replace(0, np.nan, inplace=True)
    df["friendCount"].replace(0, np.nan, inplace=True)
    
    df["Field_13"] = df["Field_13"].apply(lambda x: 1 if x == x else 0)
    df["Field_38"] = df["Field_38"].map({0: 0.0, 1: 1.0, "DN": np.nan, "TN": np.nan, "GD": np.nan})
    df["Field_62"] = df["Field_62"].map({"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "Ngoài quốc doanh Quận 7": np.nan})
    df["Field_47"] = df["Field_47"].map({"Zezo": 0, "One": 1, "Two": 2, "Three": 3, "Four": 4})
    
    df["Field_27"] = df["Field_27"].replace({0.0: np.nan})
    df["Field_28"] = df["Field_28"].replace({0.0: np.nan})
        
    for col in df.columns:
        if df[col].dtype.name == "object":
            df[col] = df[col].apply(str_normalize).astype("category")       
            df[col] = df[col].fillna(np.nan)
        
            le = LabelEncoder()
            le.fit(list(df[col]))
            df[col] = le.transform(df[col])
            
            if df[col].nunique() > 7:
                vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
                df[col] = df[col].map(vc)
            
    return df
def transform(df):
    df = process_datetime_cols(df)
    df = create_newcate(df)
    df = process_ordinal(df)
    return df.drop(DROP, 1)
def kfold(train_fe,y_label,test_fe):
    seeds = np.random.randint(0, 10000, 1)
    preds = 0    
    feature_important = None
    avg_train_gini = 0
    avg_val_gini = 0

    for s in seeds:
        skf = StratifiedKFold(n_splits=5, random_state = 6484, shuffle=True)        
        lgbm_param['random_state'] = 6484    
        seed_train_gini = 0
        seed_val_gini = 0
        for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_label)), y_label)):                
            X_train, X_val = train_fe.iloc[train_idx].drop(["id"], 1), train_fe.iloc[val_idx].drop(["id"], 1)                
            y_train, y_val = y_label[train_idx], y_label[val_idx]

            evals_result = {} 
            
            model = CatBoostClassifier(eval_metric='AUC', 
                             use_best_model=True,
                             iterations=1000, 
                             learning_rate=0.1, 
                             random_seed=42).fit(X_train, y_train, 
                                                 eval_set=(X_val, y_val), verbose=500)

            val_pred = model.predict_proba(X_val)[:,1]
            seed_val_gini += (2 * roc_auc_score(y_val, val_pred) - 1) / skf.n_splits

            avg_val_gini += (2 * roc_auc_score(y_val, val_pred) - 1) / (len(seeds) * skf.n_splits)       

            pred = model.predict_proba(test_fe.drop(["id"], 1))[:,1]
            preds += pred / (skf.n_splits * len(seeds))

            print("Fold {}: {}".format(i, avg_val_gini))
        print("Seed {}: {}".format(s, seed_val_gini))

    print("-" * 30)
    print("Avg train gini: {}".format(avg_train_gini))
    print("Avg valid gini: {}".format(avg_val_gini))
    print("=" * 30)
    return preds


def main():
    df_train = pd.read_csv('../input/klps-creditscring-challenge-for-students/train.csv')
    df_test = pd.read_csv('../input/klps-creditscring-challenge-for-students/test.csv')
    df_all = df_train.drop(['label'], 1).append(df_test)
    
    
    with timer("Preprocess"):
        df_all_fe = transform(df_all.copy())
        print("Bureau df shape:", df_all_fe.shape)
        df_all_fe['Age'] = df_all_fe.ngaySinh.apply(lambda x: 2020 - x.year)
        df_all_fe = df_all_fe.drop('ngaySinh', axis = 1)
        cols_select = [x for x in df_all_fe.columns if x not in DATE + DATETIME  + [f'{cat}_endDate' for cat in ['F', 'E', 'C', 'G', 'A']] + [f'{cat}_startDate' for cat in ['F', 'E', 'C', 'G', 'A']]]
        df_fe = df_all_fe[cols_select]
        df_fe.replace([np.inf, -np.inf], -99999, inplace=True)

        for col in df_fe.columns:
            if df_fe[col].dtype.name == "category":
                if df_fe[col].isnull().sum() > 0:
                    df_fe[col] = df_fe[col].cat.add_categories(f'missing_{col}')
                    df_fe[col].fillna(f'missing_{col}', inplace=True)
            else:
                df_fe[col].fillna(-99999, inplace=True)

        y_label = df_train["label"]
        train_fe = df_fe[df_fe["id"] < df_train.shape[0]]
        test_fe = df_fe[df_fe["id"] >= df_train.shape[0]]

        print(train_fe.shape)
        print(test_fe.shape)
    with timer("Kfold"):
        preds = kfold(train_fe,y_label,test_fe)
        df_test["label"] = preds
        df_test[['id', 'label']].to_csv('submission.csv', index=False)
        
if __name__ == "__main__":
    submission_file_name = "submission.csv"
    
    with timer("Full model run"):
        main()
0.487307