import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
import warnings

import lightgbm as lgb
from unidecode import unidecode
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from itertools import combinations
from datetime import datetime

warnings.filterwarnings('ignore')
pd.options.display.max_columns = 50
plt.style.use('ggplot')

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/kalapas/train.csv')
df_test = pd.read_csv('/kaggle/input/kalapas/test.csv')
df_all = df_train.drop(['label'], 1).append(df_test)
df_all.info()
# Process date/datetime fields
DATE = ["Field_{}".format(i) for i in [5, 6, 7, 8, 9, 11, 15, 25, 32, 33, 34, 35, 40]]
DATETIME = ["Field_{}".format(i) for i in [1, 2, 43, 44]]

df_all[DATE + DATETIME + ["Field_34", "ngaySinh"]].sample(10)
def correct_34_ngaysinh(s):
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

    df["Field_34"] = df["Field_34"].apply(correct_34_ngaysinh)
    df["ngaySinh"] = df["ngaySinh"].apply(correct_34_ngaysinh)

    cat_cols += DATE + DATETIME
    for col in DATE + DATETIME:
        df[col] = df[col].dt.strftime('%m-%Y')
    
    for cat in ['F', 'E', 'C', 'G', 'A']:
        df[f'{cat}_startDate'] = pd.to_datetime(df[f"{cat}_startDate"], infer_datetime_format=True)
        df[f'{cat}_endDate'] = pd.to_datetime(df[f"{cat}_endDate"], infer_datetime_format=True)
        
        df[f'{cat}_startDate'] = df[f'{cat}_startDate'].dt.strftime('%m-%Y')
        df[f'{cat}_endDate'] = df[f'{cat}_endDate'].dt.strftime('%m-%Y')
        
        cat_cols.append(f'{cat}_startDate')
        cat_cols.append(f'{cat}_endDate')
    
    for col in cat_cols:
        df[col] = df[col].astype("category")
        
    return df
def str_normalize(s):
    s = str(s).strip().lower()
    s = re.sub(' +', " ", s)
    return s

def process_location(df):
    for col in ["currentLocationLocationId", "homeTownLocationId", "currentLocationLatitude", "currentLocationLongitude", 
                   "homeTownLatitude", "homeTownLongitude"]:
        df[col].replace(0, np.nan, inplace=True)

    df["currentLocationLocationId"] = df["currentLocationLocationId"].apply(str_normalize).astype("category")
    df["homeTownLocationId"] = df["homeTownLocationId"].apply(str_normalize).astype("category")

    return df
def job_category(x):
    if type(x) == str:
        if "công nhân" in x or "cnv" in x or "cn" in x or "may công nghiệp" in x or "lao động" in x\
        or "thợ" in x or "coõng nhaõn trửùc tieỏp maựy may coõng nghieọp" in x or "c.n" in x or "lđ" in x:
            return "CN"
        elif "giáo viên" in x or "gv" in x or "gíao viên" in x:
            return "GV"
        elif "nhân viên" in x or "kế toán" in x or "cán bộ" in x or "nv" in x or "cb" in x or "nhõn viờn" in x:
            return "NV"
        elif "tài xế" in x or "lái" in x or "tài xê" in x:
            return "TX"
        elif "quản lý" in x or "phó phòng" in x or "hiệu phó" in x:
            return "QL"
        elif "undefined" in x:
            return "missing"
        elif "giám đốc" in x or "hiệu trưởng" in x:
            return "GĐ"
        elif "phục vụ" in x:
            return "PV"
        elif "chuyên viên" in x:
            return  "CV"
        elif "bác sĩ" in x or "dược sĩ" in x or "y sĩ" in x or "y sỹ" in x:
            return "BS"
        elif "y tá" in x:
            return "YT"
        elif "hộ sinh" in x:
            return "HS"
        elif "chủ tịch" in x:
            return "CT"
        elif "bếp" in x:
            return "ĐB"
        elif "sư" in x:
            return "KS"
        elif "dưỡng" in x:
            return "ĐD"
        elif "kỹ thuật" in x or "kĩ thuật" in x:
            return "KTV"
        elif "diễn viên" in x:
            return "DV"
        else:
            return "missing"
    else:
        return x    
    
def process_diaChi_maCv(df):
    df["maCv"] = df["maCv"].apply(str_normalize).apply(job_category).astype("category")
    return df
def combine_gender(s):
    x, y = s
    
    if x != x and y != y:
        return "nan"
    
    if x != x:
        return y.lower()
    
    return x.lower()

def process_gender(df):
    df["gender"] = df[["gioiTinh", "info_social_sex"]].apply(combine_gender, axis=1).astype("category")
    return df
def process_misc(df):        
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
            
    return df
        
# drop some fields we do not need (homeTown is optionally)
DROP = ["gioiTinh", "info_social_sex", "ngaySinh", "namSinh"] + \
        [f"Field_{c}" for c in [14, 16, 17, 24, 26, 30, 31, 37, 52, 57]]

def transform(df):
    df = process_datetime_cols(df)
    df = process_gender(df)
    df = process_location(df)
    df = process_diaChi_maCv(df)
    df = process_misc(df)
    return df.drop(DROP, 1)
df_all_fe = transform(df_all.copy())
df_all_fe.info()
df_fe = df_all_fe.copy()
df_fe.replace([np.inf, -np.inf], 999, inplace=True)

for col in df_fe.columns:
    if df_fe[col].dtype.name == "category":
        if df_fe[col].isnull().sum() > 0:
            df_fe[col] = df_fe[col].cat.add_categories(f'missing_{col}')
            df_fe[col].fillna(f'missing_{col}', inplace=True)
    else:
        df_fe[col].fillna(-1, inplace=True)

y_label = df_train["label"]
train_fe = df_fe[df_fe["id"] < df_train.shape[0]]
test_fe = df_fe[df_fe["id"] >= df_train.shape[0]]

print(train_fe.shape)
print(test_fe.shape)
def gini(y_true, y_score):
    return roc_auc_score(y_true, y_score)*2 - 1

def lgb_gini(y_pred, dataset_true):
    y_true = dataset_true.get_label()
    return 'gini', gini(y_true, y_pred), True

NUM_BOOST_ROUND = 1000

lgbm_param = {'objective':'binary',
              'boosting_type': 'gbdt',
              'metric' : 'auc',
              'learning_rate': 0.015,
              "bagging_freq": 1,
              "bagging_fraction" : 0.25,
              'tree_learner': 'serial',
              'reg_lambda': 2,
              'reg_alpha': 1,              
              "feature_fraction": 0.15,
              'num_leaves': 16,
              'max_depth': 8,
              'random_state': 16111997,
            }

seeds = np.random.randint(0, 10000, 3)
preds = 0    
feature_important = None
avg_train_gini = 0
avg_val_gini = 0

for s in seeds:
    skf = StratifiedKFold(n_splits=5, random_state=s, shuffle=True)        
    lgbm_param['random_state'] = s    
    seed_train_gini = 0
    seed_val_gini = 0
    for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_label)), y_label)):                
        X_train, X_val = train_fe.iloc[train_idx].drop(["id"], 1), train_fe.iloc[val_idx].drop(["id"], 1)                
        y_train, y_val = y_label[train_idx], y_label[val_idx]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval  = lgb.Dataset(X_val, y_val)

        evals_result = {} 
        model = lgb.train(lgbm_param,
                    lgb_train,
                    num_boost_round=NUM_BOOST_ROUND,  
                    early_stopping_rounds=50,
                    feval=lgb_gini,
                    verbose_eval=False,
                    evals_result=evals_result,
                    valid_sets=[lgb_train, lgb_eval])

        seed_train_gini += model.best_score["training"]["gini"] / skf.n_splits
        seed_val_gini += model.best_score["valid_1"]["gini"] / skf.n_splits

        avg_train_gini += model.best_score["training"]["gini"] / (len(seeds) * skf.n_splits)
        avg_val_gini += model.best_score["valid_1"]["gini"] / (len(seeds) * skf.n_splits)

        if feature_important is None:
            feature_important = model.feature_importance() / (len(seeds) * skf.n_splits)
        else:
            feature_important += model.feature_importance() / (len(seeds) * skf.n_splits)        

        pred = model.predict(test_fe.drop(["id"], 1))
        preds += pred / (skf.n_splits * len(seeds))
        
        print("Fold {}: {}/{}".format(i, model.best_score["training"]["gini"], model.best_score["valid_1"]["gini"]))
    print("Seed {}: {}/{}".format(s, seed_train_gini, seed_val_gini))

print("-" * 30)
print("Avg train gini: {}".format(avg_train_gini))
print("Avg valid gini: {}".format(avg_val_gini))
print("=" * 30)
# Let's display importances of features
features = df_fe.columns.tolist()
features.remove("id")
df_imp = pd.DataFrame(data = {'col' : features , 'imp' : feature_important})
df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)
df_imp.head(50)
# make submission file
df_test["label"] = preds
df_test[['id', 'label']].to_csv('submission.csv', index=False)
