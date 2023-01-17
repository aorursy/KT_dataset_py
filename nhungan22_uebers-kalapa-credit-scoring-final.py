#Cài đặt thêm library feature_engine vào kaggle (Settings để Internet: On)
!pip install feature_engine
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

import lightgbm as lgb
from unidecode import unidecode
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from feature_engine import categorical_encoders as ce

from itertools import combinations
from datetime import datetime
from contextlib import contextmanager
lgbm_param = {'boosting_type': 'gbdt', 
              'colsample_bytree': 0.6602479798930369, 
              'is_unbalance': False, 
              'learning_rate': 0.01,
              'max_depth': 15, 
              'metric': 'auc', 
              'min_child_samples': 25, 
              'num_leaves': 80,
              'objective': 'binary', 
              'reg_alpha': 0.4693391197064131, 
              'reg_lambda': 0.16175478669541327, 
              'subsample_for_bin': 60000}

NUM_BOOST_ROUND= 10000

DROP = ['currentLocationCity', 'currentLocationName', 'homeTownCity', 'homeTownName', 'partner0_B', 'partner0_K', 'partner0_L', 'partner1_B', 'partner1_D', 'partner1_E', 'partner1_F', 'partner1_K', 'partner1_L','partner2_B', 'partner2_G', 'partner2_K', 'partner2_L', 'partner3_B', 'partner3_C', 'partner3_F', 'partner3_G', 'partner3_H', 'partner3_K', 'partner3_L', 'partner4_A', 'partner4_B', 'partner4_C', 'partner4_D', 'partner4_E', 'partner4_F', 'partner4_G', 'partner4_H', 'partner4_K','partner5_B', 'partner5_C', 'partner5_D', 'partner5_E', 'partner5_F', 'partner5_H', 'partner5_K', 'partner5_L'] + [f"Field_{c}" for c in [11, 14, 15, 16, 17, 18, 24, 26, 30, 31, 37, 38, 42, 52, 56, 60, 61, 77, 2, 62, 66, 40, 6, 47, 7, 5, 65, 19, 27, 59, 10, 53, 57, 3, 21, 13]]
DATE = ["Field_{}".format(i) for i in [5, 6, 7, 8, 9, 11, 15, 25, 32, 33, 34, 35, 40]]
DATETIME = ["Field_{}".format(i) for i in [1, 2, 43, 44]]
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true

def lgb_gini(y_pred, dataset_true):
    y_true = dataset_true.get_label()
    return 'gini', Gini(y_true, y_pred), True
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
    
    subtract_date('Field_5','Field_6',df)
    subtrac_List = ['Field_1', 'Field_2', 'Field_43', 'Field_44', 'Field_7','Field_8', 'Field_9']
    subtract_2C = list(combinations(subtrac_List, 2))
    for l in subtract_2C:
        subtract_date(l[0],l[1],df)
      
    for cat in ['F', 'E', 'C', 'G', 'A']:
        subtract_date(f'{cat}_startDate', f'{cat}_endDate', df)
    print(df.shape) 
    return df
  
def str_normalize(s):
    s = str(s).strip().lower()
    s = re.sub(' +', " ", s)
    return s

def process_location(df):
    for col in ["currentLocationLocationId", "homeTownLocationId", "currentLocationLatitude", "currentLocationLongitude", 
                   "homeTownLatitude", "homeTownLongitude"]:
        df[col].replace(0, np.nan, inplace=True)

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
    
def process_maCv(df):
    df["maCv"] = df["maCv"].apply(str_normalize).apply(job_category).astype("category")
    return df
    
def combine_gender(s):
    x, y = s
    return x if x != None else y if y != None else None

def process_gender(df):
    df["gender"] = df[["gioiTinh", "info_social_sex"]].apply(combine_gender, axis=1).astype("category")
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
            
    return df

def transform(df):
    df = process_datetime_cols(df)
    df = process_gender(df)
    df = process_maCv(df)
    df = process_location(df)
    df = process_ordinal(df)
    df["null_sum"] = df.isnull().sum(axis=1)
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

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval  = lgb.Dataset(X_val, y_val)

            evals_result = {}
            model = lgb.train(lgbm_param,
                        lgb_train,
                        num_boost_round=NUM_BOOST_ROUND,  
                        early_stopping_rounds=400,
                        feval=lgb_gini,
                        verbose_eval= 200,
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

    print("-" * 50)
    print("Avg train gini: {}".format(avg_train_gini))
    print("Avg valid gini: {}".format(avg_val_gini))
    print("=" * 50)
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

        y_label = df_train["label"]
        train_fe = df_fe[df_fe["id"] < df_train.shape[0]]
        test_fe = df_fe[df_fe["id"] >= df_train.shape[0]]

        # Label-Encoding
        lbl = LabelEncoder()
        for col in df_fe.columns:
          if df_fe[col].dtype.name == "category":
            train_fe[col] = train_fe[col].astype(str)
            test_fe[col] = test_fe[col].astype(str)
            encoder = ce.CountFrequencyCategoricalEncoder(encoding_method='frequency')
            train_fe[col] = train_fe[col].fillna('None')
            test_fe[col] = test_fe[col].fillna('None')

            # Only take the common values in Train/Test set
            common_vals = list(set(train_fe[col]).intersection(set(test_fe[col])))

            # Replace not-common values with "Missing" or np.nan
            train_fe.loc[~train_fe[col].isin(common_vals), col] = 'Missing'
            test_fe.loc[~test_fe[col].isin(common_vals), col] = 'Missing'

            # Implement LE
            lbl.fit(train_fe[col].tolist() + test_fe[col].tolist())
            train_fe[col] = lbl.transform(train_fe[col])
            test_fe[col] = lbl.transform(test_fe[col])

        print(train_fe.shape)
        print(test_fe.shape)
    with timer("Kfold"):
        preds = kfold(train_fe,y_label,test_fe)
        df_test["label"] = preds
        df_test[['id', 'label']].to_csv('submission1.csv', index=False)
if __name__ == "__main__":
    submission_file_name = "submission1.csv"
    
    with timer("Full model run"):
        main()
lgbm_param = {'boosting_type': 'gbdt', 
              'colsample_bytree': 0.6602479798930369, 
              'is_unbalance': False, 
              'learning_rate': 0.01,
              'max_depth': 15, 
              'metric': 'auc', 
              'min_child_samples': 25, 
              'num_leaves': 80,
              'objective': 'binary', 
              'reg_alpha': 0.4693391197064131, 
              'reg_lambda': 0.16175478669541327, 
              'subsample_for_bin': 60000}

NUM_BOOST_ROUND= 10000

DROP = ["gioiTinh","info_social_sex", 'currentLocationCity', 'currentLocationName', 'homeTownCity', 'homeTownName', 'partner0_B', 'partner0_K', 'partner0_L', 'partner1_B', 'partner1_D', 'partner1_E', 'partner1_F', 'partner1_K', 'partner1_L','partner2_B', 'partner2_G', 'partner2_K', 'partner2_L', 'partner3_B', 'partner3_C', 'partner3_F', 'partner3_G', 'partner3_H', 'partner3_K', 'partner3_L', 'partner4_A', 'partner4_B', 'partner4_C', 'partner4_D', 'partner4_E', 'partner4_F', 'partner4_G', 'partner4_H', 'partner4_K','partner5_B', 'partner5_C', 'partner5_D', 'partner5_E', 'partner5_F', 'partner5_H', 'partner5_K', 'partner5_L'] + [f"Field_{c}" for c in [11, 14, 15, 16, 17, 18, 24, 26, 30, 31, 37, 38, 42, 52, 56, 60, 61]]
DATE = ["Field_{}".format(i) for i in [5, 6, 7, 8, 9, 11, 15, 25, 32, 33, 34, 35, 40]]
DATETIME = ["Field_{}".format(i) for i in [1, 2, 43, 44]]
unicode_cols = ['Field_18', 'maCv', 'diaChi', 'Field_46', 'Field_48', 'Field_49', 'Field_56', 'Field_61', 'homeTownCity', 
                'homeTownName', 'currentLocationCity', 'currentLocationName', 'currentLocationState', 'homeTownState']
object_cols = (unicode_cols + 
               [f'Field_{str(i)}' for i in '4 12 36 38 47 62 45 54 55 65 66 68'.split()] +
               ['data.basic_info.locale', 'currentLocationCountry', 'homeTownCountry', 'brief'])

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true

def lgb_gini(y_pred, dataset_true):
    y_true = dataset_true.get_label()
    return 'gini', Gini(y_true, y_pred), True

# New feature from columns 63, 64
def process_63_64(z):
    x, y = z
    if x != x and y != y:
        return np.nan
    if (x, y) in [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 8.0), (7.0, 5.0), (5.0, 6.0), (9.0, 43.0), (8.0, 9.0)]: return True
    else: return False

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
    
    subtract_date('Field_5','Field_6',df)
    subtrac_List = ['Field_1', 'Field_2', 'Field_43', 'Field_44', 'Field_7','Field_8', 'Field_9']
    subtract_2C = list(combinations(subtrac_List, 2))
    for l in subtract_2C:
        subtract_date(l[0],l[1],df)
      
    for cat in ['F', 'E', 'C', 'G', 'A']:
        subtract_date(f'{cat}_startDate', f'{cat}_endDate', df)
    print(df.shape) 
    return df
  
def str_normalize(s):
    s = str(s).strip().lower()
    s = re.sub(' +', " ", s)
    return s

def process_location(df):
    for col in ["currentLocationLocationId", "homeTownLocationId", "currentLocationLatitude", "currentLocationLongitude", 
                   "homeTownLatitude", "homeTownLongitude"]:
        df[col].replace(0, np.nan, inplace=True)

#Manh Nguyen
    df['diaChi'] = df['diaChi'].str.split(',').str[-1]
    df[unicode_cols] = df[unicode_cols].applymap(str_normalize).applymap(lambda x: unidecode(x) if x==x else x)

    df['Field_45_Q'] = df['Field_45'].str[:-3].astype('category')
    df['Field_45_TP_55'] = df['Field_45'].str[:2] == df['Field_55']
    df['is_homeTown_diaChi'] = df['homeTownCity'] == df['diaChi']
    df['is_homeTown_current_city'] = df['homeTownCity'] == df['currentLocationCity']
    df['is_homeTown_current_state'] = df['homeTownState'] == df['currentLocationState']
    df['F48_49'] = df['Field_48'] == df['Field_49']
    
    #df["gender"] = df[["gioiTinh", "info_social_sex"]].apply(combine_gender, axis=1).astype("category")
    
    df[["currentLocationLocationId", "homeTownLocationId", "currentLocationLatitude", "currentLocationLongitude", 
        "homeTownLatitude", "homeTownLongitude"]].replace(0, np.nan, inplace=True) # value == 0: noisy

    df[["currentLocationLocationId", "homeTownLocationId"]] = (df[["currentLocationLocationId", "homeTownLocationId"]]
                                                             .applymap(str_normalize).astype("category"))
    df[object_cols] = df[object_cols].astype('category')
    df['F_63_64'] = df[['Field_63', 'Field_64']].apply(process_63_64, axis=1).astype('category')
#     df["currentLocationLocationId"] = df["currentLocationLocationId"].apply(str_normalize).astype("category")
#     df["homeTownLocationId"] = df["homeTownLocationId"].apply(str_normalize).astype("category")
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
    
def process_maCv(df):
    df["maCv"] = df["maCv"].apply(str_normalize).apply(job_category).astype("category")
    return df
    
def combine_gender(s):
    x, y = s
    return x if x != None else y if y != None else None

def process_gender(df):
    df["gender"] = df[["gioiTinh", "info_social_sex"]].apply(combine_gender, axis=1).astype("category")
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
            
    return df

def transform(df):
    df = process_datetime_cols(df)
    df = process_gender(df)
    df = process_maCv(df)
    df = process_location(df)
    df = process_ordinal(df)
    df["null_sum"] = df.isnull().sum(axis=1)
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

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval  = lgb.Dataset(X_val, y_val)

            evals_result = {}
            model = lgb.train(lgbm_param,
                        lgb_train,
                        num_boost_round=NUM_BOOST_ROUND,  
                        early_stopping_rounds=400,
                        feval=lgb_gini,
                        verbose_eval= 200,
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

    print("-" * 50)
    print("Avg train gini: {}".format(avg_train_gini))
    print("Avg valid gini: {}".format(avg_val_gini))
    print("=" * 50)
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

        y_label = df_train["label"]
        train_fe = df_fe[df_fe["id"] < df_train.shape[0]]
        test_fe = df_fe[df_fe["id"] >= df_train.shape[0]]

        # Label-Encoding
        lbl = LabelEncoder()
        for col in df_fe.columns:
          if df_fe[col].dtype.name == "category":
            train_fe[col] = train_fe[col].astype(str)
            test_fe[col] = test_fe[col].astype(str)
            encoder = ce.CountFrequencyCategoricalEncoder(encoding_method='frequency')
            train_fe[col] = train_fe[col].fillna('None')
            test_fe[col] = test_fe[col].fillna('None')

            # Only take the common values in Train/Test set
            common_vals = list(set(train_fe[col]).intersection(set(test_fe[col])))

            # Take if vals appeared both 5 times
            #common_vals = set(train_fe[col].value_counts()[train_fe[col].value_counts() > 4].index).intersection(test_fe[col].value_counts()[test_fe[col].value_counts()>4].index)

            # Replace not-common values with "Missing" or np.nan
            train_fe.loc[~train_fe[col].isin(common_vals), col] = 'Missing'
            test_fe.loc[~test_fe[col].isin(common_vals), col] = 'Missing'

            # Implement LE
            lbl.fit(train_fe[col].tolist() + test_fe[col].tolist())
            train_fe[col] = lbl.transform(train_fe[col])
            test_fe[col] = lbl.transform(test_fe[col])

        print(train_fe.shape)
        print(test_fe.shape)
    with timer("Kfold"):
        preds = kfold(train_fe,y_label,test_fe)
        df_test["label"] = preds
        df_test[['id', 'label']].to_csv('submission2.csv', index=False)

if __name__ == "__main__":
    submission_file_name = "submission2.csv"
    
    with timer("Full model run"):
        main()
from lightgbm import LGBMClassifier
import category_encoders as ce
# %matplotlib inline

train_df = pd.read_csv('../input/klps-creditscring-challenge-for-students/train.csv')
test_df = pd.read_csv('../input/klps-creditscring-challenge-for-students/test.csv')

"""## 2. Feature Engineering"""

def feature_engineering(train, test):
    labels = train['label']
    data = train.drop(columns=['label']).append(test, ignore_index=True)
    remove_features = ['Field_1', 'Field_2', 'Field_4', 'Field_5', 'Field_6', 'Field_7', 'Field_8', 'Field_9',
                       'Field_11', 'Field_12', 'Field_15', 'Field_18', 'Field_25', 'Field_32', 'Field_33',
                       'Field_34', 'Field_35', 'gioiTinh', 'diaChi', 'Field_36', 'Field_38', 'Field_40',
                       'Field_43', 'Field_44', 'Field_45', 'Field_46', 'Field_47', 'Field_48', 'Field_49',
                       'Field_54', 'Field_55', 'Field_56', 'Field_61', 'Field_62', 'Field_65', 'Field_66',
                       'Field_68', 'maCv', 'info_social_sex', 'data.basic_info.locale', 'currentLocationCity',
                       'currentLocationCountry', 'currentLocationName', 'currentLocationState', 'homeTownCity',
                       'homeTownCountry', 'homeTownName', 'homeTownState', 'F_startDate', 'F_endDate',
                       'E_startDate', 'E_endDate', 'C_startDate', 'C_endDate', 'G_startDate', 'G_endDate',
                       'A_startDate', 'A_endDate', 'brief']

    cat_features_count_encode = ['Field_4', 'Field_12', 'Field_18', 'Field_34', 'gioiTinh', 'diaChi', 'Field_36',
                                 'Field_38', 'Field_45', 'Field_46', 'Field_47', 'Field_48', 'Field_49',
                       'Field_54', 'Field_55', 'Field_56', 'Field_61', 'Field_62', 'Field_65', 'Field_66',
                       'Field_68', 'maCv', 'info_social_sex', 'data.basic_info.locale', 'currentLocationCity',
                       'currentLocationCountry', 'currentLocationName', 'currentLocationState', 'homeTownCity',
                       'homeTownCountry', 'homeTownName', 'homeTownState', 'brief']
    
    cat_date_array = ['Field_1', 'Field_2', 'Field_5', 'Field_6', 'Field_7', 'Field_8', 'Field_9', 'Field_11',
                      'Field_15', 'Field_25', 'Field_32', 'Field_33', 'Field_35', 'Field_40', 'Field_43',
                      'Field_44', 'F_startDate', 'F_endDate', 'E_startDate', 'E_endDate', 'C_startDate',
                      'C_endDate', 'G_startDate', 'G_endDate', 'A_startDate', 'A_endDate']
    for col in cat_date_array:
        data[col+'Year'] = pd.DatetimeIndex(data[col]).year
        data[col+'Month'] = pd.DatetimeIndex(data[col]).month
        data[col+'Day'] = pd.DatetimeIndex(data[col]).day
    
    data[remove_features].fillna("Missing", inplace=True)
    count_en = ce.CountEncoder()
    cat_ce = count_en.fit_transform(data[cat_features_count_encode])
    data = data.join(cat_ce.add_suffix("_ce"))
    
    data.replace("None", -1, inplace=True)
    data.replace("Missing", -999, inplace=True)
    data.fillna(-999, inplace=True)

    _train = data[data['id'] < 53030]
    _test = data[data['id'] >= 53030]
    
    _train["label"] = labels

    _train.drop(columns=remove_features, inplace=True)
    _test.drop(columns=remove_features, inplace=True)
    
    return _train, _test

train_data, test_data = feature_engineering(train_df, test_df)

"""## 3. Feature Selection"""

def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    dset = dset.sort_values(by='WoE')
    return dset, iv

USELESS_PREDICTOR = []
WEAK_PREDICTOR = []
MEDIUM_PREDICTOR = []
STRONG_PREDICTOR = []
GOOD_PREDICTOR = []
IGNORE_FEATURE = USELESS_PREDICTOR + WEAK_PREDICTOR
for col in train_data.columns:
    if col == 'label' or col == 'id': continue
    elif col in IGNORE_FEATURE: continue
    else:
        print('WoE and IV for column: {}'.format(col))
        final, iv = calculate_woe_iv(train_data, col, 'label')
        iv = round(iv,2)
        print('IV score: ' + str(iv))
        print('\n')
        if (iv < 0.02) and col not in USELESS_PREDICTOR:
            USELESS_PREDICTOR.append(col)
        elif iv >= 0.02 and iv < 0.1 and col not in WEAK_PREDICTOR:
            WEAK_PREDICTOR.append(col)
        elif iv >= 0.1 and iv < 0.3 and col not in MEDIUM_PREDICTOR:
            MEDIUM_PREDICTOR.append(col)
        elif iv >= 0.3 and iv < 0.5 and col not in STRONG_PREDICTOR:
            STRONG_PREDICTOR.append(col)
        elif iv >= 0.5 and col not in GOOD_PREDICTOR:
            GOOD_PREDICTOR.append(col)

print('USELESS_PREDICTOR')
print(len(USELESS_PREDICTOR))
print('WEAK_PREDICTOR')
print(len(WEAK_PREDICTOR))
print('MEDIUM_PREDICTOR')
print(len(MEDIUM_PREDICTOR))
print('STRONG_PREDICTOR')
print(len(STRONG_PREDICTOR))
print('GOOD_PREDICTOR')
print(len(GOOD_PREDICTOR))

IGNORE_FEATURE = USELESS_PREDICTOR
final_train_data = train_data.drop(columns=IGNORE_FEATURE)
final_test_data = test_data.drop(columns=[col for col in IGNORE_FEATURE if col not in ['label']])

"""## 4. Build Model"""

import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix, recall_score, classification_report
import seaborn as sns


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
    
def display_roc_curve(y_, oof_preds_,sub_preds_,folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds_idx_):  
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = 2 * auc(fpr, tpr) -1
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (Gini = %0.4f)' % (n_fold + 1, score))
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = 2 * auc(fpr, tpr) -1
    plt.plot(fpr, tpr, color='b',
             label='Avg ROC (Gini = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')


# LightGBM GBDT with Stratified KFold
def kfold_lightgbm(train_df, test_df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    gc.collect()
    # Cross validation model
    folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=500)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['label','id']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['label'])):        
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['label'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['label'].iloc[valid_idx]

        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=128,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1
        )

        clf.fit(train_x, train_y.ravel(), eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric='auc', verbose= 1000, early_stopping_rounds= 200)

        oof_pred = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        
        pred = clf.predict(valid_x, num_iteration=clf.best_iteration_)
        print('F1 Score: ' + str( f1_score(valid_y, pred) ))
        print('Recall Score: ' + str( recall_score(valid_y, pred) ))
        
        sub_pred = clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        oof_preds[valid_idx] = oof_pred
        sub_preds += sub_pred
                
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['label'], oof_preds))
    
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(train_df[feats], train_df['label'])]
    display_roc_curve(y_=train_df['label'],oof_preds_=oof_preds,sub_preds_ = sub_preds, folds_idx_=folds_idx)
    
    # Write submission file and plot feature importance
    test_df['label'] = sub_preds
    test_df[['id', 'label']].to_csv('submission3.csv', index= False)
    #display_importances(feature_importance_df)

kfold_lightgbm(final_train_data, final_test_data, 5)
plt.style.use('fivethirtyeight')

from unidecode import unidecode

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import category_encoders as ce

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/klps-creditscring-challenge-for-students/train.csv')
test = pd.read_csv('../input/klps-creditscring-challenge-for-students/test.csv')

"""# 2. Feature Engineering"""

# Drop some columns are duplicated, with correlation = NaN
ignore_columns = (["gioiTinh", "info_social_sex", "ngaySinh", "namSinh"] + 
        [f"Field_{c}" for c in [14, 16, 17, 24, 26, 30, 31, 37, 52, 57]] + 
        ['partner0_K', 'partner0_L', 
         'partner1_B', 'partner1_D', 'partner1_E', 'partner1_F', 'partner1_K', 'partner1_L',
         'partner2_B', 'partner2_G', 'partner2_K', 'partner2_L',
         'partner3_B', 'partner3_C', 'partner3_F', 'partner3_G', 'partner3_H', 'partner3_K', 'partner3_L',
         *['partner4_' + i for i in 'ABCDEFGHK'],
         'partner5_B', 'partner5_C', 'partner5_H', 'partner5_K', 'partner5_L'])

# Some auto columns could make new better columns
all_auto_columns = list(set([c for c in train.columns if train[c].dtype in [np.int64, np.float64]])
                    .difference(ignore_columns + ['currentLocationLocationId', 'homeTownLocationId', 'label', 'id']))

auto_columns_1 = [c for c in all_auto_columns if 'Field_' in c]
auto_columns_2 = [c for c in all_auto_columns if 'partner' in c]
auto_columns_3 = [c for c in all_auto_columns if 'num' in c]
auto_columns_4 = [c for c in all_auto_columns if c not in auto_columns_1 + auto_columns_2 + auto_columns_3]
print(len(auto_columns_1), len(auto_columns_2), len(auto_columns_3), len(auto_columns_4), len(all_auto_columns))

"""### 2.1. Datetime columns"""

date_cols = ["Field_{}".format(i) for i in [5, 6, 7, 8, 9, 11, 15, 25, 32, 33, 35, 40]]
datetime_cols = ["Field_{}".format(i) for i in [1, 2, 43, 44]]
correct_dt_cols = ['Field_34', 'ngaySinh']
cat_cols = date_cols + datetime_cols + correct_dt_cols

# Normalize Field_34, ngaySinh
def ngaysinh_34_normalize(s):
    if s != s: return np.nan
    try: s = int(s)
    except ValueError: s = s.split(" ")[0]
    return datetime.strptime(str(s)[:6], "%Y%m")

# Normalize datetime data
def datetime_normalize(s):
    if s != s: return np.nan
    s = s.split(".")[0]
    if s[-1] == "Z": s = s[:-1]
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

# Normalize date data
def date_normalize(s):
    if s != s: return np.nan
    try: t = datetime.strptime(s, "%m/%d/%Y")
    except: t = datetime.strptime(s, "%Y-%m-%d")
    return t

def process_datetime_cols(df):
    df[datetime_cols] = df[datetime_cols].applymap(datetime_normalize)  
    df[date_cols] = df[date_cols].applymap(date_normalize)
    df[correct_dt_cols] = df[correct_dt_cols].applymap(ngaysinh_34_normalize)

    # Some delta columns
    for i, j in zip('43 1 2'.split(), '1 2 44'.split()): df[f'DT_{j}_{i}'] = (df[f'Field_{j}'] - df[f'Field_{i}']).dt.seconds
    for i, j in zip('5 6 7 33 8 11 9 15 25 6 7 8 9 15 25 2'.split(), '6 34 33 40 11 35 15 25 32 7 8 9 15 25 32 8'.split()): 
        df[f'DT_{j}_{i}'] = (df[f'Field_{j}'] - df[f'Field_{i}']).dt.days
    
    # Age, month
    df['age'] = 2020 - pd.DatetimeIndex(df['ngaySinh']).year
    df['birth_month'] = pd.DatetimeIndex(df['ngaySinh']).month
    
    # Days from current time & isWeekday
    for col in cat_cols:
        name = col.split('_')[-1]
        df[f'is_WD_{name}'] = df[col].dt.dayofweek.isin(range(5))
        df[f'days_from_now_{name}'] = (datetime.now() - pd.DatetimeIndex(df[col])).days
        df[col] = df[col].dt.strftime('%m-%Y')
    
    # Delta for x_startDate and x_endDate
    for cat in ['F', 'E', 'C', 'G', 'A']:
        df[f'{cat}_startDate'] = pd.to_datetime(df[f"{cat}_startDate"], infer_datetime_format=True)
        df[f'{cat}_endDate'] = pd.to_datetime(df[f"{cat}_endDate"], infer_datetime_format=True)
        
        df[f'{cat}_start_end'] = (df[f'{cat}_endDate'] - df[f'{cat}_startDate']).dt.days
        
    for i, j in zip('F E C G'.split(), 'E C G A'.split()):
        df[f'{j}_{i}_startDate'] = (df[f'{j}_startDate'] - df[f'{i}_startDate']).dt.days
        df[f'{j}_{i}_endDate'] = (df[f'{j}_endDate'] - df[f'{i}_endDate']).dt.days
    
    temp_date = [f'{i}_startDate' for i in 'ACEFG'] + [f'{i}_endDate' for i in 'ACEFG']
    
    for col in temp_date:
        df[col] = df[col].dt.strftime('%m-%Y')
        
    for col in cat_cols + temp_date:
        df[col] = df[col].astype("category")
        
    return df

"""### 2.2. Categorical columns"""

unicode_cols = ['Field_18', 'maCv', 'diaChi', 'Field_46', 'Field_48', 'Field_49', 'Field_56', 'Field_61', 'homeTownCity', 
                'homeTownName', 'currentLocationCity', 'currentLocationName', 'currentLocationState', 'homeTownState']
object_cols = (unicode_cols + 
               [f'Field_{str(i)}' for i in '4 12 36 38 47 62 45 54 55 65 66 68'.split()] +
               ['data.basic_info.locale', 'currentLocationCountry', 'homeTownCountry', 'brief'])

def str_normalize(s):
    s = str(s).strip().lower()
    s = re.sub(' +', " ", s)
    return s

def combine_gender(s):
    x, y = s 
    if x != x and y != y: return "nan"
    if x != x: return y.lower()
    return x.lower()

def process_categorical_cols(df):
    df['diaChi'] = df['diaChi'].str.split(',').str[-1]
    df[unicode_cols] = df[unicode_cols].applymap(str_normalize).applymap(lambda x: unidecode(x) if x==x else x)
    
    # Normalize some columns
    df["Field_38"] = df["Field_38"].map({0: 0.0, 1: 1.0, "DN": np.nan, "TN": np.nan, "GD": np.nan})
    df["Field_62"] = df["Field_62"].map({"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "Ngoài quốc doanh Quận 7": np.nan})
    df["Field_47"] = df["Field_47"].map({"Zezo": 0, "One": 1, "Two": 2, "Three": 3, "Four": 4})
    
    # Make some new features
    df['Field_45_Q'] = df['Field_45'].str[:-3].astype('category')
    df['Field_45_TP_55'] = df['Field_45'].str[:2] == df['Field_55']
    df['is_homeTown_diaChi'] = df['homeTownCity'] == df['diaChi']
    df['is_homeTown_current_city'] = df['homeTownCity'] == df['currentLocationCity']
    df['is_homeTown_current_state'] = df['homeTownState'] == df['currentLocationState']
    df['F48_49'] = df['Field_48'] == df['Field_49']
    
    df["gender"] = df[["gioiTinh", "info_social_sex"]].apply(combine_gender, axis=1).astype("category")
    
    df[["currentLocationLocationId", "homeTownLocationId", "currentLocationLatitude", "currentLocationLongitude", 
        "homeTownLatitude", "homeTownLongitude"]].replace(0, np.nan, inplace=True) # value == 0: noisy

    df[["currentLocationLocationId", "homeTownLocationId"]] = (df[["currentLocationLocationId", "homeTownLocationId"]]
                                                             .applymap(str_normalize).astype("category"))
    df[object_cols] = df[object_cols].astype('category')
    
    return df

"""### 2.3. Others"""

# New feature from columns 63, 64
def process_63_64(z):
    x, y = z
    if x != x and y != y:
        return np.nan
    if (x, y) in [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 8.0), (7.0, 5.0), (5.0, 6.0), (9.0, 43.0), (8.0, 9.0)]: return True
    else: return False
    
def process_others(df):        
    df[["Field_27", "Field_28"]].replace(0.0, np.nan, inplace=True)
    df['F18_isnumeric'] = df['Field_18'].str.isnumeric()
    df['F18_isalpha'] = df['Field_18'].str.isalpha()
    
    # Delta from some pairs of columns
    for i, j in [(20, 27), (28, 27), (39, 41), (41, 42), (50, 51), (51, 53)]:
        df[f'F{str(i)}_{str(j)}_delta'] = df[f'Field_{str(j)}'] - df[f'Field_{str(i)}']
    df['F_59_60'] = df['Field_59'] - df['Field_60'] - 2
    df['F_63_64'] = df[['Field_63', 'Field_64']].apply(process_63_64, axis=1).astype('category')
    
    # Mean, std from partnerX columns
    for i in '1 2 3 4 5'.split():
        col = [c for c in df.columns if f'partner{i}' in c]
        df[f'partner{i}_mean'] = df[col].mean(axis=1)
        df[f'partner{i}_std'] = df[col].std(axis=1)

    # Reference columns
    columns = set(df.columns).difference(ignore_columns)
    df['cnt_NaN'] = df[columns].isna().sum(axis=1)
    df['cnt_True'] = df[columns].applymap(lambda x: isinstance(x, bool) and x).sum(axis=1)
    df['cnt_False'] = df[columns].applymap(lambda x: isinstance(x, bool) and not x).sum(axis=1)

    # Combinations of auto columns
    lst_combination = (list(combinations(auto_columns_2, 2)) + list(combinations(auto_columns_3, 2)) + list(combinations(auto_columns_4, 2)))
    for l, r in lst_combination:
        for func in 'add subtract divide multiply'.split():
            df[f'auto_{func}_{l}_{r}'] = getattr(np, func)(df[l], df[r])
            
    return df

"""### 2.4. Combine all parts"""

def transform(df):
    df = process_datetime_cols(df)
    df = process_categorical_cols(df)
    df = process_others(df)
    return df.drop(ignore_columns, axis=1)

train = transform(train)
test = transform(test)

"""### 2.5. Try Count Encoding"""

# Support catboost modelling
cat_features = [c for c in train.columns if (train[c].dtype not in [np.float64, np.int64])]
train[cat_features] = train[cat_features].astype(str)
test[cat_features] = test[cat_features].astype(str)

# Create the encoder
t = pd.concat([train, test]).reset_index(drop=True)
count_enc = ce.CountEncoder().fit_transform(t[cat_features])
tt = t.join(count_enc.add_suffix("_count"))

f2_train = tt.loc[tt.index < train.shape[0]]
f2_test = tt.loc[tt.index >= train.shape[0]]

columns = sorted(set(f2_train.columns).intersection(f2_test.columns))
print(len(columns))

"""# 3. Modelling

### 3.1. Model
"""

gini, feature_importance_df = {}, pd.DataFrame()

TRAIN, TEST = f2_train[columns].drop(['id', 'label'], axis=1), f2_test[columns].drop(['id', 'label'], axis=1)
LABEL = f2_train['label']
preds, oof_preds = np.zeros(TRAIN.shape[0]), {}

cv = StratifiedKFold(n_splits=5, shuffle=True)
for i, (train_idx, val_idx) in enumerate(cv.split(TRAIN, LABEL)):
    X_train, y_train = TRAIN.iloc[train_idx], LABEL.iloc[train_idx]
    X_val, y_val = TRAIN.iloc[val_idx], LABEL.iloc[val_idx]

    gbm = CatBoostClassifier(eval_metric='AUC', 
                             use_best_model=True,
                             iterations=1000, 
                             learning_rate=0.1, 
                             random_seed=42).fit(X_train, y_train, 
                                                 cat_features=set(cat_features),
                                                 eval_set=(X_val, y_val), verbose=500)

    y_pred = gbm.predict(X_val)
    y_pred_proba = gbm.predict_proba(X_val)[:, 1]
        
    preds[val_idx] = y_pred_proba
    oof_preds[f'F{i+1}'] = gbm.predict_proba(TEST)[:, 1]
    
    gini[f'F{i+1}'] = 2 * roc_auc_score(y_val, y_pred_proba) - 1
    
    # For create feature importances
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = TRAIN.columns
    fold_importance_df["importance"] = gbm.feature_importances_
    fold_importance_df["fold"] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('Fold %2d GINI : %.5f' % (i + 1, 2*roc_auc_score(y_val, y_pred_proba) - 1))
    
# Resulting
roc_auc = roc_auc_score(LABEL, preds)
print('Avg GINI score:', 2*roc_auc - 1)

result = np.array(list(gini.values()))
print('GINI: {:.5f} +- {:.5f}'.format(result.mean(), result.std()))

# One tips to better performance is re-training the model with dropped low important columns
to_drop = feature_importance_df[feature_importance_df.importance == 0]['feature'].unique()
to_drop.shape

"""# 4. Submisison"""

test['label'] = pd.DataFrame(oof_preds).mean(axis=1).values
test[['id', 'label']].to_csv('submission4.csv', index=False)
sub1 = pd.read_csv('./submission1.csv')
sub2 = pd.read_csv('./submission2.csv')
sub3 = pd.read_csv('./submission3.csv')
sub4 = pd.read_csv('./submission4.csv')
submission_final = pd.DataFrame()
submission_final['id'] = sub1['id']
submission_final['label'] = sub1['label']*0.4 + sub2['label']*0.1 + sub3['label']*0.2 + sub1['label']*0.3
submission_final.to_csv('submission_final.csv', index=False)
submission_final.head()