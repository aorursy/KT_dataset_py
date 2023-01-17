import numpy as np

import scipy as sp

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import optuna

import gc

import sys



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.feature_extraction.text import CountVectorizer

from category_encoders import OrdinalEncoder

from tqdm import tqdm_notebook as tqdm



import lightgbm as lgb



!pip install optuna

import optuna
filepath1 = "/kaggle/input/aiacademy-pickled-datasets/"

filepath2 = "../input/homework-for-students2/"



# filepath1 = ""

# filepath2 = "../input/"
# テーブルのインポート

df_train = pd.read_csv(filepath2+"train.csv",parse_dates = ["issue_d","earliest_cr_line"])

df_test = pd.read_csv(filepath2+"test.csv",parse_dates = ["issue_d","earliest_cr_line"])
# その他設定

SEED = 139

TARGET = "loan_condition"
# zip_codeとstateのカテゴリを作成

df_train["state_zip_code"] = df_train["zip_code"].astype(str)+"_"+df_train["addr_state"].astype(str)

df_test["state_zip_code"] = df_test["zip_code"].astype(str)+"_"+df_test["addr_state"].astype(str)



# zip_codeを数値化

df_test["zip_code"] = df_test["zip_code"].str[0:3].astype(int)

df_train["zip_code"] = df_train["zip_code"].str[0:3].astype(int)



# 時間を保存

df_train["issue_year"]=df_train.issue_d.dt.year

df_train["issue_month"]=df_train.issue_d.dt.month

df_train["earliest_cr_line_year"]=df_train.earliest_cr_line.dt.year

df_train["earliest_cr_line_month"]=df_train.earliest_cr_line.dt.month

df_test["issue_year"]=df_test.issue_d.dt.year

df_test["issue_month"]=df_test.issue_d.dt.month

df_test["earliest_cr_line_year"]=df_test.earliest_cr_line.dt.year

df_test["earliest_cr_line_month"]=df_test.earliest_cr_line.dt.month



# ダミーカラムを追加

df_train["dummy"] = 0

df_test["dummy"] = 0



# トレインとテストを区別

df_train["istest"] = 0

df_test["istest"] = 1
# オリジナルを保存

df_train_base = df_train.copy()

df_test_base = df_test.copy()
# テキストカラムを保存

txt_col = ['emp_title',"title"]



# カテゴリカラムの抽出

cat = []

for col in df_train.columns:

    if (df_train[col].dtype == "object") and (col not in txt_col ):

        cat.append(col)



# 利用予定のカラム

f_col = []

        

# 削除カラム

del_col = []



# 不要なカラムに追加

del_col.extend(["ID","issue_d","earliest_cr_line",'emp_title',"title","loan_condition","istest","dummy"])
f_col = ['loan_amnt',

 'installment',

 'grade',

 'sub_grade',

 'home_ownership',

 'annual_inc',

 'purpose',

 'zip_code',

 'addr_state',

 'dti',

 'delinq_2yrs',

 'inq_last_6mths',

 'mths_since_last_delinq',

 'mths_since_last_record',

 'open_acc',

 'pub_rec',

 'revol_bal',

 'revol_util',

 'total_acc',

 'collections_12_mths_ex_med',

 'mths_since_last_major_derog',

 'acc_now_delinq',

 'tot_coll_amt',

 'tot_cur_bal',

 'state_zip_code',

 'earliest_cr_line_year']
del_col = ['ID',

 'issue_d',

 'earliest_cr_line',

 'emp_title',

 'title',

 'loan_condition',

 'istest',

 'dummy',

 'initial_list_status',

 'application_type',

 'issue_year',

 'issue_month',

 'earliest_cr_line_month']
# numpyとして保存



# カテゴリは変換しておく

for col in f_col:

    if col in cat:

        le = OrdinalEncoder()

        le.fit(df_train[col].sort_values())    

        df_train[col] = le.fit_transform(df_train[col])    

        df_test[col] = le.transform(df_test[col])    

        print(le.mapping)



X_final_train = df_train.loc[:,f_col].values.astype(float)

y_final_train = df_train[TARGET].values.astype(int)

X_final_test = df_test.loc[:,f_col].values.astype(float)
train_idx = df_train[(df_train.issue_year<2015) | (df_train.issue_month < 7)].index.tolist()

val_idx = df_train[(df_train.issue_year == 2015) & (df_train.issue_month>=7)].index.tolist()
# カラム番号を保存

f_col_num = {}

for idx, col in enumerate(f_col):

    f_col_num[col] = idx
del df_train,df_test
# オリジナルを読み込み

df_train = df_train_base.copy()

df_test = df_test_base.copy()
# emp_lengthを整理

df_train.loc[df_train["emp_length"] == '< 1 year',"emp_length_year"] = 0

df_train.loc[df_train["emp_length"] == '1 year',"emp_length_year"] = 1

df_train.loc[df_train["emp_length"] == '2 years',"emp_length_year"] = 2

df_train.loc[df_train["emp_length"] == '3 years',"emp_length_year"] = 3

df_train.loc[df_train["emp_length"] == '4 years',"emp_length_year"] = 4

df_train.loc[df_train["emp_length"] == '5 years',"emp_length_year"] = 5

df_train.loc[df_train["emp_length"] == '6 years',"emp_length_year"] = 6

df_train.loc[df_train["emp_length"] == '7 years',"emp_length_year"] = 7

df_train.loc[df_train["emp_length"] == '8 years',"emp_length_year"] = 8

df_train.loc[df_train["emp_length"] == '9 years',"emp_length_year"] = 9

df_train.loc[df_train["emp_length"] == '10+ years',"emp_length_year"] = 10

df_train.loc[df_train["emp_length"] == '10+ years',"over_10_year"] = 1

df_train.loc[df_train["emp_length"] != '10+ years',"over_10_year"] = 0

df_train.loc[df_train["emp_length"] == '< 1 year',"under_1_year"] = 1

df_train.loc[df_train["emp_length"] != '< 1 year',"under_1_year"] = 0

df_train["isnull_emp_length"] = df_train["emp_length"].isnull()*1



df_test.loc[df_test["emp_length"] == '< 1 year',"emp_length_year"] = 0

df_test.loc[df_test["emp_length"] == '1 year',"emp_length_year"] = 1

df_test.loc[df_test["emp_length"] == '2 years',"emp_length_year"] = 2

df_test.loc[df_test["emp_length"] == '3 years',"emp_length_year"] = 3

df_test.loc[df_test["emp_length"] == '4 years',"emp_length_year"] = 4

df_test.loc[df_test["emp_length"] == '5 years',"emp_length_year"] = 5

df_test.loc[df_test["emp_length"] == '6 years',"emp_length_year"] = 6

df_test.loc[df_test["emp_length"] == '7 years',"emp_length_year"] = 7

df_test.loc[df_test["emp_length"] == '8 years',"emp_length_year"] = 8

df_test.loc[df_test["emp_length"] == '9 years',"emp_length_year"] = 9

df_test.loc[df_test["emp_length"] == '10+ years',"emp_length_year"] = 10

df_test.loc[df_test["emp_length"] == '10+ years',"over_10_year"] = 1

df_test.loc[df_test["emp_length"] != '10+ years',"over_10_year"] = 0

df_test.loc[df_test["emp_length"] == '< 1 year',"under_1_year"] = 1

df_test.loc[df_test["emp_length"] != '< 1 year',"under_1_year"] = 0

df_test["isnull_emp_length"] = df_test["emp_length"].isnull()*1



del_col.append("emp_length")
# ISNULLカラム作成

df_train["isnull_annual_inc"] = df_train["annual_inc"].isnull()*1

df_train["isnull_delinq_2yrs"] = df_train["delinq_2yrs"].isnull()*1

df_train["isnull_mths_since_last_delinq"] = df_train["mths_since_last_delinq"].isnull()*1

df_train["isnull_mths_since_last_record"] = df_train["mths_since_last_record"].isnull()*1

df_train["isnull_mths_since_last_major_derog"] = df_train["mths_since_last_major_derog"].isnull()*1

df_train["isnull_emp_length"] = df_train["emp_length"].isnull()*1

df_train["isnull_emp_title"] = df_train["emp_title"].isnull()*1

df_train["isnull_inq_last_6mths"] = df_train["inq_last_6mths"].isnull()*1

df_train["isnull_collections_12_mths_ex_med"] = df_train["collections_12_mths_ex_med"].isnull()*1

df_train["isnull_pub_rec"] = df_train["pub_rec"].isnull()*1



df_test["isnull_annual_inc"] = df_test["annual_inc"].isnull()*1

df_test["isnull_delinq_2yrs"] = df_test["delinq_2yrs"].isnull()*1

df_test["isnull_mths_since_last_delinq"] = df_test["mths_since_last_delinq"].isnull()*1

df_test["isnull_mths_since_last_record"] = df_test["mths_since_last_record"].isnull()*1

df_test["isnull_mths_since_last_major_derog"] = df_test["mths_since_last_major_derog"].isnull()*1

df_test["isnull_emp_length"] = df_test["emp_length"].isnull()*1

df_test["isnull_emp_title"] = df_test["emp_title"].isnull()*1

df_test["isnull_inq_last_6mths"] = df_test["inq_last_6mths"].isnull()*1

df_test["isnull_collections_12_mths_ex_med"] = df_test["collections_12_mths_ex_med"].isnull()*1

df_test["isnull_pub_rec"] = df_test["pub_rec"].isnull()*1
# リボ払いの有無

df_train["isnull_revol_util"] = df_train["revol_util"].isnull()*1

df_train["isnull_revol_bal"] = df_train["revol_bal"].isnull()*1

df_train["iszero_revol_util"] = (df_train["revol_util"] <= 0)*1

df_train["iszero_revol_bal"] = (df_train["revol_bal"] <= 0)*1



df_test["isnull_revol_util"] = df_test["revol_util"].isnull()*1

df_test["isnull_revol_bal"] = df_test["revol_bal"].isnull()*1

df_test["iszero_revol_util"] = (df_test["revol_util"] <= 0)*1

df_test["iszero_revol_bal"] = (df_test["revol_bal"] <= 0)*1
# Subgradeの分離

df_train["sub_grade_num"] = df_train["sub_grade"].str[1].astype(int)

df_test["sub_grade_num"] = df_test["sub_grade"].str[1].astype(int)
# 外部データのGDPロード

df_GDP = pd.read_csv(filepath2+"US_GDP_by_State.csv").rename(columns={"State":"City"})

df_GDP["SLS/PM"] = df_GDP["State & Local Spending"]/df_GDP["Population (million)"]

df_GDP["GSP/PM"] = df_GDP["Gross State Product"]/df_GDP["Population (million)"]

df_GDP["RSG/PM"] = df_GDP["Real State Growth %"]/df_GDP["Population (million)"]

df_GDP["SLS/GSP"] = df_GDP["State & Local Spending"]/df_GDP["Gross State Product"]

df_GDP["RSG/SLS"] = df_GDP["Real State Growth %"]/df_GDP["State & Local Spending"]

df_GDP["GSP/RSG"] = df_GDP["Gross State Product"]/df_GDP["Real State Growth %"]



df_GDP_2013 = df_GDP[df_GDP["year"]==2013].drop("year",axis=1)

df_GDP_2014 = df_GDP[df_GDP["year"]==2014].drop("year",axis=1)

df_GDP_2015 = df_GDP[df_GDP["year"]==2015].drop("year",axis=1)



df_GDP_2013.columns = df_GDP_2013.columns.map(lambda x:"2013_"+x if x != "City" else x)

df_GDP_2014.columns = df_GDP_2014.columns.map(lambda x:"2014_"+x if x != "City" else x)

df_GDP_2015.columns = df_GDP_2015.columns.map(lambda x:"2015_"+x if x != "City" else x)



df_GDP_year = pd.merge(pd.merge(df_GDP_2013,df_GDP_2014,on="City",how="left"),df_GDP_2015,on="City",how="left")
df_GDP_year["diff_SLS_14-13"] = df_GDP_year["2014_State & Local Spending"] - df_GDP_year["2013_State & Local Spending"]

df_GDP_year["diff_SLS_15-14"] = df_GDP_year["2015_State & Local Spending"] - df_GDP_year["2014_State & Local Spending"]

df_GDP_year["diff_SLS_15-13"] = df_GDP_year["2015_State & Local Spending"] - df_GDP_year["2013_State & Local Spending"]



df_GDP_year["diff_GSP_14-13"] = df_GDP_year["2014_Gross State Product"] - df_GDP_year["2013_Gross State Product"]

df_GDP_year["diff_GSP_15-14"] = df_GDP_year["2015_Gross State Product"] - df_GDP_year["2014_Gross State Product"]

df_GDP_year["diff_GSP_15-13"] = df_GDP_year["2015_Gross State Product"] - df_GDP_year["2013_Gross State Product"]



df_GDP_year["diff_RSG_14-13"] = df_GDP_year["2014_Real State Growth %"] - df_GDP_year["2013_Real State Growth %"]

df_GDP_year["diff_RSG_15-14"] = df_GDP_year["2015_Real State Growth %"] - df_GDP_year["2014_Real State Growth %"]

df_GDP_year["diff_RSG_15-13"] = df_GDP_year["2015_Real State Growth %"] - df_GDP_year["2013_Real State Growth %"]



df_GDP_year["diff_PM_14-13"] = df_GDP_year["2014_Population (million)"] - df_GDP_year["2013_Population (million)"]

df_GDP_year["diff_PM_15-14"] = df_GDP_year["2015_Population (million)"] - df_GDP_year["2014_Population (million)"]

df_GDP_year["diff_PM_15-13"] = df_GDP_year["2015_Population (million)"] - df_GDP_year["2013_Population (million)"]
df_state = pd.read_csv(filepath2+"statelatlong.csv")

state_GDP = pd.merge(df_state,df_GDP_year,on="City",how="left").drop("City",axis=1)

state_GDP = state_GDP.rename(columns={"State":"addr_state"})
# 外部データのfree-zipcode-database.csvロード

df_zip = pd.read_csv(filepath2+"free-zipcode-database.csv").rename(columns = {"State":"addr_state"})

df_zip["zip_code"] = df_zip["Zipcode"].astype(str).str[0:3].astype(int)



df_zip = df_zip[(~df_zip["TaxReturnsFiled"].isnull()) |(~df_zip["EstimatedPopulation"].isnull()) |(~df_zip["TotalWages"].isnull())]



df_zip_min = df_zip.groupby(["addr_state","zip_code"],as_index=False)["TaxReturnsFiled","EstimatedPopulation","TotalWages"].min()

df_zip_min.columns = ["addr_state","zip_code","min_TaxReturnsFiled","min_EstimatedPopulation","min_TotalWages"]

df_zip_mean = df_zip.groupby(["addr_state","zip_code"],as_index=False)["TaxReturnsFiled","EstimatedPopulation","TotalWages"].mean()

df_zip_mean.columns = ["addr_state","zip_code","mean_TaxReturnsFiled","mean_EstimatedPopulation","mean_TotalWages"]

df_zip_max = df_zip.groupby(["addr_state","zip_code"],as_index=False)["TaxReturnsFiled","EstimatedPopulation","TotalWages"].max()

df_zip_max.columns = ["addr_state","zip_code","max_TaxReturnsFiled","max_EstimatedPopulation","max_TotalWages"]



df_zip_state_min = df_zip.groupby(["addr_state"],as_index=False)["TaxReturnsFiled","EstimatedPopulation","TotalWages"].min()

df_zip_state_min.columns = ["addr_state","min_TaxReturnsFiled_state","min_EstimatedPopulation_state","min_TotalWages_state"]

df_zip_state_mean = df_zip.groupby(["addr_state"],as_index=False)["TaxReturnsFiled","EstimatedPopulation","TotalWages"].mean()

df_zip_state_mean.columns = ["addr_state","mean_TaxReturnsFiled_state","mean_EstimatedPopulation_state_state","mean_TotalWages_state"]

df_zip_state_max = df_zip.groupby(["addr_state"],as_index=False)["TaxReturnsFiled","EstimatedPopulation","TotalWages"].max()

df_zip_state_max.columns = ["addr_state","max_TaxReturnsFiled_state","max_EstimatedPopulation_state","max_TotalWages_state"]
df_zip_zip = pd.merge(df_zip_max,pd.merge(df_zip_min,df_zip_mean,on=["addr_state","zip_code"],how="left"),on=["addr_state","zip_code"],how="left")

df_zip_state = pd.merge(df_zip_state_max,pd.merge(df_zip_state_min,df_zip_state_mean,on=["addr_state"],how="left"),on=["addr_state"],how="left")
# Train/Testにマージ

df_train = pd.merge(df_train,state_GDP,on="addr_state",how="left")

df_test = pd.merge(df_test,state_GDP,on="addr_state",how="left")

del df_GDP,df_GDP_2013,df_GDP_2014,df_GDP_2015,df_GDP_year,state_GDP
# それぞれをTrainにマージ

df_train = pd.merge(df_train,df_zip_zip,on=["addr_state","zip_code"],how="left")

df_train = pd.merge(df_train,df_zip_state,on=["addr_state"],how="left")



df_test = pd.merge(df_test,df_zip_zip,on=["addr_state","zip_code"],how="left")

df_test = pd.merge(df_test,df_zip_state,on=["addr_state"],how="left")

del df_zip,df_zip_zip,df_zip_state,df_zip_max,df_zip_mean,df_zip_min
# タイトルを集計

df_train["title"] = df_train["title"].str.lower().str.strip()

df_train["len_title"] = df_train["title"].str.len()

df_train["len_title"] = df_train["len_title"].astype(float)



df_test["title"] = df_test["title"].str.lower().str.strip()

df_test["len_title"] = df_test["title"].str.len()

df_test["len_title"] = df_test["len_title"].astype(float)



df_train["num_of_words_title"] = df_train["title"].astype(str).str.split(" ").apply(lambda x:len(x))

df_test["num_of_words_title"] = df_test["title"].astype(str).str.split(" ").apply(lambda x:len(x))
# 職業を集計

df_train["emp_title"] = df_train["emp_title"].str.lower().str.strip()

df_train["len_emp_title"] = df_train["emp_title"].str.len()

df_test["emp_title"] = df_test["emp_title"].str.lower().str.strip()

df_test["len_emp_title"] = df_test["emp_title"].str.len()

df_train["num_of_words_emp_title"] = df_train["emp_title"].astype(str).str.split(" ").apply(lambda x:len(x))

df_test["num_of_words_emp_title"] = df_test["emp_title"].astype(str).str.split(" ").apply(lambda x:len(x))



# TOP30以外をotherとしてでラベルエンコーディング

tmp = df_train["emp_title"] .value_counts()

df_train["emp_title_label"] = df_train["emp_title"]

df_train.loc[~df_train["emp_title_label"].isin(tmp[0:30].index),"emp_title_label"]  = "others"

df_test["emp_title_label"] = df_test["emp_title"]

df_test.loc[~df_test["emp_title_label"].isin(tmp[0:30].index),"emp_title_label"]  = "others"



cat.append("emp_title_label")
new_col = ['emp_length_year',

 'over_10_year',

 'under_1_year',

 'isnull_emp_length',

 'isnull_mths_since_last_delinq',

 'isnull_mths_since_last_record',

 'isnull_mths_since_last_major_derog',

 'isnull_emp_title',

 'iszero_revol_util',

 'iszero_revol_bal',

 'sub_grade_num',

 'Latitude',

 'Longitude',

 '2013_State & Local Spending',

 '2013_Gross State Product',

 '2013_Real State Growth %',

 '2013_Population (million)',

 '2013_SLS/PM',

 '2013_GSP/PM',

 '2013_RSG/PM',

 '2013_SLS/GSP',

 '2013_RSG/SLS',

 '2013_GSP/RSG',

 '2014_State & Local Spending',

 '2014_Gross State Product',

 '2014_Real State Growth %',

 '2014_Population (million)',

 '2014_SLS/PM',

 '2014_GSP/PM',

 '2014_RSG/PM',

 '2014_SLS/GSP',

 '2014_RSG/SLS',

 '2014_GSP/RSG',

 '2015_State & Local Spending',

 '2015_Gross State Product',

 '2015_Real State Growth %',

 '2015_Population (million)',

 '2015_SLS/PM',

 '2015_GSP/PM',

 '2015_RSG/PM',

 '2015_SLS/GSP',

 '2015_RSG/SLS',

 '2015_GSP/RSG',

 'diff_SLS_14-13',

 'diff_SLS_15-14',

 'diff_SLS_15-13',

 'diff_GSP_14-13',

 'diff_GSP_15-14',

 'diff_GSP_15-13',

 'diff_RSG_14-13',

 'diff_RSG_15-14',

 'diff_RSG_15-13',

 'diff_PM_14-13',

 'diff_PM_15-14',

 'diff_PM_15-13',

 'max_TaxReturnsFiled',

 'max_EstimatedPopulation',

 'max_TotalWages',

 'min_TaxReturnsFiled',

 'min_EstimatedPopulation',

 'min_TotalWages',

 'mean_TaxReturnsFiled',

 'mean_EstimatedPopulation',

 'mean_TotalWages',

 'max_TaxReturnsFiled_state',

 'max_EstimatedPopulation_state',

 'max_TotalWages_state',

 'min_TaxReturnsFiled_state',

 'min_EstimatedPopulation_state',

 'min_TotalWages_state',

 'mean_TaxReturnsFiled_state',

 'mean_EstimatedPopulation_state_state',

 'mean_TotalWages_state',

 'len_title',

 'num_of_words_title',

 'len_emp_title',

 'num_of_words_emp_title',

 'emp_title_label']
f_col.extend(new_col)



# カラム番号を保存

f_col_num = {}

for idx, col in enumerate(f_col):

    f_col_num[col] = idx
# カテゴリは変換しておく

for col in new_col:

    if col in cat:

        le = OrdinalEncoder()

        le.fit(df_train[col].sort_values())    

        df_train[col] = le.fit_transform(df_train[col])    

        df_test[col] = le.transform(df_test[col])    

        print(le.mapping)



# numpyとして保存

X_train_new = df_train.loc[:,new_col].values.astype(float)

X_test_new = df_test.loc[:,new_col].values.astype(float)

X_final_train = np.hstack([X_final_train, X_train_new])

X_final_test = np.hstack([X_final_test, X_test_new])
# メモリ確保

del df_train,df_test,X_train_new,X_test_new,tmp

gc.collect()
# 下準備

isTest_train = np.zeros(X_final_train.shape[0]).reshape(X_final_train.shape[0],-1)

isTest_test = np.ones(X_final_test.shape[0]).reshape(X_final_test.shape[0],-1)



y = np.vstack([isTest_train,isTest_test])

y = y.flatten()

X = np.vstack([X_final_train,X_final_test])

del isTest_test,isTest_train



leak_name = ["title","issue_d","issue_year","issue_month","earliest_cr_line_year","earliest_cr_line_month","issue_earliest_cr_line_diff"]



leak_col = []

for leak in leak_name:

    leak_col.extend([s for s in f_col if leak in s])

leak_col = list(set(leak_col))



inc_col = []

inc_col_idx = []

for col in f_col :

    if col not in leak_col:

        inc_col_idx.append(f_col_num[col])

        inc_col.append(col)

        

X = X[:,inc_col_idx]
skf=StratifiedKFold(n_splits=5, shuffle=True,random_state=SEED)



y_oof = np.zeros(len(y))

feature_imortance = np.zeros(len(inc_col))



params = {

    'objective': 'binary',

    'metric':"auc",

    'n_estimators': 2000

}



for train_ix, test_ix in tqdm(skf.split(X, y)):

    X_train, y_train = X[train_ix], y[train_ix]

    X_val, y_val = X[test_ix], y[test_ix]



    clf =  lgb.LGBMClassifier(**params)

    clf.fit(X_train, y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=200, verbose=200)

    

    y_oof[test_ix] = clf.predict_proba(X_val)[:,1]

    feature_imortance += clf.feature_importances_/5

    del X_train, y_train,X_val, y_val



feature_imp = pd.DataFrame(feature_imortance,index = inc_col,columns =["feature_imortance"]).sort_values("feature_imortance")

fig, ax = plt.subplots(1, 1, figsize=(12, 12))

feature_imp[-50:].plot(kind="barh",ax=ax,color="b")

    

score = pd.DataFrame(y_oof[:X_final_train.shape[0]],columns=["Score"])

score.to_csv("Av_Score.csv")
# 下位20%を除去

av_idx = score[score["Score"]>0.2].index.tolist()
# #メモリの表示

# print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))

# print(" ------------------------------------ ")

# for var_name in dir():

#     if not var_name.startswith("_") and sys.getsizeof(eval(var_name)) > 10000: #ここだけアレンジ

#         print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))
# メモリ確保

del X,y,feature_imp,idx,y_oof,test_ix,train_ix

gc.collect()
# オリジナルを読み込み

df_train = df_train_base.copy()

df_test = df_test_base.copy()
# 申告年収に対する交互作用

df_train["loan_amnt_div_annual_inc"]=df_train["loan_amnt"]/(df_train["annual_inc"]+1e6)

df_train["installment_div_annual_inc"]=df_train["installment"]/(df_train["annual_inc"]+1e6)

df_train["tot_cur_bal_div_annual_inc"]=df_train["tot_cur_bal"]/(df_train["annual_inc"]+1e6)

df_train["revol_bal_div_annual_inc"]=df_train["revol_bal"]/(df_train["annual_inc"]+1e6)

df_train["tot_coll_amt_div_annual_inc"]=df_train["tot_coll_amt"]/(df_train["annual_inc"]+1e6)



# 回収率

df_train["tot_coll_amt_div_tot_cur_bal"]=df_train["tot_coll_amt"]/(df_train["tot_cur_bal"]+1e6)

df_train["tot_coll_amt_div_loan_amnt"]=df_train["tot_coll_amt"]/(df_train["loan_amnt"]+1e6)

df_train["tot_cur_bal_div_loan_amnt"]=df_train["tot_cur_bal"]/(df_train["loan_amnt"]+1e6)



# ローン額/口座開設数

df_train["loan_amnt_div_open_acc"]=df_train["loan_amnt"]/(df_train["open_acc"]+1e6)

df_train["installment_div_open_acc"]=df_train["installment"]/(df_train["open_acc"]+1e6)

df_train["tot_cur_bal_div_open_acc"]=df_train["tot_cur_bal"]/(df_train["open_acc"]+1e6)



# リボ払いの交互作用

df_train["revol_bal_div_revol_util"]=df_train["revol_bal"]/(df_train["revol_util"]+1e6)

df_train["revol_bal_div_tot_cur_bal"]=df_train["revol_bal"]/(df_train["tot_cur_bal"]+1e6)

df_train["revol_bal_div_installment"]=df_train["revol_bal"]/(df_train["installment"]+1e6)

df_train["revol_bal_div_revol_util"]=df_train["installment"]*df_train["revol_util"]
# 申告年収に対する交互作用

df_test["loan_amnt_div_annual_inc"]=df_test["loan_amnt"]/(df_test["annual_inc"]+1e6)

df_test["installment_div_annual_inc"]=df_test["installment"]/(df_test["annual_inc"]+1e6)

df_test["tot_cur_bal_div_annual_inc"]=df_test["tot_cur_bal"]/(df_test["annual_inc"]+1e6)

df_test["revol_bal_div_annual_inc"]=df_test["revol_bal"]/(df_test["annual_inc"]+1e6)

df_test["tot_coll_amt_div_annual_inc"]=df_test["tot_coll_amt"]/(df_test["annual_inc"]+1e6)



# 回収率

df_test["tot_coll_amt_div_tot_cur_bal"]=df_test["tot_coll_amt"]/(df_test["tot_cur_bal"]+1e6)

df_test["tot_coll_amt_div_loan_amnt"]=df_test["tot_coll_amt"]/(df_test["loan_amnt"]+1e6)

df_test["tot_cur_bal_div_loan_amnt"]=df_test["tot_cur_bal"]/(df_test["loan_amnt"]+1e6)



# ローン額/口座開設数

df_test["loan_amnt_div_open_acc"]=df_test["loan_amnt"]/(df_test["open_acc"]+1e6)

df_test["installment_div_open_acc"]=df_test["installment"]/(df_test["open_acc"]+1e6)

df_test["tot_cur_bal_div_open_acc"]=df_test["tot_cur_bal"]/(df_test["open_acc"]+1e6)



# リボ払いの交互作用

df_test["revol_bal_div_revol_util"]=df_test["revol_bal"]/(df_test["revol_util"]+1e6)

df_test["revol_bal_div_tot_cur_bal"]=df_test["revol_bal"]/(df_test["tot_cur_bal"]+1e6)

df_test["revol_bal_div_installment"]=df_test["revol_bal"]/(df_test["installment"]+1e6)

df_test["revol_bal_div_revol_util"]=df_test["installment"]*df_test["revol_util"]
# 最新日との差分

df_train["issue_d_diff"] = (df_test["issue_d"].max()-df_train["issue_d"]).astype(int)

df_test["issue_d_diff"] = (df_test["issue_d"].max()-df_test["issue_d"]).astype(int)



# earliest_cr_lineとの差分

df_train["issue_earliest_cr_line_diff"] = (df_train["earliest_cr_line"]-df_train["issue_d"]).astype(int)

df_test["issue_earliest_cr_line_diff"] = (df_test["earliest_cr_line"]-df_test["issue_d"]).astype(int)
# 集計用にTrain/Testをマージ

df_merge = pd.concat([df_train.drop(TARGET,axis=1),df_test])



# グレード毎の集計、サブグレード毎の集計、家持ちごとの集計

parents_col = ["grade","sub_grade","home_ownership","purpose","zip_code","addr_state","state_zip_code"]

agg_cat = ["annual_inc","installment","tot_cur_bal","revol_bal","open_acc","tot_coll_amt","dti"]



for pcol in tqdm(parents_col):

    df_agg = df_merge.groupby(pcol,as_index=False)["issue_d"].count().rename(columns={"issue_d":f"{pcol}_count"})

    for col in agg_cat:

        df_agg = pd.merge(df_agg,df_merge.groupby(pcol,as_index=False)[col].max().rename(columns={col:f"{pcol}_{col}_max"}),on = pcol,how="left")

        df_agg = pd.merge(df_agg,df_merge.groupby(pcol,as_index=False)[col].min().rename(columns={col:f"{pcol}_{col}_min"}),on = pcol,how="left")

        df_agg = pd.merge(df_agg,df_merge.groupby(pcol,as_index=False)[col].mean().rename(columns={col:f"{pcol}_{col}_mean"}),on = pcol,how="left")

        df_agg = pd.merge(df_agg,df_merge.groupby(pcol)[col].std().reset_index().rename(columns={col:f"{pcol}_{col}_std"}),on = pcol,how="left")

    df_train = pd.merge(df_train,df_agg,on=pcol,how="left")

    df_test = pd.merge(df_test,df_agg,on=pcol,how="left")

    del df_agg

del df_merge

gc.collect()



# 平均との差分とDIV

# グレード毎の集計、サブグレード毎の集計、家持ちごとの集計

parents_col = ["grade","sub_grade","home_ownership","purpose","zip_code","addr_state","state_zip_code"]

agg_cat = ["annual_inc","installment","tot_cur_bal","revol_bal","open_acc","tot_coll_amt","dti"]



for pcol in tqdm(parents_col):

    for col in agg_cat:

        df_train[f"{pcol}_{col}_diff"] = df_train[col] - df_train[f"{pcol}_{col}_mean"]

        df_train[f"{pcol}_{col}_div"] = df_train[col] /( df_train[f"{pcol}_{col}_mean"]+1e6)

        df_test[f"{pcol}_{col}_diff"] = df_test[col] - df_test[f"{pcol}_{col}_mean"]

        df_test[f"{pcol}_{col}_div"] = df_test[col] /( df_test[f"{pcol}_{col}_mean"]+1e6)



new_col = ['loan_amnt_div_annual_inc',

 'installment_div_annual_inc',

 'tot_cur_bal_div_annual_inc',

 'revol_bal_div_annual_inc',

 'tot_coll_amt_div_annual_inc',

 'tot_coll_amt_div_tot_cur_bal',

 'tot_coll_amt_div_loan_amnt',

 'tot_cur_bal_div_loan_amnt',

 'loan_amnt_div_open_acc',

 'installment_div_open_acc',

 'tot_cur_bal_div_open_acc',

 'revol_bal_div_revol_util',

 'revol_bal_div_tot_cur_bal',

 'revol_bal_div_installment',

 'issue_earliest_cr_line_diff',

 'grade_count',

 'grade_annual_inc_max',

 'grade_annual_inc_mean',

 'grade_annual_inc_std',

 'grade_installment_max',

 'grade_installment_min',

 'grade_installment_mean',

 'grade_installment_std',

 'grade_tot_cur_bal_max',

 'grade_tot_cur_bal_mean',

 'grade_tot_cur_bal_std',

 'grade_revol_bal_max',

 'grade_revol_bal_mean',

 'grade_revol_bal_std',

 'grade_open_acc_max',

 'grade_open_acc_min',

 'grade_open_acc_mean',

 'grade_open_acc_std',

 'grade_tot_coll_amt_max',

 'grade_tot_coll_amt_mean',

 'grade_tot_coll_amt_std',

 'grade_dti_max',

 'grade_dti_min',

 'grade_dti_mean',

 'grade_dti_std',

 'sub_grade_count',

 'sub_grade_annual_inc_max',

 'sub_grade_annual_inc_min',

 'sub_grade_annual_inc_mean',

 'sub_grade_annual_inc_std',

 'sub_grade_installment_max',

 'sub_grade_installment_min',

 'sub_grade_installment_mean',

 'sub_grade_installment_std',

 'sub_grade_tot_cur_bal_max',

 'sub_grade_tot_cur_bal_min',

 'sub_grade_tot_cur_bal_mean',

 'sub_grade_tot_cur_bal_std',

 'sub_grade_revol_bal_max',

 'sub_grade_revol_bal_mean',

 'sub_grade_revol_bal_std',

 'sub_grade_open_acc_max',

 'sub_grade_open_acc_min',

 'sub_grade_open_acc_mean',

 'sub_grade_open_acc_std',

 'sub_grade_tot_coll_amt_max',

 'sub_grade_tot_coll_amt_mean',

 'sub_grade_tot_coll_amt_std',

 'sub_grade_dti_max',

 'sub_grade_dti_min',

 'sub_grade_dti_mean',

 'sub_grade_dti_std',

 'home_ownership_count',

 'home_ownership_annual_inc_max',

 'home_ownership_annual_inc_mean',

 'home_ownership_annual_inc_std',

 'home_ownership_installment_max',

 'home_ownership_installment_min',

 'home_ownership_installment_mean',

 'home_ownership_installment_std',

 'home_ownership_tot_cur_bal_max',

 'home_ownership_tot_cur_bal_mean',

 'home_ownership_tot_cur_bal_std',

 'home_ownership_revol_bal_max',

 'home_ownership_revol_bal_mean',

 'home_ownership_revol_bal_std',

 'home_ownership_open_acc_max',

 'home_ownership_open_acc_min',

 'home_ownership_open_acc_mean',

 'home_ownership_open_acc_std',

 'home_ownership_tot_coll_amt_max',

 'home_ownership_tot_coll_amt_mean',

 'home_ownership_tot_coll_amt_std',

 'home_ownership_dti_max',

 'home_ownership_dti_min',

 'home_ownership_dti_mean',

 'home_ownership_dti_std',

 'purpose_count',

 'purpose_annual_inc_max',

 'purpose_annual_inc_min',

 'purpose_annual_inc_mean',

 'purpose_annual_inc_std',

 'purpose_installment_max',

 'purpose_installment_min',

 'purpose_installment_mean',

 'purpose_installment_std',

 'purpose_tot_cur_bal_max',

 'purpose_tot_cur_bal_min',

 'purpose_tot_cur_bal_mean',

 'purpose_tot_cur_bal_std',

 'purpose_revol_bal_max',

 'purpose_revol_bal_mean',

 'purpose_revol_bal_std',

 'purpose_open_acc_max',

 'purpose_open_acc_min',

 'purpose_open_acc_mean',

 'purpose_open_acc_std',

 'purpose_tot_coll_amt_max',

 'purpose_tot_coll_amt_mean',

 'purpose_tot_coll_amt_std',

 'purpose_dti_max',

 'purpose_dti_min',

 'purpose_dti_mean',

 'purpose_dti_std',

 'zip_code_count',

 'zip_code_annual_inc_max',

 'zip_code_annual_inc_min',

 'zip_code_annual_inc_mean',

 'zip_code_annual_inc_std',

 'zip_code_installment_max',

 'zip_code_installment_min',

 'zip_code_installment_mean',

 'zip_code_installment_std',

 'zip_code_tot_cur_bal_max',

 'zip_code_tot_cur_bal_min',

 'zip_code_tot_cur_bal_mean',

 'zip_code_tot_cur_bal_std',

 'zip_code_revol_bal_max',

 'zip_code_revol_bal_min',

 'zip_code_revol_bal_mean',

 'zip_code_revol_bal_std',

 'zip_code_open_acc_max',

 'zip_code_open_acc_min',

 'zip_code_open_acc_mean',

 'zip_code_open_acc_std',

 'zip_code_tot_coll_amt_max',

 'zip_code_tot_coll_amt_mean',

 'zip_code_tot_coll_amt_std',

 'zip_code_dti_max',

 'zip_code_dti_min',

 'zip_code_dti_mean',

 'zip_code_dti_std',

 'addr_state_count',

 'addr_state_annual_inc_max',

 'addr_state_annual_inc_min',

 'addr_state_annual_inc_mean',

 'addr_state_annual_inc_std',

 'addr_state_installment_max',

 'addr_state_installment_min',

 'addr_state_installment_mean',

 'addr_state_installment_std',

 'addr_state_tot_cur_bal_max',

 'addr_state_tot_cur_bal_min',

 'addr_state_tot_cur_bal_mean',

 'addr_state_tot_cur_bal_std',

 'addr_state_revol_bal_max',

 'addr_state_revol_bal_mean',

 'addr_state_revol_bal_std',

 'addr_state_open_acc_max',

 'addr_state_open_acc_min',

 'addr_state_open_acc_mean',

 'addr_state_open_acc_std',

 'addr_state_tot_coll_amt_max',

 'addr_state_tot_coll_amt_mean',

 'addr_state_tot_coll_amt_std',

 'addr_state_dti_max',

 'addr_state_dti_min',

 'addr_state_dti_mean',

 'addr_state_dti_std',

 'state_zip_code_count',

 'state_zip_code_annual_inc_max',

 'state_zip_code_annual_inc_min',

 'state_zip_code_annual_inc_mean',

 'state_zip_code_annual_inc_std',

 'state_zip_code_installment_max',

 'state_zip_code_installment_min',

 'state_zip_code_installment_mean',

 'state_zip_code_installment_std',

 'state_zip_code_tot_cur_bal_max',

 'state_zip_code_tot_cur_bal_min',

 'state_zip_code_tot_cur_bal_mean',

 'state_zip_code_tot_cur_bal_std',

 'state_zip_code_revol_bal_max',

 'state_zip_code_revol_bal_min',

 'state_zip_code_revol_bal_mean',

 'state_zip_code_revol_bal_std',

 'state_zip_code_open_acc_max',

 'state_zip_code_open_acc_min',

 'state_zip_code_open_acc_mean',

 'state_zip_code_open_acc_std',

 'state_zip_code_tot_coll_amt_max',

 'state_zip_code_tot_coll_amt_mean',

 'state_zip_code_tot_coll_amt_std',

 'state_zip_code_dti_max',

 'state_zip_code_dti_min',

 'state_zip_code_dti_mean',

 'state_zip_code_dti_std',

 'grade_annual_inc_diff',

 'grade_annual_inc_div',

 'grade_installment_diff',

 'grade_installment_div',

 'grade_tot_cur_bal_diff',

 'grade_tot_cur_bal_div',

 'grade_revol_bal_diff',

 'grade_revol_bal_div',

 'grade_open_acc_diff',

 'grade_open_acc_div',

 'grade_tot_coll_amt_diff',

 'grade_tot_coll_amt_div',

 'grade_dti_diff',

 'grade_dti_div',

 'sub_grade_annual_inc_diff',

 'sub_grade_annual_inc_div',

 'sub_grade_installment_diff',

 'sub_grade_installment_div',

 'sub_grade_tot_cur_bal_diff',

 'sub_grade_tot_cur_bal_div',

 'sub_grade_revol_bal_diff',

 'sub_grade_revol_bal_div',

 'sub_grade_open_acc_diff',

 'sub_grade_open_acc_div',

 'sub_grade_tot_coll_amt_diff',

 'sub_grade_tot_coll_amt_div',

 'sub_grade_dti_diff',

 'sub_grade_dti_div',

 'home_ownership_annual_inc_diff',

 'home_ownership_annual_inc_div',

 'home_ownership_installment_diff',

 'home_ownership_installment_div',

 'home_ownership_tot_cur_bal_diff',

 'home_ownership_tot_cur_bal_div',

 'home_ownership_revol_bal_diff',

 'home_ownership_revol_bal_div',

 'home_ownership_open_acc_diff',

 'home_ownership_open_acc_div',

 'home_ownership_tot_coll_amt_diff',

 'home_ownership_tot_coll_amt_div',

 'home_ownership_dti_diff',

 'home_ownership_dti_div',

 'purpose_annual_inc_diff',

 'purpose_annual_inc_div',

 'purpose_installment_diff',

 'purpose_installment_div',

 'purpose_tot_cur_bal_diff',

 'purpose_tot_cur_bal_div',

 'purpose_revol_bal_diff',

 'purpose_revol_bal_div',

 'purpose_open_acc_diff',

 'purpose_open_acc_div',

 'purpose_tot_coll_amt_diff',

 'purpose_tot_coll_amt_div',

 'purpose_dti_diff',

 'purpose_dti_div',

 'zip_code_annual_inc_diff',

 'zip_code_annual_inc_div',

 'zip_code_installment_diff',

 'zip_code_installment_div',

 'zip_code_tot_cur_bal_diff',

 'zip_code_tot_cur_bal_div',

 'zip_code_revol_bal_diff',

 'zip_code_revol_bal_div',

 'zip_code_open_acc_diff',

 'zip_code_open_acc_div',

 'zip_code_tot_coll_amt_diff',

 'zip_code_tot_coll_amt_div',

 'zip_code_dti_diff',

 'zip_code_dti_div',

 'addr_state_annual_inc_diff',

 'addr_state_annual_inc_div',

 'addr_state_installment_diff',

 'addr_state_installment_div',

 'addr_state_tot_cur_bal_diff',

 'addr_state_tot_cur_bal_div',

 'addr_state_revol_bal_diff',

 'addr_state_revol_bal_div',

 'addr_state_open_acc_diff',

 'addr_state_open_acc_div',

 'addr_state_tot_coll_amt_diff',

 'addr_state_tot_coll_amt_div',

 'addr_state_dti_diff',

 'addr_state_dti_div',

 'state_zip_code_annual_inc_diff',

 'state_zip_code_annual_inc_div',

 'state_zip_code_installment_diff',

 'state_zip_code_installment_div',

 'state_zip_code_tot_cur_bal_diff',

 'state_zip_code_tot_cur_bal_div',

 'state_zip_code_revol_bal_diff',

 'state_zip_code_revol_bal_div',

 'state_zip_code_open_acc_diff',

 'state_zip_code_open_acc_div',

 'state_zip_code_tot_coll_amt_diff',

 'state_zip_code_tot_coll_amt_div',

 'state_zip_code_dti_diff',

 'state_zip_code_dti_div']



# 0.501以上のもののみを利用    

f_col.extend(new_col)

# del_candidate = [s for s in base_cols if s not in f_col]

# del_col.extend(del_candidate)



# カラム番号を保存

f_col_num = {}

for idx, col in enumerate(f_col):

    f_col_num[col] = idx



# カテゴリは変換しておく

for col in new_col:

    if col in cat:

        le = OrdinalEncoder()

        le.fit(df_train[col].sort_values())    

        df_train[col] = le.fit_transform(df_train[col])    

        df_test[col] = le.transform(df_test[col])    

        print(le.mapping)



# numpyとして保存

X_train_new = df_train.loc[:,new_col].values.astype(float)

X_test_new = df_test.loc[:,new_col].values.astype(float)

X_final_train = np.hstack([X_final_train, X_train_new])

X_final_test = np.hstack([X_final_test, X_test_new])



# メモリ確保

del df_train,df_test,X_test_new,X_train_new

gc.collect()
# 下準備

leak_name = ["annual_inc","loan_amnt","installment","dti","revol_util","tot_coll_amt","tot_cur_bal","revol_bal"]



leak_col = []

for leak in leak_name:

    leak_col.extend([s for s in f_col if leak in s])

leak_col = list(set(leak_col))



inc_col = []

inc_col_idx = []

for col in f_col :

    if col not in leak_col:

        inc_col_idx.append(f_col_num[col])

        inc_col.append(col)

        

TARGET_inc = "loan_amnt"

TARGET_inc_idx = f_col_num[TARGET_inc]
X_base = np.vstack([X_final_train,X_final_test])

X = X_base[:,inc_col_idx]

y = X_base[:,TARGET_inc_idx]

del X_base

y = y.flatten()



kf=KFold(n_splits=5, shuffle=True,random_state=SEED)

y_oof = np.zeros(y.shape[0])



feature_imp = np.zeros(len(inc_col))



params = {

    'n_estimators': 2000

}



for train_ix, test_ix in tqdm(kf.split(X, y)):

    X_train, y_train = X[train_ix], y[train_ix]

    X_val, y_val = X[test_ix], y[test_ix]



    clf =  lgb.LGBMRegressor(**params)

    clf.fit(X_train, y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=200, verbose=200)



    y_oof[test_ix] += clf.predict(X_val)

    feature_imp += clf.feature_importances_/5

    del X_train, y_train,X_val, y_val,test_ix,train_ix

del X,y



feature_imp = pd.DataFrame(feature_imp,index = inc_col,columns =["feature_imortance"]).sort_values("feature_imortance")

fig, ax = plt.subplots(1, 1, figsize=(12, 12))

feature_imp[-50:].plot(kind="barh",ax=ax,color="b")
np.savetxt('loam_amt_pred.txt', y_oof)

# y_oof = np.loadtxt('../input/subset/loam_amt_pred.txt')
y_oof_train = y_oof[0:X_final_train.shape[0]]

y_oof_test = y_oof[X_final_train.shape[0]:]

y_oof_train = y_oof_train.reshape(len(y_oof_train),-1)

y_oof_test =y_oof_test.reshape(len(y_oof_test),-1)
X_final_train = np.hstack([X_final_train,y_oof_train])

X_final_test = np.hstack([X_final_test,y_oof_test])

f_col.append(f"pred_{TARGET_inc}")



# カラム番号を保存

f_col_num = {}

for idx, col in enumerate(f_col):

    f_col_num[col] = idx
# #メモリの表示

# print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))

# print(" ------------------------------------ ")

# for var_name in dir():

#     if not var_name.startswith("_") and sys.getsizeof(eval(var_name)) > 10000: 

#         print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))
del feature_imp,y_oof
# オリジナルを読み込み

df_train = df_train_base.copy()

df_test = df_test_base.copy()
# revol_utilをBIN化

df_train["revol_util_bin"] = 0

df_train.loc[df_train["revol_util"].isna(),"revol_util_bin"] = 0

df_train.loc[(df_train["revol_util"]>0)&(df_train["revol_util"]<=10),"revol_util_bin"] = 10

df_train.loc[(df_train["revol_util"]>10)&(df_train["revol_util"]<=20),"revol_util_bin"] = 20

df_train.loc[(df_train["revol_util"]>20)&(df_train["revol_util"]<=30),"revol_util_bin"] = 30

df_train.loc[(df_train["revol_util"]>30)&(df_train["revol_util"]<=40),"revol_util_bin"] = 40

df_train.loc[(df_train["revol_util"]>40)&(df_train["revol_util"]<=50),"revol_util_bin"] = 50

df_train.loc[(df_train["revol_util"]>50)&(df_train["revol_util"]<=60),"revol_util_bin"] = 60

df_train.loc[(df_train["revol_util"]>60)&(df_train["revol_util"]<=70),"revol_util_bin"] = 70

df_train.loc[(df_train["revol_util"]>70)&(df_train["revol_util"]<=80),"revol_util_bin"] = 80

df_train.loc[(df_train["revol_util"]>80)&(df_train["revol_util"]<=90),"revol_util_bin"] = 90

df_train.loc[(df_train["revol_util"]>90)&(df_train["revol_util"]<=100),"revol_util_bin"] = 100

df_train.loc[(df_train["revol_util"]>100),"revol_util_bin"] = 200



# revol_utilをBIN化

df_test["revol_util_bin"] = 0

df_test.loc[df_test["revol_util"].isna(),"revol_util_bin"] = 0

df_test.loc[(df_test["revol_util"]>0)&(df_test["revol_util"]<=10),"revol_util_bin"] = 10

df_test.loc[(df_test["revol_util"]>10)&(df_test["revol_util"]<=20),"revol_util_bin"] = 20

df_test.loc[(df_test["revol_util"]>20)&(df_test["revol_util"]<=30),"revol_util_bin"] = 30

df_test.loc[(df_test["revol_util"]>30)&(df_test["revol_util"]<=40),"revol_util_bin"] = 40

df_test.loc[(df_test["revol_util"]>40)&(df_test["revol_util"]<=50),"revol_util_bin"] = 50

df_test.loc[(df_test["revol_util"]>50)&(df_test["revol_util"]<=60),"revol_util_bin"] = 60

df_test.loc[(df_test["revol_util"]>60)&(df_test["revol_util"]<=70),"revol_util_bin"] = 70

df_test.loc[(df_test["revol_util"]>70)&(df_test["revol_util"]<=80),"revol_util_bin"] = 80

df_test.loc[(df_test["revol_util"]>80)&(df_test["revol_util"]<=90),"revol_util_bin"] = 90

df_test.loc[(df_test["revol_util"]>90)&(df_test["revol_util"]<=100),"revol_util_bin"] = 100

df_test.loc[(df_test["revol_util"]>100),"revol_util_bin"] = 200
# dtiを5毎でカテゴリ分け

df_train["dti_cat"] = df_train["dti"]

df_train.loc[df_train["dti_cat"]<0,"dti_cat"] = 0

df_train.loc[(df_train["dti_cat"]>=0) & (df_train["dti_cat"]<5 ),"dti_cat"]= 0

df_train.loc[(df_train["dti_cat"]>=5) & (df_train["dti_cat"]<10 ),"dti_cat"] = 5

df_train.loc[(df_train["dti_cat"]>=10) & (df_train["dti_cat"]<15 ),"dti_cat"] = 10

df_train.loc[(df_train["dti_cat"]>=15) & (df_train["dti_cat"]<20 ),"dti_cat"]= 15

df_train.loc[(df_train["dti_cat"]>=20) & (df_train["dti_cat"]<25 ),"dti_cat"] = 20

df_train.loc[(df_train["dti_cat"]>=25) & (df_train["dti_cat"]<30 ),"dti_cat"] = 25

df_train.loc[(df_train["dti_cat"]>=30) & (df_train["dti_cat"]<35 ),"dti_cat"]= 30

df_train.loc[(df_train["dti_cat"]>=35) & (df_train["dti_cat"]<40 ),"dti_cat"] = 35

df_train.loc[(df_train["dti_cat"]>40 ),"dti_cat"]= 40



df_test["dti_cat"] = df_test["dti"]

df_test.loc[df_test["dti_cat"]<0,"dti_cat"] = 0

df_test.loc[(df_test["dti_cat"]>=0) & (df_test["dti_cat"]<5 ),"dti_cat"]= 0

df_test.loc[(df_test["dti_cat"]>=5) & (df_test["dti_cat"]<10 ),"dti_cat"] = 5

df_test.loc[(df_test["dti_cat"]>=10) & (df_test["dti_cat"]<15 ),"dti_cat"] = 10

df_test.loc[(df_test["dti_cat"]>=15) & (df_test["dti_cat"]<20 ),"dti_cat"]= 15

df_test.loc[(df_test["dti_cat"]>=20) & (df_test["dti_cat"]<25 ),"dti_cat"] = 20

df_test.loc[(df_test["dti_cat"]>=25) & (df_test["dti_cat"]<30 ),"dti_cat"] = 25

df_test.loc[(df_test["dti_cat"]>=30) & (df_test["dti_cat"]<35 ),"dti_cat"]= 30

df_test.loc[(df_test["dti_cat"]>=35) & (df_test["dti_cat"]<40 ),"dti_cat"] = 35

df_test.loc[(df_test["dti_cat"]>40 ),"dti_cat"]= 40
# 1年毎にターゲットエンコーディング、初年度はNULL

t_cols =  ["sub_grade","grade","inq_last_6mths","addr_state","state_zip_code","dti_cat","revol_util_bin"] 

for col in t_cols:

    te_year = df_train.groupby(["issue_year",col],as_index=False)[TARGET].mean().rename(columns={TARGET:f"te_{col}"})

    te_year["issue_year"] += 1

    df_train = pd.merge(df_train,te_year,on=["issue_year",col],how="left")

    df_test = pd.merge(df_test,te_year,on=["issue_year",col],how="left")
new_col = [ 'revol_util_bin',

 'dti_cat',

 'te_sub_grade',

 'te_grade',

 'te_inq_last_6mths',

 'te_addr_state',

 'te_state_zip_code',

 'te_dti_cat',

 'te_revol_util_bin']
# 0.501以上のもののみを利用    

f_col.extend(new_col)

# del_candidate = [s for s in base_cols if s not in f_col]

# del_col.extend(del_candidate)
# カラム番号を保存

f_col_num = {}

for idx, col in enumerate(f_col):

    f_col_num[col] = idx
# numpyとして保存

X_train_new = df_train.loc[:,new_col].values.astype(float)

X_test_new = df_test.loc[:,new_col].values.astype(float)

X_final_train = np.hstack([X_final_train, X_train_new])

X_final_test = np.hstack([X_final_test, X_test_new])
# メモリ確保

del df_train,df_test,X_train_new,X_test_new

gc.collect()
# #メモリの表示

# print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))

# print(" ------------------------------------ ")

# for var_name in dir():

#     if not var_name.startswith("_") and sys.getsizeof(eval(var_name)) > 10000: #ここだけアレンジ

#         print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))
# オリジナルを読み込み

df_train = df_train_base.copy()

df_test = df_test_base.copy()



len_df_train = len(df_train)



# テーブルマージ

df_merge = pd.concat([df_train.drop(TARGET,axis=1),df_test])



# 対象カラム

col = "emp_title"



# テキストとそれ以外に分割

TXT = df_merge[col].copy()

del df_merge,df_train,df_test
many_words = ['design',

 'creative',

 'respiratory',

 'agency',

 'svp',

 'kaiser',

 'teller',

 'direct',

 'developer',

 'cna',

 'rehab',

 'regional',

 'fire',

 'lieutenant',

 'cpa',

 'solutions',

 'agent',

 'owner',

 'buyer',

 'investigator',

 'pharmacist',

 'home',

 'company',

 'operations',

 'power',

 'administration',

 'healthcare',

 'information',

 'network',

 'laborer',

 'man',

 'psychologist',

 'water',

 'patient',

 'collector',

 'inside',

 'asst',

 'budget',

 'education',

 'welder',

 'support',

 'writer',

 'applications',

 'escrow',

 'csr',

 'vice',

 'scientist',

 'music',

 'cashier',

 'technical',

 'capital',

 'producer',

 'cook',

 'mechanic',

 'global',

 'president',

 'planner',

 'research',

 'gas',

 'accountant',

 'morgan',

 'store',

 'programmer',

 'clinic',

 'dispatcher',

 'international',

 'public',

 'registered',

 'pilot',

 'auto',

 'firefighter',

 'insurance',

 'executive',

 'superintendent',

 'development',

 'physical',

 'practitioner',

 'partner',

 'certified',

 'trust',

 'office',

 'dept',

 'coordinator',

 'counsel',

 'carrier',

 'texas',

 'corrections',

 'account',

 'controller',

 'federal',

 'co',

 'san',

 'consulting',

 'isd',

 'corporation',

 'business',

 'program',

 'national',

 'cfo',

 'dealer',

 'server',

 'group',

 'corporate',

 'principal',

 'police',

 'technologies',

 'church',

 'social',

 'chief',

 'accounting',

 'aide',

 'officer',

 'district',

 'state',

 'engineering',

 'hospital',

 'college',

 'llp',

 'services',

 'product',

 'security',

 'bank',

 'operator',

 'it',

 'llc',

 'customer',

 'professor',

 'supervisor',

 'department',

 'school',

 'architect',

 'teacher',

 'management',

 'center',

 'lead',

 'assistant',

 'pastor',

 'engineer',

 'and',

 'physician',

 'attorney',

 'project',

 'associate',

 'county',

 'driver',

 'sr',

 'vp',

 'specialist',

 'administrator',

 'clerk',

 'new',

 'medical',

 'service',

 'university',

 'software',

 'tech',

 'care',

 'city',

 'health',

 'sales',

 'analyst',

 'financial',

 'systems',

 'director',

 'senior',

 'inc',

 'manager']
# テキストの出現数をカウント

cv = CountVectorizer()

cv.fit(many_words)

TXT_cv = cv.transform(TXT.fillna('#'))



cols_cv = cv.get_feature_names()

for idx,_ in enumerate(cols_cv):

    cols_cv[idx] = col+"_"+cols_cv[idx]+"_cv"



TXT_cv_train = TXT_cv[0:len_df_train,:]

TXT_cv_test = TXT_cv[len_df_train:,:]
TXT_cv_train = TXT_cv_train.toarray()

TXT_cv_test = TXT_cv_test.toarray()
X_final_train = np.hstack([X_final_train,TXT_cv_train])

X_final_test = np.hstack([X_final_test,TXT_cv_test])
f_col.extend(cols_cv)



# カラム番号を保存

f_col_num = {}

for idx, col in enumerate(f_col):

    f_col_num[col] = idx



# #メモリの表示

# print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))

# print(" ------------------------------------ ")

# for var_name in dir():

#     if not var_name.startswith("_") and sys.getsizeof(eval(var_name)) > 10000: #ここだけアレンジ

#         print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))



del TXT,TXT_cv,TXT_cv_train,TXT_cv_test
# オリジナルを読み込み

df_train = df_train_base.copy()

df_test = df_test_base.copy()



len_df_train = len(df_train)



# テーブルマージ

df_merge = pd.concat([df_train.drop(TARGET,axis=1),df_test])



# テーブルマージ

col = "title"



# テキストとそれ以外に分割

TXT = df_merge[col].copy()

del df_merge,df_train,df_test
many_words = ['lower',

 'down',

 'help',

 'high',

 'on',

 'get',

 'car',

 'payoff',

 'new',

 'refinance',

 'free',

 'bills',

 'improvement',

 'rate',

 'cc',

 'consolidate',

 'home',

 'interest',

 'cards',

 'off',

 'pay',

 'card',

 'and',

 'my',

 'for',

 'to',

 'consolidation',

 'credit',

 'debt',

 'loan']
# テキストの出現数をカウント

cv = CountVectorizer()

cv.fit(many_words)

TXT_cv = cv.transform(TXT.fillna('#'))



cols_cv = cv.get_feature_names()

for idx,_ in enumerate(cols_cv):

    cols_cv[idx] = col+"_"+cols_cv[idx]+"_cv"



TXT_cv_train = TXT_cv[0:len_df_train,:]

TXT_cv_test = TXT_cv[len_df_train:,:]



TXT_cv_train = TXT_cv_train.toarray()

TXT_cv_test = TXT_cv_test.toarray()



X_final_train = np.hstack([X_final_train,TXT_cv_train])

X_final_test = np.hstack([X_final_test,TXT_cv_test])



f_col.extend(cols_cv)



# カラム番号を保存

f_col_num = {}

for idx, col in enumerate(f_col):

    f_col_num[col] = idx

    

del TXT,TXT_cv,TXT_cv_train,TXT_cv_test
X_final_train = X_final_train[av_idx,:]

y_final_train = y_final_train[av_idx]

df_train_base = df_train_base.iloc[av_idx,:].reset_index()
# 検証用のインデックスの作成

train_idx = df_train_base[(df_train_base.issue_year<2015) | (df_train_base.issue_month < 7)].index.tolist()

val_idx = df_train_base[(df_train_base.issue_year == 2015) & (df_train_base.issue_month>=7)].index.tolist()



len_train = len(train_idx)

idx_all   = np.random.choice(train_idx, len_train, replace=False)

train_idx_1 = idx_all[0:round(len_train/3)]

train_idx_2 = idx_all[round(len_train/3):round(len_train/3)*2]

train_idx_3 = idx_all[round(len_train/3)*2:]



len_val = len(val_idx)

idx_all   = np.random.choice(val_idx, len_val, replace=False)

val_idx_1 = idx_all[0:round(len_val/3)]

val_idx_2 = idx_all[round(len_val/3):round(len_val/3)*2]

val_idx_3 = idx_all[round(len_val/3)*2:]
def objective(trial):

        

    feature_fraction = trial.suggest_uniform('feature_fraction', 0, 1.0)

    learning_rate = trial.suggest_uniform('learning_rate', 0, 1.0)

#     subsample = trial.suggest_uniform('subsample', 0.8, 1.0)

    num_leaves = trial.suggest_int('num_leaves', 5, 1000)

    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 10, 10000)

    reg_alpha = trial.suggest_uniform('reg_alpha', 0, 1.0)

    reg_lambda = trial.suggest_uniform('reg_lambda', 0, 1.0)



    params = {"objective": "binary",

              "boosting_type": "gbdt",

              'metric':"auc",

              "max_depth":-1,

              "learning_rate": learning_rate,

              "num_leaves": num_leaves,

              "feature_fraction": feature_fraction,

              "verbosity": 1,

              "min_split_gain": 0,

              "min_data_in_leaf": min_data_in_leaf,

              "subsample": 1,

              "reg_alpha":reg_alpha,

              "reg_lambda":reg_lambda

              }

    

    scores = []

    

    for train_idx,val_idx in [(train_idx_1,val_idx_1),(train_idx_2,val_idx_2),(train_idx_3,val_idx_3)]:    

        X_train = X_final_train[train_idx]

        X_val = X_final_train[val_idx]

        y_train = y_final_train[train_idx]

        y_val =  y_final_train[val_idx]

    

        clf =  lgb.LGBMClassifier(**params,n_estimators=2000)

        clf.fit(X_train, y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=200, verbose=200)

        score = clf.best_score_["valid_0"]["auc"]

        scores.append(score)

        del X_train,X_val,y_train,y_val

    #     AUCを最小化させる

    return (1 - sum(scores)/3)



study = optuna.create_study()

study.optimize(objective, n_trials=30)

study.best_params
params = {"objective": "binary",

          "boosting_type": "gbdt",

          'metric':"auc",

          "max_depth":-1,

          "learning_rate": study.best_params["learning_rate"],

          "num_leaves": study.best_params["num_leaves"],

          "feature_fraction": study.best_params["feature_fraction"],

          "verbosity": 1,

          "min_split_gain": 0,

          "min_data_in_leaf": study.best_params["min_data_in_leaf"],

          "subsample": 1,

          "reg_alpha":study.best_params["reg_alpha"],

          "reg_lambda":study.best_params["reg_lambda"]

          }



opt_params = study.best_params



clf =  lgb.LGBMClassifier(**params,n_estimators=2000)

clf.fit(X_final_train, y_final_train,eval_set=[(X_final_train,y_final_train)],early_stopping_rounds=200, verbose=100)



y_pred = clf.predict_proba(X_final_test)[:,1]
# submissionにマージ

submission = pd.read_csv(filepath2+'sample_submission.csv',index_col=0)

submission.loan_condition = y_pred

submission.to_csv('submission.csv')