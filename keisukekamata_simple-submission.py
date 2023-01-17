#base and visualization

import scipy as sp

import statistics 

from pandas import DataFrame, Series

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns

from datetime import datetime





# preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import quantile_transform

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import LabelEncoder 

from tqdm import tqdm_notebook as tqdm

import category_encoders as ce

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix



from sklearn.decomposition import SparsePCA

from sklearn.decomposition import TruncatedSVD



# modeling

from sklearn.tree import ExtraTreeClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.ensemble.bagging import BaggingClassifier

from sklearn.ensemble.forest import ExtraTreesClassifier

from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model.stochastic_gradient import SGDClassifier

from lightgbm import LGBMClassifier,LGBMRegressor

import lightgbm

from catboost import CatBoostClassifier, FeaturesData, Pool,CatBoostRegressor



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

import tensorflow as tf









# validation

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GroupKFold

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve

from sklearn.metrics import mean_squared_error,mean_squared_log_error



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

# gkf = GroupKFold(n_splits = 5, random_state = 71)
station_info = pd.read_csv('/kaggle/input/exam-for-students20200527/station_info.csv', index_col=0)

city_info = pd.read_csv('/kaggle/input/exam-for-students20200527/city_info.csv', index_col=0)

#df_train = pd.read_csv('/kaggle/input/exam-for-students20200527/sample_submission.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/exam-for-students20200527/test.csv', index_col=0)

df_train = pd.read_csv('/kaggle/input/exam-for-students20200527/train.csv', index_col=0)



# prepare train data

y_train = df_train.TradePrice

X_train = df_train.drop(['TradePrice'], axis = 1)

X_test  = df_test
X_train.describe()
mask = np.zeros_like(X_train.corr())

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(15, 10))

    ax = sns.heatmap(X_train.corr(), mask=mask, vmax=1, vmin=-1, square=True,linewidths=.5,xticklabels=1, yticklabels=1)
X_train.isnull().sum()
X_test.isnull().sum()
#Type



#Region



#Municipality (testが埼玉だけなので使えない) #激都会だけ 除外新宿区, 千代田区, 港区, 中央区, 渋谷区

print(X_train.shape[0])

Label = (X_train["Municipality"] != "Shinjuku Ward")&(X_train["Municipality"] != "Chiyoda Ward")&(X_train["Municipality"] != "Minato Ward")&(X_train["Municipality"] != "Chuo Ward")&(X_train["Municipality"] != "Shibuya Ward")

X_train = X_train[Label]

y_train = y_train[Label]

print(X_train.shape[0])

#TimeToNearestStation (MinTimeToNearestStationとMaxTimeToNearestStationの平均とする)

X_all = pd.concat([X_train,X_test],axis =0)

X_all['TimeToNearestStation'] = np.average([X_all['MinTimeToNearestStation'], X_all['MaxTimeToNearestStation']])

X_train = X_all.iloc[:X_train.shape[0],:]

X_test = X_all.iloc[X_train.shape[0]:,:]



#MinTimeToNearestStation



#MaxTimeToNearestStation (nullはminで置き換え)

X_all = pd.concat([X_train,X_test],axis =0)

X_all['MaxTimeToNearestStation'][X_all['MaxTimeToNearestStation'].isnull()] = X_all['MinTimeToNearestStation'][X_all['MaxTimeToNearestStation'].isnull()]

X_train = X_all.iloc[:X_train.shape[0],:]

X_test = X_all.iloc[X_train.shape[0]:,:]



#FloorPlan



#Area



#LandShape



#Frontage



#TotalFloorArea



#BuildingYear (築年数に変換)

X_all = pd.concat([X_train,X_test],axis =0)

X_all['BuildingYear'][X_all['BuildingYear'].isnull()]= 2020-X_all['BuildingYear'][X_all['BuildingYear'].isnull()]

X_train = X_all.iloc[:X_train.shape[0],:]

X_test = X_all.iloc[X_train.shape[0]:,:]



#Structure (いくつかのパターンが,で区切られている)



#Use (いくつかのパターンが,で区切られている)



#Purpose

#Direction

#Classification

#X_all = pd.concat([X_train,X_test],axis =0)

#X_all["Classification"][X_all["Classification"] =="Kyoto/ Osaka Prefectural Road"] = "Other"

#X_all["Classification"][X_all["Classification"] =="Hokkaido Prefectural Road"] = "Other"

#X_train = X_all.iloc[:X_train.shape[0],:]

#X_test = X_all.iloc[X_train.shape[0]:,:]

#Breadth

#CityPlanning



#CoverageRatio

#FloorAreaRatio

#Year

#Quarter

#Renovation (nullはNot yetとしてもいいかも)

X_all = pd.concat([X_train,X_test],axis =0)

X_all['Renovation'][X_all['Renovation'].isnull()] = "Not yet"

X_all['Renovation'][X_all['Renovation']=="Not yet"] = 0

X_all['Renovation'][X_all['Renovation']=="Done"]    = 1

X_train = X_all.iloc[:X_train.shape[0],:]

X_test = X_all.iloc[X_train.shape[0]:,:]



#Remarks (tfidf?)



#AreaIsGreaterFlag #重要ではなかったので消す

#TotalFloorAreaIsGreaterFlag #重要ではなかったので消す

#FrontageIsGreaterFlag #重要ではなかったので消す

#PrewarBuilding  #重要ではなかったので消す

#Prefecture (testが埼玉だけなので使えない)

#DistrictName　(testが埼玉だけなので使えない)

#NearestStation (testが埼玉だけなので使えない)



#TradePrice

col = "Type" #OK

print(col)

print("train")

print(X_train[col].value_counts(ascending=True))

print("test")

print(X_test[col].value_counts(ascending=True))





col = "Region" # OK

print(col)

print("train")

print(X_train[col].value_counts(ascending=True))

print("test")

print(X_test[col].value_counts(ascending=True))



col = "FloorPlan" # map対応すみ

print(col)

print("train")

print(X_train[col].value_counts(ascending=True))

print("test")

print(X_test[col].value_counts(ascending=True))



col = 'LandShape' # OK

print(col)

print("train")

print(X_train[col].value_counts(ascending=True))

print("test")

print(X_test[col].value_counts(ascending=True))



#col = 'Structure' #tfidfがいいかな?

#print(col)

#print("train")

#print(X_train[col].value_counts(ascending=True))

#print("test")

#print(X_test[col].value_counts(ascending=True))



#col = 'Use' #tfidfがいいかな?

#print(col)

#print("train")

#print(X_train[col].value_counts(ascending=True))

#print("test")

#print(X_test[col].value_counts(ascending=True))



col = 'Purpose' #OK

print(col)

print("train")

print(X_train[col].value_counts(ascending=True))

print("test")

print(X_test[col].value_counts(ascending=True))



col = 'Direction' #OK

print(col)

print("train")

print(X_train[col].value_counts(ascending=True))

print("test")

print(X_test[col].value_counts(ascending=True))



col = 'Classification' #2つまとめた

print(col)

print("train")

print(X_train[col].value_counts(ascending=True))

print("test")

print(X_test[col].value_counts(ascending=True))



#mapping



#交互作用を検討する際は0を含まないようにする

mapping_dict = {

    "FloorPlan": {

        "1K": 1,"1R": 2,"1K+S": 3,"1R+S": 4,"1DK": 5,"1DK+S": 6,"1LDK": 7,"1LDK+S": 8,

        "2K": 9,"2DK": 10,"2K+S": 11,"2DK": 12,"2LK": 13,"2LD": 14,"2DK+S": 15,"2LK+S": 16,"2LD+S": 17,"2LDK": 18,"2LDK+S": 19,

        "3K": 20,"3DK": 21,"3LK": 22,"3DK+S": 23,"3LD": 24,"3LDK": 25,"3LDK+S": 26,"3LDK+K": 27,

        "4K": 28,"4DK": 29,"4DK+S": 30,"4LDK": 31,"4LDK+K": 32,"4LDK+S": 33,

        "5DK": 34,"5LDK": 35,"5LDK+K": 36,"5LDK+S": 37,

        "6LDK": 38,"Duplex": 39,"Open Floor": 40,"Studio Apartment": 41

    }

}



X_train = X_train.replace(mapping_dict)

X_test  = X_test.replace(mapping_dict)



mapping_col = ['FloorPlan']

X_train[mapping_col] = X_train[mapping_col].fillna(-1)

X_test[mapping_col]  = X_test[mapping_col].fillna(-1)

X_train[mapping_col] = X_train[mapping_col].astype(int)

X_test[mapping_col]  = X_test[mapping_col].astype(int)


# crete derived attribute

X_all = pd.concat([X_train,X_test],axis =0)







# 交互作用

#X_all["Year_Quarter"] = X_all["Year"]*10+X_all["Quarter"]

#X_all["CityPlanning_Type"] = X_all["CityPlanning"]+X_all["Type"]

#X_all["CityPlanning_Classification"] = X_all["CityPlanning"]+X_all["Classification"]





#X_all["CityPlanning_Type"][X_all["CityPlanning_Type"] =="Outside City Planning AreaPre-owned Condominiums, etc."] = "Other"

#X_all["CityPlanning_Type"][X_all["CityPlanning_Type"] =="Non-divided City Planning AreaPre-owned Condominiums, etc."] = "Other"

#X_all["CityPlanning_Type"][X_all["CityPlanning_Type"] =="Quasi-city Planning AreaResidential Land(Land and Building)"] = "Other"

#X_all["CityPlanning_Type"][X_all["CityPlanning_Type"] =="Quasi-city Planning AreaResidential Land(Land Only)"] = "Other"

#X_all["CityPlanning_Type"][X_all["CityPlanning_Type"] =="Exclusively Industrial ZonePre-owned Condominiums, etc."] = "Other"





#X_all["CityPlanning_Classification"][X_all["CityPlanning_Classification"] =="Outside City Planning AreaPre-owned Condominiums, etc."] = "Other"

#X_all["CityPlanning_Classification"][X_all["CityPlanning_Classification"] =="Non-divided City Planning AreaPre-owned Condominiums, etc."] = "Other"

#X_all["CityPlanning_Classification"][X_all["CityPlanning_Classification"] =="Quasi-city Planning AreaResidential Land(Land and Building)"] = "Other"

#X_all["CityPlanning_Classification"][X_all["CityPlanning_Classification"] =="Quasi-city Planning AreaResidential Land(Land Only)"] = "Other"

#X_all["CityPlanning_Classification"][X_all["CityPlanning_Classification"] =="Exclusively Industrial ZonePre-owned Condominiums, etc."] = "Other"





X_train = X_all.iloc[:X_train.shape[0],:]

X_test = X_all.iloc[X_train.shape[0]:,:]
drop_col = ["Prefecture",

            "Municipality",

            "DistrictName",

            "PrewarBuilding",

            "NearestStation",

            "AreaIsGreaterFlag",

            "FrontageIsGreaterFlag",

            "TotalFloorAreaIsGreaterFlag",

            "TimeToNearestStation"] 

X_train = X_train.drop(columns=drop_col)

X_test  = X_test.drop(columns=drop_col)



# 欠損している項目の数

X_train['num_nulls'] = X_train.isnull().sum(axis=1)

X_test['num_nulls']  = X_test.isnull().sum(axis=1)

# dtypeがobjectのカラム名とユニーク数を確認してみましょう。

cats_cols  = []

num_cols = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats_cols.append(col)

    if (X_train[col].dtype == 'float64')|(X_train[col].dtype == 'int64'):

        num_cols.append(col)

        

        

cats_cols.remove('Structure')

cats_cols.remove('Use')

cats_cols.remove('Remarks')
one_hot_enc_cols = []

count_enc_cols   = []

ordinal_enc_cols = []

target_enc_cols  = cats_cols
# one hot encoding

if one_hot_enc_cols != []:

    ce_ohe = ce.OneHotEncoder(cols=one_hot_enc_cols,handle_unknown='impute')

    X_train = ce_ohe.fit_transform(X_train)

    X_test  = ce_ohe.transform(X_test)



# ordinal encoding

if ordinal_enc_cols != []:

    for col in ordinal_enc_cols:

        encoder   = OrdinalEncoder()

        X_train[col] = encoder.fit_transform(X_train[col].values)

        X_test[col]  = encoder.transform(X_test[col].values)



# count encoding

if count_enc_cols != []:

    for col in count_enc_cols:

        summary   = X_train[col].value_counts()

        X_train[col] = X_train[col].map(summary)

        X_test[col]  = X_test[col].map(summary)

        

# target encoding

if target_enc_cols != []:

    target = "TradePrice"

    #skf is set up in the initial setting

    for col in target_enc_cols:  

        # X_testはX_trainでエンコーディングする

        X_temp = pd.concat([X_train, y_train], axis=1)

        summary = X_temp.groupby([col])[target].mean()

        enc_test = X_test[col].map(summary) 



        enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



        for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

            X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

            X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



            summary = X_train_.groupby([col])[target].mean()

            enc_train.iloc[val_ix] = X_val[col].map(summary)



        # target_encoding項目追加

        X_train[col] = enc_train

        X_test[col] = enc_test

    X_train.drop(columns = target_enc_cols)

    X_test.drop(columns = target_enc_cols)
log_cols       = []

RankGauss_cols = num_cols

standard_cols  = []

MinMax_cols    = []

Box_cols       = []

yeo_cols       = []
# log1p

if log_cols != []:

    X_train[log_cols] = np.log1p(X_train[log_cols])

    X_test [log_cols] = np.log1p(X_test[log_cols])

    # new_x = sign(x)*log(abs(x))



# RankGauss

if RankGauss_cols != []:

    n_quantiles = 500

    

    X_all = pd.concat([X_train, X_test], axis=0)

    X_all[RankGauss_cols] = quantile_transform(X_all[RankGauss_cols],n_quantiles=n_quantiles, random_state=71, output_distribution='normal')

    X_train = X_all.iloc[:X_train.shape[0], :]

    X_test = X_all.iloc[X_train.shape[0]:, :]

    

# StandardScaler

if standard_cols != []:

    scaler = StandardScaler()

    scaler.fit(X_train[standard_cols])

    X_train[standard_cols] = scaler.transform(X_train[standard_cols])

    X_test[standard_cols] = scaler.transform(X_test[standard_cols])

    

# MinMaxScaler

if MinMax_cols != []:

    scaler = MinMaxScaler()

    scaler.fit(X_train[MinMax_cols])

    X_train[MinMax_cols] = scaler.transform(X_train[MinMax_cols])

    X_test[MinMax_cols] = scaler.transform(X_test[MinMax_cols])



# Box_Cox

if Box_cols != []:

    # 正の値を持つ変数が対象

    pos_cols = [col for col in Box_cols if min(X_train[col]) > 0 and min(X_test[col]) > 0]

    pt = PowerTransformer(method = 'box-cox')

    pt.fit(X_train[pos_cols])

    X_train[pos_cols] = pt.transform(X_train[pos_cols])

    X_test[pos_cols] = pt.transform(X_test[pos_cols])



# Yeo-Johnson

if yeo_cols != []:

    pt = PowerTransformer(method='yeo-johnson')

    pt.fit(X_train[yeo_cols])

    X_train[yeo_cols] = pt.transform(X_train[yeo_cols])

    X_test[yeo_cols] = pt.transform(X_test[yeo_cols])




X_train_TXT = X_train.copy()

X_test_TXT  = X_test.copy()



X_train     = X_train.drop(columns = ['Structure','Use','Remarks']) # dropしないと機能しない

X_test      = X_test.drop(columns = ['Structure','Use','Remarks'])



#### Structure

TXT_train = X_train_TXT.Structure.copy() # この方式で変数を指定しないと動かない

TXT_test = X_test_TXT.Structure.copy()



# 欠損値埋

TXT_train.fillna('#', inplace=True)

TXT_test.fillna('#', inplace=True)



# TfidfVectorizer

tfidf     = TfidfVectorizer(max_features=5, analyzer='word', ngram_range=(1, 2))  

TXT_train = tfidf.fit_transform(TXT_train)

TXT_test  = tfidf.transform(TXT_test)



# svdする場合

n_components = 2

svd = TruncatedSVD(n_components=n_components)

TXT_train = svd.fit_transform(TXT_train.toarray())

TXT_test  = svd.transform(TXT_test.toarray()) #.todense()



columns = []

for i in range(0,n_components):

    columns.append("Structure_{}".format(i))

TXT_train = pd.DataFrame(TXT_train,columns=columns,index=X_train.index)

TXT_test  = pd.DataFrame(TXT_test,columns=columns,index=X_test.index)

X_train   = pd.concat([X_train,TXT_train],axis=1)

X_test    = pd.concat([X_test,TXT_test],axis=1)



#### Use

TXT_train = X_train_TXT.Use.copy() # この方式で変数を指定しないと動かない

TXT_test = X_test_TXT.Use.copy()



# 欠損値埋

TXT_train.fillna('#', inplace=True)

TXT_test.fillna('#', inplace=True)



# TfidfVectorizer

tfidf     = TfidfVectorizer(max_features=5, analyzer='word', ngram_range=(1, 2))  

TXT_train = tfidf.fit_transform(TXT_train)

TXT_test  = tfidf.transform(TXT_test)



# svdする場合

n_components = 2

svd = TruncatedSVD(n_components=n_components)

TXT_train = svd.fit_transform(TXT_train.toarray())

TXT_test  = svd.transform(TXT_test.toarray()) #.todense()



columns = []

for i in range(0,n_components):

    columns.append("Use_{}".format(i))

TXT_train = pd.DataFrame(TXT_train,columns=columns,index=X_train.index)

TXT_test  = pd.DataFrame(TXT_test,columns=columns,index=X_test.index)

X_train   = pd.concat([X_train,TXT_train],axis=1)

X_test    = pd.concat([X_test,TXT_test],axis=1)



#### Remarks

TXT_train = X_train_TXT.Remarks.copy() # この方式で変数を指定しないと動かない

TXT_test = X_test_TXT.Remarks.copy()



# 欠損値埋

TXT_train.fillna('#', inplace=True)

TXT_test.fillna('#', inplace=True)



# TfidfVectorizer

tfidf     = TfidfVectorizer(max_features=5, analyzer='word', ngram_range=(1, 2))  

TXT_train = tfidf.fit_transform(TXT_train)

TXT_test  = tfidf.transform(TXT_test)



# svdする場合

n_components = 2

svd = TruncatedSVD(n_components=n_components)

TXT_train = svd.fit_transform(TXT_train.toarray())

TXT_test  = svd.transform(TXT_test.toarray()) #.todense()



columns = []

for i in range(0,n_components):

    columns.append("Remarks_{}".format(i))

TXT_train = pd.DataFrame(TXT_train,columns=columns,index=X_train.index)

TXT_test  = pd.DataFrame(TXT_test,columns=columns,index=X_test.index)

X_train   = pd.concat([X_train,TXT_train],axis=1)

X_test    = pd.concat([X_test,TXT_test],axis=1)
X_train.isnull().sum()
X_test.isnull().sum()
# 欠損フラグ(テストデータに欠損があるもののみ) #Regressionの際に積極的に使う

#cols = []

#for col in cols:

#    flag_name = col + '_isnull'

#    df_train[flag_name] = df_train[col].map(lambda x: 1 if pd.isnull(x) else 0) 

#    df_test[flag_name] = df_test[col].map(lambda x: 1 if pd.isnull(x) else 0)
# 補完





X_train.fillna(-9999, inplace=True)

X_test.fillna(-9999, inplace=True)

#X_train.fillna(X_train.median(), inplace=True)

#X_test.fillna(X_train.median(), inplace=True)
# 上記計算が無限大対応

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)

X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.describe()
X_test.describe()
scores = []

lgb_y_pred_train = np.zeros(X_train.shape[0])

lgb_y_pred_test = np.zeros(X_test.shape[0])



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

        

    #clf = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

    #                            importance_type='split', learning_rate=0.05, max_depth=-1,

    #                            min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

    #                            n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

    #                            random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

    #                            subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    clf = LGBMRegressor(boosting_type='gbdt', num_leaves=15, max_depth=- 1, learning_rate=0.1,

                        n_estimators=1000, subsample_for_bin=200000, objective=None, 

                        class_weight=None, min_split_gain=0.0, min_child_weight=0.001,

                        min_child_samples=30, subsample=0.8, subsample_freq=0, colsample_bytree=0.8,

                        reg_alpha=1, reg_lambda=1, random_state=None, n_jobs=- 1,loss_function = 'RMSE',eval_metric = 'RMSE')

    

    

    clf.fit(X_train_, np.log1p(y_train_))            #RMSLEなので np.log1p(y_train_)にする

    y_pred = np.expm1(clf.predict(X_val)) #RMSLEなので expm1にする

    

    lgb_y_pred_train[test_ix] = y_pred

    score = mean_squared_log_error(y_val, y_pred)**0.5 #RMSLE

    scores.append(score)

    lgb_y_pred_test += np.expm1(clf.predict(X_test))

    

    print('CV Score of Fold_%d is %f' % (i, score))







lgb_y_pred_test /= 5
imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

imp
fig, ax = plt.subplots(figsize=(5, 8))

lightgbm.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
scores = []

catb_y_pred_train = np.zeros(X_train.shape[0])

catb_y_pred_test = np.zeros(X_test.shape[0])



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



    clf = CatBoostRegressor(n_estimators = 500,loss_function = 'RMSE',eval_metric = 'RMSE')

#    clf = CatBoostRegressor(n_estimators = 100, learning_rate=0.1, depth=5,loss_function = 'RMSE',eval_metric = 'RMSE')    

    

    clf.fit(X_train_, np.log1p(y_train_))            #RMSLEなので np.log1p(y_train_)にする

    y_pred = np.expm1(clf.predict(X_val)) #RMSLEなので expm1にする

    

    catb_y_pred_train[test_ix] = y_pred

    score = mean_squared_log_error(y_val, y_pred)**0.5 #RMSLE

    scores.append(score)

    catb_y_pred_test += np.expm1(clf.predict(X_test))

    

    

    print('CV Score of Fold_%d is %f' % (i, score))





catb_y_pred_test /= 5
cols = ['Remarks_0','Structure_1','Quarter','Use_1','Structure_0',]

X_train = X_train.drop(columns = cols)

X_test  = X_train.drop(columns = cols)



scores = []

lgb_y_pred_train2 = np.zeros(X_train.shape[0])

lgb_y_pred_test2 = np.zeros(X_test.shape[0])



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

        

    #clf = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

    #                            importance_type='split', learning_rate=0.05, max_depth=-1,

    #                            min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

    #                            n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

    #                            random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

    #                            subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    clf = LGBMRegressor(boosting_type='gbdt', num_leaves=15, max_depth=- 1, learning_rate=0.1,

                        n_estimators=1000, subsample_for_bin=200000, objective=None, 

                        class_weight=None, min_split_gain=0.0, min_child_weight=0.001,

                        min_child_samples=30, subsample=0.8, subsample_freq=0, colsample_bytree=0.8,

                        reg_alpha=1, reg_lambda=1, random_state=None, n_jobs=- 1,loss_function = 'RMSE',eval_metric = 'RMSE')

    

    

    clf.fit(X_train_, np.log1p(y_train_))            #RMSLEなので np.log1p(y_train_)にする

    y_pred = np.expm1(clf.predict(X_val)) #RMSLEなので expm1にする

    

    lgb_y_pred_train2[test_ix] = y_pred

    score = mean_squared_log_error(y_val, y_pred)**0.5 #RMSLE

    scores.append(score)

    lgb_y_pred_test2 += np.expm1(clf.predict(X_test))

    

    print('CV Score of Fold_%d is %f' % (i, score))





lgb_y_pred_test2 /= 5
submission = pd.read_csv('/kaggle/input/exam-for-students20200527/sample_submission.csv', index_col=0)

submission.TradePrice = (lgb_y_pred_test + lgb_y_pred_test2 + catb_y_pred_test)/2

submission.to_csv('submission.csv')
y_train