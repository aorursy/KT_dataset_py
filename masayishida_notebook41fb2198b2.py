# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import gc

import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series

import seaborn as sns

sns.set_style("whitegrid")



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



import os

from sklearn.metrics import roc_auc_score,mean_squared_error,mean_squared_log_error

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import quantile_transform,StandardScaler,MinMaxScaler

from sklearn.decomposition import SparsePCA,TruncatedSVD

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb

from lightgbm import LGBMRegressor



from catboost import CatBoostClassifier, FeaturesData, Pool,CatBoostRegressor
#ヒートマップ

def heatmap(y_data,x_data):

    fig, ax = plt.subplots(figsize=(12, 9)) 

    sns.heatmap(pd.concat([y_data,x_data], axis=1).corr(), square=True, vmax=1, vmin=-1, center=0)
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique() #ユニークな要素の個数、頻度（出現回数）をカウント

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
datapath='/kaggle/input/ml-exam-20201006/'
df_train = pd.read_csv(datapath+'train.csv', index_col='id')

df_test = pd.read_csv(datapath+'test.csv', index_col='id')
df_city_info=pd.read_csv(datapath+'city_info.csv')

df_station_info=pd.read_csv(datapath+'station_info.csv')
df_train.columns
df_train=pd.merge(df_train,df_city_info, on=['Prefecture', 'Municipality'], how='left')

df_test=pd.merge(df_test,df_city_info, on=['Prefecture', 'Municipality'], how='left')
#主要駅からの距離を計算(特徴量ENG)

#東京

df_station_info['dis_from_tokyo'] = (df_station_info['Latitude']- 35.6875)**2 + (df_station_info['Longitude']- 139.75)**2



#池袋

df_station_info['dis_from_ikebukuro'] = (df_station_info['Latitude']- 35.7295384)**2 + (df_station_info['Longitude']- 139.7131303)**2



#新宿

df_station_info['dis_from_shinjuku'] = (df_station_info['Latitude']- 35.6896067)**2 + (df_station_info['Longitude']- 139.7005713)**2



#渋谷

df_station_info['dis_from_shibuya'] = (df_station_info['Latitude']- 35.6580339)**2 + (df_station_info['Longitude']- 139.7016358)**2



#上野

#df_station_info['dis_from_ueno'] = (df_station_info['Latitude']- 35.7141672)**2 + (df_station_info['Longitude']- 139.7774091)**2



'''#大宮

df_station_info['dis_lat_from_omiya'] = (df_station_info['Latitude']- 35.9064485)**2

df_station_info['dis_lon_from_omiya'] = (df_station_info['Longitude']- 139.6238548)**2

df_station_info['dis_from_omiya'] = df_station_info['dis_lat_from_omiya'] + df_station_info['dis_lon_from_omiya']

df_station_info

'''

#新橋35.666379　139.7583398

df_station_info['dis_from_shinbashi'] = (df_station_info['Latitude']- 35.666379)**2 + (df_station_info['Longitude']- 139.7583398)**2



df_station_info
df_station_info=df_station_info.rename(columns={'Station':'NearestStation','Latitude':'st_lat','Longitude':'st_lon'})

df_station_info
df_train=pd.merge(df_train,df_station_info, on=['NearestStation'], how='left')

df_test=pd.merge(df_test,df_station_info, on=['NearestStation'], how='left')
df_train
df_test
#Xとyに分割

target = 'TradePrice'

y_train = df_train[target] #yは貸し倒れフラグ

X_train = df_train.drop([target], axis=1) #xはそれ以外



X_test = df_test
# dtypeがobjectとnumericのものに分ける

cats = []

nums = []



for col in X_train.columns:

    if X_train[col].dtype == 'object' or X_train[col].dtype == 'datetime64[ns]' or X_train[col].dtype == 'bool':

        cats.append(col)

    else:

        nums.append(col)
#テキスト特徴量はcatsからDrop

txt= ['Structure','Use','Remarks','CityPlanning']

for l in txt:

    cats.remove(l)
X_train[nums]
X_train['PrewarBuilding'].value_counts()
X_test['PrewarBuilding'].value_counts()
plotPerColumnDistribution(X_train[nums], 10, 5)
plotPerColumnDistribution(X_test[nums], 10, 5)
cats
plotPerColumnDistribution(X_train[cats], 10, 5)
plotPerColumnDistribution(X_test[cats], 10, 5)
X_train['CityPlanning'].value_counts()
X_train['Structure'].value_counts()
X_train[nums]
nums
X_test['PrewarBuilding'].describe()
X_train['PrewarBuilding'].describe()
remove_nums=['MaxTimeToNearestStation'] #MinTimeに集約

X_train = X_train.drop(remove_nums,axis=1)

X_test = X_test.drop(remove_nums,axis=1)
for i in remove_nums:

    try:

        nums.remove(i)

    except: pass
# RankGauss

RankGauss_cols = nums

n_quantiles = 500

    

X_all = pd.concat([X_train, X_test], axis=0)

X_all[RankGauss_cols] = quantile_transform(X_all[RankGauss_cols],n_quantiles=n_quantiles, random_state=71, output_distribution='normal')

X_train = X_all.iloc[:X_train.shape[0], :]

X_test = X_all.iloc[X_train.shape[0]:, :]
#Municipalityは使えないので、Ward Village CountyでOH

X_train['Ward_Municipality'] = X_train['Municipality'].apply(lambda x: 1 if 'Ward' in str(x) else 0 )

X_train['Village_Municipality'] = X_train['Municipality'].apply(lambda x: 1 if 'Village' in str(x) else 0)

X_train['County_Municipality'] = X_train['Municipality'].apply(lambda x: 1 if 'County' in str(x) else 0)



X_test['Ward_Municipality'] = X_test['Municipality'].apply(lambda x: 1 if 'Ward' in str(x) else 0 )

X_test['Village_Municipality'] = X_test['Municipality'].apply(lambda x: 1 if 'Village' in str(x) else 0)

X_test['County_Municipality'] = X_test['Municipality'].apply(lambda x: 1 if 'County' in str(x) else 0)
X_train['Renovation']=X_train['Renovation'].fillna('Not yet')

X_test['Renovation']=X_test['Renovation'].fillna('Not yet')
remove_cats=['Prefecture','Municipality','DistrictName','NearestStation','TimeToNearestStation']

X_train = X_train.drop(remove_cats,axis=1)

X_test = X_test.drop(remove_cats,axis=1)
for i in remove_cats:

    try:

        cats.remove(i)

    except: pass  
#target enc

cate_list = ['Purpose',

             'Direction']

target = 'Area'



X_train[cate_list].fillna('#',inplace=True)

X_test[cate_list].fillna('#',inplace=True)



X_temp = pd.concat([X_train, y_train], axis=1)



for col in cate_list:



    summary = X_temp.groupby([col])[target].mean()

    enc_test = X_test[col].map(summary) 





    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

        

    # target_encoding項目追加

    X_train['target_' + col] = enc_train

    X_test['target_' + col] = enc_test
X_train = X_train.drop(cate_list,axis=1)

X_test = X_test.drop(cate_list,axis=1)



for i in cate_list:

    try:

        cats.remove(i)

    except: pass    
#ordinal encoder

oe = OrdinalEncoder(cols=cats, return_df=False)

X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.fit_transform(X_test[cats])
X_train[cats].fillna(-9999,inplace=True)

X_test[cats].fillna(-9999,inplace=True)



X_train['target_Purpose'].fillna(-9999,inplace=True)

X_test['target_Purpose'].fillna(-9999,inplace=True)

       

X_train['target_Direction'].fillna(-9999,inplace=True)

X_test['target_Direction'].fillna(-9999,inplace=True)
X_train
for l in txt:

    TXT_train = X_train[l].copy() # この方式で変数を指定しないと動かない

    TXT_test = X_test[l].copy()



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

        columns.append(l+"_{}".format(i))

    TXT_train = pd.DataFrame(TXT_train,columns=columns,index=X_train.index)

    TXT_test  = pd.DataFrame(TXT_test,columns=columns,index=X_test.index)

    X_train   = pd.concat([X_train,TXT_train],axis=1)

    X_test    = pd.concat([X_test,TXT_test],axis=1)
X_train = X_train.drop(txt,axis=1)

X_test = X_test.drop(txt,axis=1)
X_train.columns
X_test.columns
#最寄り駅からの距離

X_train['dis_from_nearst'] = (X_train['Latitude']- X_train['st_lat'])**2 + (X_train['Longitude']- X_train['st_lon'])**2

X_test['dis_from_nearst'] = (X_test['Latitude']- X_test['st_lat'])**2 + (X_test['Longitude']- X_test['st_lon'])**2
remove_nums=['st_lat','st_lon'] ####

X_train = X_train.drop(remove_nums,axis=1)

X_test = X_test.drop(remove_nums,axis=1)
# 学習用と検証用に分割する

X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.05, random_state=71)
#まずはチューニングなし

scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





clf = LGBMRegressor(boosting_type='gbdt', num_leaves=15, max_depth=- 1, learning_rate=0.1,

                        n_estimators=1000, subsample_for_bin=200000, objective=None, 

                        class_weight=None, min_split_gain=0.0, min_child_weight=0.001,

                        min_child_samples=30, subsample=0.8, subsample_freq=0, colsample_bytree=0.8,

                        reg_alpha=1, reg_lambda=1, random_state=None, n_jobs=- 1,loss_function = 'RMSE',eval_metric = 'RMSE')

 
%%time

clf.fit(X_train_, np.log1p(y_train_))            #RMSLEなので np.log1p(y_train_)にする
clf.booster_.feature_importance(importance_type='gain')

imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

fig, ax = plt.subplots(figsize=(5, 8))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
th=1000



use_col = imp.index[imp.importance > th] #importance閾値を切って特長項目を厳選。
X_train_=X_train_[use_col]

X_train=X_train[use_col]

X_val=X_val[use_col]

X_test=X_test[use_col]
heatmap(y_train,X_train[use_col])
scores = []

lgb_y_pred_train = np.zeros(X_train.shape[0])

lgb_y_pred_test = np.zeros(X_test.shape[0])



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

        

    clf = LGBMRegressor(boosting_type='gbdt', num_leaves=15, max_depth=- 1, learning_rate=0.1,

                        n_estimators=1000, subsample_for_bin=200000, objective=None, 

                        class_weight=None, min_split_gain=0.0, min_child_weight=0.001,

                        min_child_samples=30, subsample=0.8, subsample_freq=0, colsample_bytree=0.8,

                        reg_alpha=1, reg_lambda=1, random_state=None, n_jobs=- 1,loss_function = 'RMSE',eval_metric = 'RMSE')

    

    

    clf.fit(X_train_, np.log1p(y_train_))

    y_pred = np.expm1(clf.predict(X_val))

    

    lgb_y_pred_train[test_ix] = y_pred

    score = mean_squared_log_error(y_val, y_pred)**0.5 #RMSLE

    scores.append(score)

    lgb_y_pred_test += np.expm1(clf.predict(X_test))

    

    print('CV Score of Fold_%d is %f' % (i, score))



lgb_y_pred_test /= 5



ave_scores=0

for l in scores:

    ave_scores += l/len(scores)

print('Average_score is %f' % ( score))
scores = []

catb_y_pred_train = np.zeros(X_train.shape[0])

catb_y_pred_test = np.zeros(X_test.shape[0])



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



    clf = CatBoostRegressor(n_estimators = 500,loss_function = 'RMSE',eval_metric = 'RMSE')

    

    clf.fit(X_train_, np.log1p(y_train_)) 

    y_pred = np.expm1(clf.predict(X_val)) 

    

    catb_y_pred_train[test_ix] = y_pred

    score = mean_squared_log_error(y_val, y_pred)**0.5 #RMSLE

    scores.append(score)

    catb_y_pred_test += np.expm1(clf.predict(X_test))

    

    

    print('CV Score of Fold_%d is %f' % (i, score))





catb_y_pred_test /= 5
ave_scores=0

for l in scores:

    ave_scores += l/len(scores)

print('Average_score is %f' % ( score))
submission = pd.read_csv(datapath+'sample_submission.csv', index_col=0)

submission.TradePrice = (lgb_y_pred_test+catb_y_pred_test)/2

submission.to_csv('submission.csv')
submission