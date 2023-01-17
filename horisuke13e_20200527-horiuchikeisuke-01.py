import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import scipy.stats as stats

import seaborn as sns

import time

import math



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_log_error

from sklearn import preprocessing



%matplotlib inline

pd.options.display.max_rows = 1000

pd.options.display.max_columns = 100



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_station_info = pd.read_csv('/kaggle/input/exam-for-students20200527/station_info.csv')

df_city_info = pd.read_csv('/kaggle/input/exam-for-students20200527/city_info.csv')

df_data_dict = pd.read_csv('/kaggle/input/exam-for-students20200527/data_dictionary.csv')
#print(df_station_info.shape)

#df_station_info.head()
#print(df_city_info.shape)

#df_city_info.head()
#print(df_data_dict.shape)

#df_data_dict.head()
df_train = pd.read_csv('/kaggle/input/exam-for-students20200527/train.csv')

df_test = pd.read_csv('/kaggle/input/exam-for-students20200527/test.csv')

df_merged = pd.concat([df_train, df_test], axis=0, sort=False)



print(df_train.shape)
COL_TARGET = 'TradePrice'

COL_ID = 'id'
#df_train[df_train['Prefecture']=='Tokyo'].head(50)
### 市の位置をマージ



df_train = pd.merge(df_train, df_city_info, on=['Prefecture', 'Municipality'], how='left')

df_train = df_train.rename(columns={'Latitude': 'MunicipalityLatitude', 'Longitude': 'MunicipalityLongitude'})

df_test = pd.merge(df_test, df_city_info, on=['Prefecture', 'Municipality'], how='left')

df_test = df_test.rename(columns={'Latitude': 'MunicipalityLatitude', 'Longitude': 'MunicipalityLongitude'})

df_test.head(30)
### 駅の位置をマージ



#df_train = pd.merge(df_train, df_station_info, left_on='NearestStation', right_on='Station', how='left')

#df_train = df_train.rename(columns={'Latitude': 'MunicipalityLatitude', 'Longitude': 'MunicipalityLongitude'})

#df_test = pd.merge(df_test, df_city_info, on=['Prefecture', 'Municipality'], how='left')

#df_test = df_test.rename(columns={'Latitude': 'MunicipalityLatitude', 'Longitude': 'MunicipalityLongitude'})

#df_test.head(30)



#for col in df_train
df_train['Municipality'].nunique()
#set(df_test['NearestStation'].unique()) & set(df_station_info['Station'].unique())
#df_train = pd.merge(df_train, df_station_info, left_on='NearestStation', right_on='Station', how='left')

#df_train.head()
### ユニーク数

#for col in df_train.columns:

#    print(col.ljust(len('TotalFloorAreaIsGreaterFlag')), ':', df_train[col].nunique())
### COL_TARGET (TradePrice) の対数変換



df_train[COL_TARGET] = np.log1p(df_train[COL_TARGET])
### AreaIsGreaterFlag



#print(df_merged['AreaIsGreaterFlag'].unique())



convert_mae = [0, 1]

convert_ato = [False, True]

df_train['AreaIsGreaterFlag'] = df_train['AreaIsGreaterFlag'].replace(convert_mae, convert_ato)

df_train['AreaIsGreaterFlag'] = df_train['AreaIsGreaterFlag'].astype(bool)

df_test['AreaIsGreaterFlag'] = df_test['AreaIsGreaterFlag'].replace(convert_mae, convert_ato)

df_test['AreaIsGreaterFlag'] = df_test['AreaIsGreaterFlag'].astype(bool)
### PrewarBuilding



print(df_merged['PrewarBuilding'].unique())



convert_mae = [0, 1]

convert_ato = [False, True]

df_train['PrewarBuilding'] = df_train['PrewarBuilding'].replace(convert_mae, convert_ato)

df_train['PrewarBuilding'] = df_train['PrewarBuilding'].astype(bool)

df_test['PrewarBuilding'] = df_test['PrewarBuilding'].replace(convert_mae, convert_ato)

df_test['PrewarBuilding'] = df_test['PrewarBuilding'].astype(bool)
### TimeToNearestStation



print(df_merged['TimeToNearestStation'].unique())



convert_mae = ['1H-1H30', '30-60minutes', '1H30-2H', '2H-']

convert_ato = [75, 45, 105, 120]

df_train['TimeToNearestStation'] = df_train['TimeToNearestStation'].replace(convert_mae, convert_ato)

df_train['TimeToNearestStation'] = df_train['TimeToNearestStation'].astype(float)

df_test['TimeToNearestStation'] = df_test['TimeToNearestStation'].replace(convert_mae, convert_ato)

df_test['TimeToNearestStation'] = df_test['TimeToNearestStation'].astype(float)
### FloorPlan



print(sorted(df_merged['FloorPlan'].dropna().unique()))



convert_mae = ['1K', '1K+S', '1R', '1R+S', # 4

               '1DK', '1DK+S', '1LDK', '1LDK+S',  # 4

               '2DK', '2DK+S', '2K', '2K+S', '2LD', '2LD+S', '2LDK', '2LDK+S', '2LK', '2LK+S', # 10

               '3DK', '3DK+S', '3K', '3LD', '3LDK', '3LDK+K', '3LDK+S', '3LK', # 8

               '4DK', '4DK+S', '4K', '4LDK', '4LDK+K', '4LDK+S', # 6

               '5DK', '5LDK', '5LDK+K', '5LDK+S', # 4

               '6LDK', # 1

               'Duplex', 'Open Floor', 'Studio Apartment'] # 3

convert_ato = [1, 1, 1, 1, 

               2, 2, 2, 2, 

               3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 

               4, 4, 4, 4, 4, 4, 4, 4, 

               5, 5, 5, 5, 5, 5, 

               6, 6, 6, 6, 

               7, 

               10, 10, 10]

df_train['FloorPlan'] = df_train['FloorPlan'].replace(convert_mae, convert_ato)

df_train['FloorPlan'] = df_train['FloorPlan'].astype(float)

df_test['FloorPlan'] = df_test['FloorPlan'].replace(convert_mae, convert_ato)

df_test['FloorPlan'] = df_test['FloorPlan'].astype(float)



df_train['FloorPlan_KikakugaiFlg'] = False

df_train.loc[(df_train['FloorPlan']=='Duplex') |

             (df_train['FloorPlan']=='Open Floor') |

             (df_train['FloorPlan']=='Studio Apartment'),

             'FloorPlan_KikakugaiFlg'] = True

df_test['FloorPlan_KikakugaiFlg'] = False

df_test.loc[(df_test['FloorPlan']=='Duplex') |

            (df_test['FloorPlan']=='Open Floor') |

            (df_test['FloorPlan']=='Studio Apartment'),

            'FloorPlan_KikakugaiFlg'] = True
### Renovation



#print(df_merged['Renovation'].unique())



convert_mae = ['Not yet', 'Done']

convert_ato = [False, True]

df_train['Renovation'] = df_train['Renovation'].replace(convert_mae, convert_ato)

df_train['Renovation'] = df_train['Renovation'].astype(bool)

df_test['Renovation'] = df_test['Renovation'].replace(convert_mae, convert_ato)

df_test['Renovation'] = df_test['Renovation'].astype(bool)
### BuildingYear

df_train['BuildingAge'] = 2020 - df_train['BuildingYear']

df_test['BuildingAge'] = 2020 - df_test['BuildingYear']
### MinTimeToNearestStation, MaxTimeToNearestStation



#df_train['DiffMaxMin_TimeToNearestStation'] = df_train['MaxTimeToNearestStation'] - df_train['MinTimeToNearestStation']

#df_train['DiffMaxMin_TimeToNearestStation'].unique()
# 不要列削除

cols_drop = [COL_TARGET, COL_ID] + ['Year', 'Quarter', 'BuildingYear', 'MinTimeToNearestStation', 'MaxTimeToNearestStation']

df_train_dropped = df_train.drop(columns=cols_drop)



# カテゴリ、数値

cols_qualitative = [col for col, dtype in df_train_dropped.dtypes.iteritems() if dtype == 'object']

cols_quantitative = [col for col, dtype in df_train_dropped.dtypes.iteritems() if dtype != 'object']
# 欠損値：カテゴリ特徴量

if cols_qualitative != []:

    missing = df_train[cols_qualitative].isnull().sum()

    missing = missing[missing > 0]

    missing.sort_values(inplace=True)

    #print(missing.index.values)

    #missing.plot.bar(color='tab:blue', ylim=[0, len(df_train)])
#df_train[cols_qualitative].head(10)
# 欠損値：数値特徴量

if cols_quantitative != []:

    missing = df_train[cols_quantitative].isnull().sum()

    missing = missing[missing > 0]

    missing.sort_values(inplace=True)

    #print(missing.index.values)

    #missing.plot.bar(color='tab:blue', ylim=[0, len(df_train)])
#df_train[cols_quantitative].head(10)
# まずは千葉県とみなしてサブミット、物件価格が一番近い

df_test['Prefecture'] = 'Chiba Prefecture'
### レコードごとの欠損値数を特徴量に追加



df_train['n_nan'] = df_train.isnull().sum(axis=1)

df_test['n_nan'] = df_test.isnull().sum(axis=1)
!pip install pyproj
# https://qiita.com/damyarou/items/9cb633e844c78307134a

def cal_rho(lon_a,lat_a,lon_b,lat_b):

    ra=6378.140  # equatorial radius (km)

    rb=6356.755  # polar radius (km)

    F=(ra-rb)/ra # flattening of the earth

    rad_lat_a=np.radians(lat_a)

    rad_lon_a=np.radians(lon_a)

    rad_lat_b=np.radians(lat_b)

    rad_lon_b=np.radians(lon_b)

    pa=np.arctan(rb/ra*np.tan(rad_lat_a))

    pb=np.arctan(rb/ra*np.tan(rad_lat_b))

    xx=np.arccos(np.sin(pa)*np.sin(pb)+np.cos(pa)*np.cos(pb)*np.cos(rad_lon_a-rad_lon_b))

    c1=(np.sin(xx)-xx)*(np.sin(pa)+np.sin(pb))**2/np.cos(xx/2)**2

    c2=(np.sin(xx)+xx)*(np.sin(pa)-np.sin(pb))**2/np.sin(xx/2)**2

    dr=F/8*(c1-c2)

    rho=ra*(xx+dr)

    return rho



municipality_locations_train = df_train[['MunicipalityLatitude', 'MunicipalityLongitude']].values.tolist()

municipality_locations_test = df_test[['MunicipalityLatitude', 'MunicipalityLongitude']].values.tolist()

distances_train = [cal_rho(loc[0], loc[1], location_chiyoda[0], location_chiyoda[1]) for loc in municipality_locations_train]

distances_test = [cal_rho(loc[0], loc[1], location_chiyoda[0], location_chiyoda[1]) for loc in municipality_locations_test]

df_train['DistanceToTokyo'] = distances_train

df_test['DistanceToTokyo'] = distances_test
### 欠損値補完：カテゴリ特徴量

for col in cols_qualitative_strong:

    df_train[col] = df_train[col].fillna('NA')

    df_test[col] = df_test[col].fillna('NA')



if False:

    # 各カテゴリ列ごとに、カテゴリ値に対するターゲット値の分布を検定

    # その列のカテゴリ値の変化によってターゲット値が大きく変わる（あるいは変わらない）かどうかみる

    def anova(df, cols_qualitative):

        df_anova = pd.DataFrame()

        df_anova['feature'] = cols_qualitative

        pvals = []

        for col in cols_qualitative:

            samples = []

            for cls in df[col].unique():

                s = df[df[col] == cls][COL_TARGET].values

                samples.append(s)

            pval = stats.f_oneway(*samples)[1]

            pvals.append(pval)

        df_anova['pval'] = pvals

        return df_anova.sort_values('pval')



    sr = df_train[cols_qualitative].nunique()

    cols_qualitative_tmp = sr[sr < 50].index.tolist() # ユニーク数50以下のものだけ計算

    cols_qualitative_tmp_over50 = sr[~(sr < 50)].index.tolist()



    df_anova = anova(df_train, cols_qualitative_tmp)

    cols_qualitative_strong = df_anova[df_anova['pval'] < 0.05]['feature'].tolist()

    cols_qualitative_weak = df_anova[~(df_anova['pval'] < 0.05)]['feature'].tolist()



    cols_qualitative_strong += cols_qualitative_tmp_over50



    # 重要な列は欠損値らしく補完、その他は最頻値で補完

    for col in cols_qualitative_strong:

        df_train[col] = df_train[col].fillna('NA')

        df_test[col] = df_test[col].fillna('NA')

    for col in cols_qualitative_weak:

        mode = df_train[col].mode()[0]

        df_train[col] = df_train[col].fillna(mode)

        df_test[col] = df_test[col].fillna(mode)



    df_anova
### 欠損値補完：数値特徴量



# 相関を計算

cols_quantitative_strong = []

cols_quantitative_weak = []

for col in cols_quantitative:

    corr = df_train[COL_TARGET].corr(df_train[col])

    if abs(corr) >= 0.5:

        cols_quantitative_strong.append(col)

    else:

        cols_quantitative_weak.append(col)



# 重要な列は欠損値らしく補完、その他は最頻値で補完

for col in cols_quantitative_strong:

    df_train[col] = df_train[col].fillna(-999999)

    df_test[col] = df_test[col].fillna(-999999)

for col in cols_quantitative_weak:

    median = df_train[col].median()

    df_train[col] = df_train[col].fillna(median)

    df_test[col] = df_test[col].fillna(median)
#df_train.drop(columns=cols_drop).head()
### エンコーディング



# めんどくさいので全てカウントエンコーディング

for col in cols_qualitative:

    freq = df_train[col].value_counts()

    df_train[col] = df_train[col].map(freq)

    df_test[col] = df_test[col].map(freq)
#df_train.drop(columns=cols_drop).head()
#df_test.head()
from lightgbm import LGBMRegressor



# ver1では、千葉県とみなしてサブミット。駅情報とかは使わない

cols_drop += ['Municipality', 'DistrictName', 'NearestStation', 'MunicipalityLatitude', 'MunicipalityLongitude']



X_train = df_train.drop(columns=cols_drop).values

y_train = df_train[COL_TARGET].values



cols_drop.remove(COL_TARGET)

X_test = df_test.drop(columns=cols_drop).values



### 標準化してみる

sc = preprocessing.StandardScaler()

sc.fit(X_train)

X_train = sc.transform(X_train)

X_test = sc.transform(X_test)



model = LGBMRegressor(boosting_type='gbdt')

model.fit(X_train, y_train)

y_pred = math.e ** model.predict(X_test)



# 埼玉の家賃は千葉の家賃の1.0336倍

# http://grading.jpn.org/y1805018.html

cs_rate = 1.0336

y_pred = [val*cs_rate for val in y_pred]
scores = []

kf = KFold(n_splits=5, random_state=0, shuffle=True)

for i, (train_ix, val_ix) in enumerate(kf.split(X_train, y_train)):

    _X_train, _y_train = X_train[train_ix], y_train[train_ix]

    _X_val, _y_val = X_train[val_ix], y_train[val_ix]

    model = LGBMRegressor(boosting_type='gbdt')

    

    model.fit(_X_train, _y_train, eval_set=[(_X_val, _y_val)])

    _y_pred = math.e ** model.predict(_X_val)

    _y_pred = [val*cs_rate for val in _y_pred]

    score = np.sqrt(mean_squared_log_error(_y_val, _y_pred))

    scores.append(score)
y_pred_train = math.e ** model.predict(X_train)

y_pred_train = [val*cs_rate for val in y_pred_train]

print('Train Score: {}'.format(np.sqrt(mean_squared_log_error(y_train, y_pred_train))))



print('CV {} Score: {}'.format(i+1, scores))

print('Average Score: {}'.format(np.mean(scores)))
submission = pd.DataFrame({COL_ID: df_test[COL_ID]})

submission[COL_TARGET] = y_pred

submission.head()
submission.to_csv('submission_04_rate.csv', index=False)

print(' > saved')