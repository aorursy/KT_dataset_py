# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df=pd.read_csv('../input/ga-customer-revenue-prediction/train.csv')
df_sample = df.iloc[:1000, :]

import json
from pandas.io.json import json_normalize
df2 = df_sample.copy().iloc[:1000, :]

df_device = json_normalize(df2['device'].apply(lambda x: json.loads(x)))
df_geo = json_normalize(df2['geoNetwork'].apply(lambda x: json.loads(x)))
df_totals = json_normalize(df2['totals'].apply(lambda x: json.loads(x)))
df_traffice = json_normalize(df2['trafficSource'].apply(lambda x: json.loads(x)))
df_concat = pd.concat([df2, df_device, df_geo, df_totals, df_traffice], axis=1)
features = pd.get_dummies(df_concat.drop(columns=['transactionRevenue', 'device', 'geoNetwork', 'totals', 'trafficSource', 'sessionId', 'fullVisitorId']))
labels = df_concat['transactionRevenue'].fillna(0)
df_concat2 = pd.concat([features, labels], axis=1)
df_concat2.info()
df_concat2['transactionRevenue'] = df_concat2['transactionRevenue'].values.astype(int)
cor_ = df_concat2.corr().abs()
cor_ = cor_.fillna(0)
sorted_cor = cor_[['transactionRevenue']].sort_values('transactionRevenue', ascending=False)
sorted_cor[
    sorted_cor['transactionRevenue'] > 0.05
]
df_concat.to_csv('/kaggle/working/sample.csv', index=False)
df1.info()
df1.describe()
# 正規化用にdatasetの一部を抜き出し
df2 = df1[["date", "visitId", "visitNumber", "visitStartTime"]]
df2.head()
# z-score normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df2_std = sc.fit_transform(df2)
df2_std
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import itertools

def get_best_features(x, y, feature_names, model):
    # すべての説明変数名の組み合わせを入れるリスト
    _name_list = []

    # 指定された長さの組み合わせを作成する
    for i in range(1, len(feature_names)+1):
        for sublist in itertools.permutations(feature_names, i):
            _name_list.append(list(sublist))

    # 最良のMSEを入れる変数（大きい値を入れておく）
    _best_mse = 999999999999
    _best_mse_name = ''

    # 総当りで比較する
    for _name in _name_list:
        _x = np.array(x[_name])

        # データセットの分割。X4を使っていることに注意
        _x_train, _x_test, _y_train, _y_test = train_test_split(_x, y, test_size=0.3, random_state=0)
        _x_train, _x_valid, _y_train, _y_valid = train_test_split(_x_train, _y_train, test_size=0.3, random_state=0)

        # モデルの作成～予測
        model.fit(_x_train, _y_train)
        _y_pred = model.predict(_x_valid)

        # MSEを算出
        _mse = mean_squared_error(_y_valid, _y_pred)

        # 最小のmseを保管
        if _mse < _best_mse:
            _best_mse = _mse
            _best_mse_name = _name

    print(model.__class__.__name__, ":", ','.join(_best_mse_name), ": MSE=", _best_mse)
# 説明変数名のリスト
feature_names = ['date', 'sessionId', 'visitId', 'visitNumber', 'visitStartTime', 'channelGrouping_(Other)', 'channelGrouping_Affiliates']
#	visitId	visitNumber	visitStartTime	channelGrouping_(Other)	channelGrouping_Affiliates	channelGrouping_Direct	channelGrouping_Display	channelGrouping_Organic Search	channelGrouping_Paid Search	channelGrouping_Referral	channelGrouping_Social	socialEngagementType_Not Socially Engaged

# モデルのリストを用意
model_list = []
model_list.append(LinearRegression())
model_list.append(Lasso())
model_list.append(Ridge())
model_list.append(DecisionTreeRegressor())
model_list.append(RandomForestRegressor(n_estimators=100, random_state=0))

for model in model_list:
    get_best_features(df1, df1['fullVisitorld'], feature_names, model)
import json
def json_to_dict(json_str):
    json_str_ = json_str.replace("'", '"', "'")
    obj = json.loads(json_str_)

df2['hits_'] = df2['hits'].apply(lambda json_str: json_to_dict(json_str))
df2.head()
#browser = [device['key']['value'] if len(device) > 0 else 'None' for device in df2['device_'].values]
##df2['Browser'] = browser
#df2 = pd.get_dummies(df2, columns=['Browser'])
#df2.head()
import json
def json_to_dict(json_str):
    json_str_ = json_str.replace("'", '"', "'")
    obj = json.loads(json_str_)
#    for dimension in obj:
#         for key, value in dimension.items():
#            print(f'{key}={value}')
#    return obj
df2 = df1.copy()
df2['device_'] = df2['device'].apply(lambda json_str: json_to_dict(json_str))
df2.head()
browsers = [browser[0]['value'] if len(customDimension) > 0 else 'None' for device in df2['device_'].values]
df2['Browsers'] = browsers
df2 = pd.get_dummies(df2, columns=['Browsers'])
df2.head()

#{"browser": "Chrome", "browserVersion": "not available in demo dataset", "browserSize": "not available in demo dataset", "operatingSystem": "Windows", "operatingSystemVersion": "not available in demo dataset", "isMobile": false, "mobileDeviceBranding": "n
import json
def json_to_dict(json_str):
    json_str_ = json_str.replace("'", '"')
    obj = json.loads(json_str_)
#     for dimension in obj:
#         for key, value in dimension.items():
#             print(f'{key}={value}')
    return obj
df2 = df.copy()
df2['customDimensions_'] = df2['customDimensions'].apply(lambda json_str: json_to_dict(json_str))
df2.head()
areas = [customDimension[0]['value'] if len(customDimension) > 0 else 'None' for customDimension in df2['customDimensions_'].values]
df2['Area'] = areas
df2 = pd.get_dummies(df2, columns=['Area'])
df2.head()

# Y:目的変数に該当する列
Y = np.array(dataset3['fullVisitorId'])
# X:説明変数に該当する列
X = np.array(dataset3[['date', 'visitNumber', 'visitStartTime', 'visitId']])
