import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import gc
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from itertools import cycle
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/russiandata/train.csv')
macro = pd.read_csv('/kaggle/input/russiandata/macro.csv')
test = pd.read_csv('/kaggle/input/russiandata/test.csv')
#submit = pd.read_csv(f'{mydir}/RussianHouse/sample_submission.csv')

train.timestamp = pd.to_datetime(train.timestamp)
macro.timestamp = pd.to_datetime(macro.timestamp)
test.timestamp = pd.to_datetime(test.timestamp)
tmp_train = train.drop("price_doc", axis=1)
mdata = pd.concat([tmp_train, test]).reset_index(drop=True)
del tmp_train

mdata
def plot_columns(columns:list):
    fig = plt.figure(figsize=(6.0*3, 6.0*len(columns)))
    for size in range(len(columns)):
        ax_ = fig.add_subplot(len(columns), 3, size+1)
        column = columns[size]
        ax_.set_title(column)
        sns.violinplot(mdata[column], jitter=True , dodge=True ,ax=ax_)

# 外れ値やばすぎ問題
columns = ['life_sq', 'floor', 'num_room', 'kitch_sq','max_floor']
plot_columns(columns)
# 欠損値埋め
mdata.life_sq = mdata.life_sq.fillna(mdata.floor.mean())
mdata.floor = mdata.floor.fillna(mdata.floor.mean())
mdata.num_room = mdata.num_room.fillna(mdata.num_room.mean())
mdata.kitch_sq = mdata.kitch_sq.fillna(mdata.kitch_sq.mean())

mdata.max_floor = mdata.max_floor.fillna(mdata.max_floor.mean())
mdata.max_floor = [floor if floor > mfloor else mfloor for mfloor, floor in zip(mdata.max_floor, mdata.floor)]
columns = ['raion_build_count_with_material_info', 'build_count_block',
       'build_count_wood', 'build_count_frame','build_count_brick',
       'build_count_monolith','build_count_panel','build_count_foam',
       'build_count_slag','build_count_mix']
plot_columns(columns)
for column in columns:
    mdata[column] = mdata[column].fillna(mdata[column].mean())
columns = ['raion_build_count_with_builddate_info','build_count_before_1920',
       'build_count_1921-1945','build_count_1946-1970',
       'build_count_1971-1995','build_count_after_1995']
plot_columns(columns)
for column in columns:
    mdata[column] = mdata[column].fillna(mdata[column].mean())
columns = ['metro_min_walk','railroad_station_walk_km',
       'railroad_station_walk_min', 'metro_km_walk']
plot_columns(columns)
for column in columns:
    mdata[column] = mdata[column].fillna(mdata[column].mean())
# カフェシリーズ
columns = ['cafe_sum_500_min_price_avg','cafe_sum_500_max_price_avg','cafe_avg_price_500',
           'cafe_sum_1000_min_price_avg', 'cafe_sum_1000_max_price_avg', 'cafe_avg_price_1000',
           'cafe_sum_1500_min_price_avg', 'cafe_sum_1500_max_price_avg', 'cafe_avg_price_1500',
           'cafe_sum_2000_min_price_avg','cafe_sum_2000_max_price_avg','cafe_avg_price_2000',
           'cafe_sum_3000_min_price_avg','cafe_sum_3000_max_price_avg','cafe_avg_price_3000',
           'cafe_sum_5000_min_price_avg','cafe_sum_5000_max_price_avg','cafe_avg_price_5000']
plot_columns(columns)
for column in columns:
    mdata[column] = mdata[column].fillna(mdata[column].mean())
# その他シリーズ
columns = ['preschool_quota','school_quota','prom_part_5000','hospital_beds_raion', 'green_part_2000']
plot_columns(columns)
for column in columns:
    mdata[column] = mdata[column].fillna(mdata[column].mean())
mdata.build_year
# カテゴリデータ列の取得
cat_names = mdata.select_dtypes(include=object).columns.values
for cat_name in cat_names:
    print(f"cat_name:{cat_name} data:{mdata[cat_name].unique()}")
# nanが入った際にdtypesが上手く取れなかったので止む無し…
def check_categories(cat_names: list, datatype: str):
    """
      カテゴリ数の差異やnanデータの確認

      cat_names: list
        対象のカテゴリ名
      datatype: str
        対象のデータタイプ
          "cat": カテゴリ
          "num": 数値
    """
    has_nan_data = []
    is_one_hot = []

    for cat_name in cat_names:
        trainc = pd.Series(train[cat_name].unique())
        testc = pd.Series(test[cat_name].unique())
        mdatac = pd.Series(mdata[cat_name].unique())

        # nanがあるかを確認
        if mdatac.isnull().values.sum() > 0:
            has_nan_data.append("あり")
        else:
            has_nan_data.append("なし")

        # 項目数が等しいかを確認
        if datatype == 'cat':
            trainc = trainc.fillna('欠損').sort_values().reset_index(drop=True)
            testc = testc.fillna('欠損').sort_values().reset_index(drop=True)
        elif datatype == 'num':
            trainc = trainc.fillna(-9999).sort_values().reset_index(drop=True)
            testc = testc.fillna(-9999).sort_values().reset_index(drop=True)
        else:
            print("dtype = 'cat' or 'num'")
            return

        if trainc.equals(testc):
            is_one_hot.append("〇")
        else:
            is_one_hot.append("×")
 
    catinfo = pd.DataFrame({'カテゴリ名': cat_names,
                      'NANの有無': has_nan_data,
                      'カテゴリ数の一致': is_one_hot})
    display(catinfo)
check_categories(cat_names, "cat")
# 欠損カテゴリ埋め
mdata.product_type = mdata.product_type.fillna("no_data")
fig = plt.figure(figsize=(6.0*3, 6.0*cat_names.size))

for size in range(cat_names.size):
    ax_ = fig.add_subplot(cat_names.size, 3, size+1)
    column = cat_names[size]
    ax_.set_title(column)
    sns.countplot(mdata[column], ax=ax_)
ID_names = [x for x in mdata.columns.values if x.find('ID_') == 0]
ID_names
mdata.material = mdata.material.astype('object')
mdata.state = mdata.state.astype('object')
mdata[ID_names] = mdata[ID_names].astype('object')
print(mdata.material.unique())
print(mdata.material.value_counts())
mdata.material = mdata.material.fillna(0)
sns.countplot(mdata.material)
# 明らかに怪しい1点がある。これは3のうち間違えと判断
print(mdata.state.unique())
print(mdata.state.value_counts())
# 欠損あったので置換
mdata.state = mdata.state.fillna(0)
# 怪しいデータの確認
mdata[mdata.state == 33]
# state=11番目
print(f"変更前のカテゴリ値{mdata.iat[10089, 10]}")
mdata.iat[10089, 10] = 3
print(f"変更後のカテゴリ値{mdata.iat[10089, 10]}")
sns.countplot(mdata.state)
# 多すぎたのでただのエンコーディング処理にする
for ID_name in ID_names:
    print(f"names:{ID_name} value:{mdata[ID_name].unique()}")
check_categories(ID_names, "num")
# 徒歩での最寄りの駅名＝車での最寄りの駅名と同様にする
mdata.ID_railroad_station_walk = mdata.ID_railroad_station_walk.fillna(mdata.ID_railroad_station_avto)
mdata.build_year.describe()
# 1やら0やらnanやら適当なデータがあるので処理する
irr_num = 1000
mdata[mdata.build_year < irr_num].build_year.describe()
# 適当なデータ＋欠損値を置換
mdata.loc[mdata.build_year < irr_num, 'build_year'] = 9999
mdata.build_year = mdata.build_year.fillna(9999)

year_bins = [irr_num,1949,1959,1969,1979,1989,1999,2009,9999]
year_labels = ['under1950s','1950s','1960s','1970s','1980s','1990s', '2000s', 'unknown']
mdata['build_year_cat'] = pd.cut(mdata.build_year, bins=year_bins, labels=year_labels)
mdata = mdata.drop('build_year', axis=1)
sns.countplot(mdata.build_year_cat)
# カテゴリデータ列の取得
cat_names = mdata.select_dtypes(include=object).columns.values
for cat_name in cat_names:
    print(f"{cat_name} data:{mdata[cat_name].unique()}")
# ワンホットエンコーディングが出来そうなものはそうしておく
# 正直全部ラベルエンコーディングでいい気もするが…
dummy = pd.get_dummies(mdata.build_year_cat, prefix='build_year_cat', drop_first=True)
mdata = pd.concat([mdata, dummy], axis=1)
mdata = mdata.drop('build_year_cat', axis=1)

dummy = pd.get_dummies(mdata.ecology, prefix='ecology', drop_first=True)
mdata = pd.concat([mdata, dummy], axis=1)
mdata = mdata.drop('ecology', axis=1)
# 残りは脳死でラベルエンコーディング
cat_names = cat_names.tolist()
cat_names.remove('ecology')
le = LabelEncoder()
for cat_name in cat_names:
    mdata[cat_name] =  le.fit_transform(mdata[cat_name]).astype('int8')
mdata.head()
# 前処理後のデータ作成
new_train = mdata[0:train.shape[0]].reset_index(drop=True)
new_train = pd.concat([new_train, train.price_doc], axis=1)
new_test = mdata[train.shape[0]: ].reset_index(drop=True)
new_train.price_doc.describe()
sns.violinplot(new_train['price_doc'], jitter=True , dodge=True)
# 対数変換
sns.violinplot(np.log(new_train['price_doc']), jitter=True , dodge=True)
# box-cox変換
from sklearn.preprocessing import PowerTransformer
size = new_train.shape[0]
pt = PowerTransformer(method='box-cox')
box = pt.fit_transform(new_train['price_doc'].values.reshape(size,-1))
sns.violinplot(box, jitter=True , dodge=True)
attrs = [
        "year",
        "month"
    ]

for attr in attrs:
    dtype = np.int16 if attr == "year" else np.int8
    new_train[attr] = getattr(new_train.timestamp.dt, attr).astype(dtype)
    new_test[attr] = getattr(new_test.timestamp.dt, attr).astype(dtype)

new_train = new_train.drop(['id', 'timestamp'], axis=1)
new_test = new_test.drop(['id', 'timestamp'], axis=1)
new_train.head()
new_test.head()
test_vals = new_test.values.tolist()
datalist = []
for val in test_vals:
    datalist.append(val) 
import requests
import json


# URL for the web service
# 1st
#scoring_uri = 'http://a4b72a80-42eb-4e76-b6d7-13c9bc65dcc4.southeastasia.azurecontainer.io/score'
# 2nd
scoring_uri = 'http://9a309bca-d9ea-430e-9dc0-91907d51b6a2.southeastasia.azurecontainer.io/score'
# If the service is authenticated, set the key or token
#key = '<your key or token>'

# Two sets of data to score, so we get two results back
data = {"data":
         datalist
        }
# Convert to JSON string
input_data = json.dumps(data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
#headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
import json
d = json.loads(resp.json())
#d["result"]
result_submit = pd.DataFrame({'id': test.id.to_list(),
                   'price_doc': d["result"]})
result_submit
result_submit.to_csv('submission_2.csv', index=False)