# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# https://www.kaggle.com/girmdshinsei/for-japanese-beginner-with-wrmsse-in-lgbm

import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

import seaborn as sns; sns.set()

import lightgbm as lgb

from sklearn import preprocessing, metrics

from sklearn.preprocessing import LabelEncoder

import gc

import os

from scipy.sparse import csr_matrix

from scipy.stats import gamma



for dirname, _, filenames in os.walk('./input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns: #columns毎に処理

        col_type = df[col].dtypes

        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
def read_data():

    print('Reading files...')

    calendar = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/calendar.csv')

    calendar = reduce_mem_usage(calendar)

    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))

    #

    sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv')

    sell_prices = reduce_mem_usage(sell_prices)

    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))

    #

    sales_train_val = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sales_train_validation.csv')

    print(

        'Sales train validation has {} rows and {} columns'.format(sales_train_val.shape[0], sales_train_val.shape[1]))

    #

    submission = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sample_submission.csv')

    #

    return calendar, sell_prices, sales_train_val, submission
calendar, sell_prices, sales_train_val, submission = read_data()
NUM_ITEMS = sales_train_val.shape[0]  # 30490

DAYS_PRED = submission.shape[1] - 1  # 28
def encode_categorical(df, cols):

    for col in cols:

        # Leave NaN as it is.

        le = LabelEncoder()

        not_null = df[col][df[col].notnull()]

        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)

    #

    return df
calendar = encode_categorical(

    calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]

).pipe(reduce_mem_usage)



sales_train_val = encode_categorical(

    sales_train_val, ["item_id", "dept_id", "cat_id", "store_id", "state_id"],

).pipe(reduce_mem_usage)



sell_prices = encode_categorical(sell_prices, ["item_id", "store_id"]).pipe(

    reduce_mem_usage

)
nrows = 365 * 1 * NUM_ITEMS
sales_train_val = pd.melt(sales_train_val,

                                     id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],

                                     var_name = 'day', value_name = 'demand')

print('Melted sales train validation has {} rows and {} columns'.format(sales_train_val.shape[0],

                                                                            sales_train_val.shape[1]))
sales_train_val = sales_train_val.iloc[-nrows:,:]
# separate test dataframes

forecast_submission = pd.concat([submission.iloc[0:int(771120/2),:].iloc[-30490:,:],submission.iloc[int(771120/2):,:].iloc[-30490:,:]])

forecast_submission['id'] = forecast_submission['id'].str.replace('_.\...._','_')

forecast_submission.drop_duplicates(inplace=True)



# submission fileのidのvalidation部分と, ealuation部分の名前を取得

test1_rows = [row for row in forecast_submission['id'] if 'validation' in row]

test2_rows = [row for row in forecast_submission['id'] if 'evaluation' in row]



# submission fileのvalidation部分をtest1, ealuation部分をtest2として取得

test1 = forecast_submission[forecast_submission['id'].isin(test1_rows)]

test2 = forecast_submission[forecast_submission['id'].isin(test2_rows)]



# test1, test2の列名の"F_X"の箇所をd_XXX"の形式に変更

test1.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + DAYS_PRED)]

test2.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + DAYS_PRED)]



# test2のidの'_evaluation'を置換

#test1['id'] = test1['id'].str.replace('_validation','')

test2['id'] = test2['id'].str.replace('_evaluation','_validation')



# sales_train_valからidの詳細部分(itemやdepartmentなどのid)を重複なく一意に取得。

product = sales_train_val[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()



# idをキーにして, idの詳細部分をtest1, test2に結合する.

test1 = test1.merge(product, how = 'left', on = 'id')

test2 = test2.merge(product, how = 'left', on = 'id')



# test1, test2をともにmelt処理する.（売上数量:demandは0）

test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 

                var_name = 'day', value_name = 'demand')



test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 

                var_name = 'day', value_name = 'demand')



# validation部分と, evaluation部分がわかるようにpartという列を作り、 test1,test2のラベルを付ける。

sales_train_val['part'] = 'train'

test1['part'] = 'test1'

test2['part'] = 'test2'



# sales_train_valとtest1, test2の縦結合.

data = pd.concat([sales_train_val, test1, test2], axis = 0)



# memoryの開放

del test1, test2, sales_train_val, forecast_submission



# delete test2 for now(6/1以前は, validation部分のみ提出のため.)

data = data[data['part'] != 'test2']



gc.collect()
#calendarの結合

# drop some calendar features(不要な変数の削除:weekdayやwdayなどはdatetime変数から後ほど作成できる。)

calendar.drop(['weekday', 'wday', 'month', 'year'],

              inplace = True, axis = 1)



# notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)(dayとdをキーにdataに結合)

data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])

data.drop(['d', 'day'], inplace = True, axis = 1)



# memoryの開放

del  calendar

gc.collect()



#sell priceの結合

# get the sell price data (this feature should be very important)

data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')

print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))



# memoryの開放

del  sell_prices

gc.collect()
def simple_fe(data):

    # demand features(過去の数量から変数生成)

    #

    for diff in [0, 1, 2]:

        shift = DAYS_PRED + diff

        data[f"shift_t{shift}"] = data.groupby(["id"])["demand"].transform(

            lambda x: x.shift(shift)

        )

    #

    for size in [7, 30, 60, 90, 180]:

        data[f"rolling_std_t{size}"] = data.groupby(["id"])["demand"].transform(

            lambda x: x.shift(DAYS_PRED).rolling(size).std()

        )

    #

    for size in [7, 30, 60, 90, 180]:

        data[f"rolling_mean_t{size}"] = data.groupby(["id"])["demand"].transform(

            lambda x: x.shift(DAYS_PRED).rolling(size).mean()

        )

    #

    data["rolling_skew_t30"] = data.groupby(["id"])["demand"].transform(

        lambda x: x.shift(DAYS_PRED).rolling(30).skew()

    )

    data["rolling_kurt_t30"] = data.groupby(["id"])["demand"].transform(

        lambda x: x.shift(DAYS_PRED).rolling(30).kurt()

    )

    #

    # price features

    # priceの動きと特徴量化（価格の変化率、過去1年間の最大価格との比など）

    #

    data["shift_price_t1"] = data.groupby(["id"])["sell_price"].transform(

        lambda x: x.shift(1)

    )

    data["price_change_t1"] = (data["shift_price_t1"] - data["sell_price"]) / (

        data["shift_price_t1"]

    )

    data["rolling_price_max_t365"] = data.groupby(["id"])["sell_price"].transform(

        lambda x: x.shift(1).rolling(365).max()

    )

    data["price_change_t365"] = (data["rolling_price_max_t365"] - data["sell_price"]) / (

        data["rolling_price_max_t365"]

    )

    #

    data["rolling_price_std_t7"] = data.groupby(["id"])["sell_price"].transform(

        lambda x: x.rolling(7).std()

    )

    data["rolling_price_std_t30"] = data.groupby(["id"])["sell_price"].transform(

        lambda x: x.rolling(30).std()

    )

    #

    # time features

    # 日付に関するデータ

    dt_col = "date"

    data[dt_col] = pd.to_datetime(data[dt_col])

    #

    attrs = [

        "year",

        "quarter",

        "month",

        "week",

        "day",

        "dayofweek",

        "is_year_end",

        "is_year_start",

        "is_quarter_end",

        "is_quarter_start",

        "is_month_end",

        "is_month_start",

    ]

    #

    for attr in attrs:

        dtype = np.int16 if attr == "year" else np.int8

        data[attr] = getattr(data[dt_col].dt, attr).astype(dtype)

    #

    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(np.int8)

    #

    return data
data = simple_fe(data)

data = reduce_mem_usage(data)
# going to evaluate with the last 28 days

x_train = data[data['date'] <= '2016-03-27']

y_train = x_train['demand']

x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]

y_val = x_val['demand']

test = data[(data['date'] > '2016-04-24')]
features = [

    "item_id",

    "dept_id",

    "cat_id",

    "store_id",

    "state_id",

    "event_name_1",

    "event_type_1",

    "event_name_2",

    "event_type_2",

    "snap_CA",

    "snap_TX",

    "snap_WI",

    "sell_price",

    # demand features.

    "shift_t28",

    "shift_t29",

    "shift_t30",

    "rolling_std_t7",

    "rolling_std_t30",

    "rolling_std_t60",

    "rolling_std_t90",

    "rolling_std_t180",

    "rolling_mean_t7",

    "rolling_mean_t30",

    "rolling_mean_t60",

    "rolling_mean_t90",

    "rolling_mean_t180",

    "rolling_skew_t30",

    "rolling_kurt_t30",

    # price features

    "price_change_t1",

    "price_change_t365",

    "rolling_price_std_t7",

    "rolling_price_std_t30",

    # time features.

    "year",

    "month",

    "week",

    "day",

    "dayofweek",

    "is_year_end",

    "is_year_start",

    "is_quarter_end",

    "is_quarter_start",

    "is_month_end",

    "is_month_start",

    "is_weekend"

]
weight_mat = np.c_[np.identity(NUM_ITEMS).astype(np.int8),  # item :level 12

                   np.ones([NUM_ITEMS, 1]).astype(np.int8),  # level 1

                   pd.get_dummies(product.state_id.astype(str), drop_first=False).astype('int8').values,

                   pd.get_dummies(product.store_id.astype(str), drop_first=False).astype('int8').values,

                   pd.get_dummies(product.cat_id.astype(str), drop_first=False).astype('int8').values,

                   pd.get_dummies(product.dept_id.astype(str), drop_first=False).astype('int8').values,

                   pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str), drop_first=False).astype(

                       'int8').values,

                   pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str), drop_first=False).astype(

                       'int8').values,

                   pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str), drop_first=False).astype(

                       'int8').values,

                   pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str), drop_first=False).astype(

                       'int8').values,

                   pd.get_dummies(product.item_id.astype(str), drop_first=False).astype('int8').values,

                   pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str), drop_first=False).astype(

                       'int8').values

].T



weight_mat_csr = csr_matrix(weight_mat)

del weight_mat

gc.collect()



def weight_calc(data):

    # calculate the denominator of RMSSE, and calculate the weight base on sales amount

    #

    sales_train_val = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sales_train_validation.csv')

    #

    d_name = ['d_' + str(i + 1) for i in range(1913)]

    #

    sales_train_val = weight_mat_csr * sales_train_val[d_name].values

    #

    # calculate the start position(first non-zero demand observed date) for each item / 商品の最初の売上日

    # 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算

    df_tmp = ((sales_train_val > 0) * np.tile(np.arange(1, 1914), (weight_mat_csr.shape[0], 1)))

    #

    start_no = np.min(np.where(df_tmp == 0, 9999, df_tmp), axis=1) - 1

    #

    # denominator of RMSSE / RMSSEの分母

    weight1 = np.sum((np.diff(sales_train_val, axis=1) ** 2), axis=1) / (1913 - start_no)

    #

    # calculate the sales amount for each item/level

    df_tmp = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]

    df_tmp['amount'] = df_tmp['demand'] * df_tmp['sell_price']

    df_tmp = df_tmp.groupby(['id'])['amount'].apply(np.sum).values

    #

    weight2 = weight_mat_csr * df_tmp

    #

    weight2 = weight2 / np.sum(weight2)

    #

    del sales_train_val

    gc.collect()

    #

    return weight1, weight2





weight1, weight2 = weight_calc(data)





def wrmsse(preds, data):

    # actual obserbed values / 正解ラベル

    y_true = np.array(data.get_label())

    #

    # number of columns

    num_col = len(y_true) // NUM_ITEMS

    #

    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す

    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T

    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T

    #

    x_name = ['pred_' + str(i) for i in range(num_col)]

    x_name2 = ["act_" + str(i) for i in range(num_col)]

    #

    train = np.array(weight_mat_csr * np.c_[reshaped_preds, reshaped_true])

    #

    score = np.sum(

        np.sqrt(

            np.mean(

                np.square(

                    train[:, :num_col] - train[:, num_col:])

                , axis=1) / weight1) * weight2)

    #

    return 'wrmsse', score, False





def wrmsse_simple(preds, data):

    # actual obserbed values / 正解ラベル

    y_true = np.array(data.get_label())

    #

    # number of columns

    num_col = len(y_true) // NUM_ITEMS

    #

    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す

    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T

    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T

    #

    train = np.c_[reshaped_preds, reshaped_true]

    #

    weight2_2 = weight2[:NUM_ITEMS]

    weight2_2 = weight2_2 / np.sum(weight2_2)

    #

    score = np.sum(

        np.sqrt(

            np.mean(

                np.square(

                    train[:, :num_col] - train[:, num_col:])

                , axis=1) / weight1[:NUM_ITEMS]) * weight2_2)

    #

    return 'wrmsse', score, False
params = {

    'boosting_type': 'gbdt',

    'metric': 'custom',

    'objective': 'regression',

    'n_jobs': -1,

    'seed': 236,

    'learning_rate': 0.1,

    'bagging_fraction': 0.75,

    'bagging_freq': 10,

    'colsample_bytree': 0.75}
train_set = lgb.Dataset(x_train[features], y_train)

val_set = lgb.Dataset(x_val[features], y_val)



del x_train, y_train

gc.collect()
model = lgb.train(params, train_set, num_boost_round = 2500, early_stopping_rounds = 50,

                  valid_sets = [train_set, val_set], verbose_eval = 100, feval= wrmsse)
# predictions for the mean of the uncertainty distribution

y_pred = model.predict(test[features])
# treat each prediction y as the estimated mean of a gamma distribution with mean y and scale 1

# and use ppf to obtain the corresponding estimates of the percentiles

ids = test['id'].unique()

theta=1.0

y_pred_q = pd.DataFrame.from_dict({'id': np.repeat(ids,28), 

                               'date': np.tile(np.arange(np.datetime64("2016-04-25"),np.datetime64("2016-05-23")),30490),

                               '0_005': gamma.ppf(0.005, y_pred, theta),

                               '0_025': gamma.ppf(0.025, y_pred, theta),

                               '0_165': gamma.ppf(0.165, y_pred, theta),

                               '0_250': gamma.ppf(0.25, y_pred, theta),

                               '0_500': gamma.ppf(0.5, y_pred, theta),

                               '0_750': gamma.ppf(0.75, y_pred, theta),

                               '0_835': gamma.ppf(0.835, y_pred, theta),

                               '0_975': gamma.ppf(0.975, y_pred, theta),

                               '0_995': gamma.ppf(0.995, y_pred, theta)

                            })

y_pred_q['item_id'] = y_pred_q['id'].map(lambda x: '_'.join(x.split('_')[0:3]))

y_pred_q['dept_id'] = y_pred_q['id'].map(lambda x: '_'.join(x.split('_')[0:2]))

y_pred_q['cat_id'] = y_pred_q['id'].map(lambda x: '_'.join(x.split('_')[0:1]))

y_pred_q['store_id'] = y_pred_q['id'].map(lambda x: '_'.join(x.split('_')[3:5]))

y_pred_q['state_id'] = y_pred_q['id'].map(lambda x: '_'.join(x.split('_')[3:4]))
def agg_series(preds12, q):

    # preds12 contains 30490 series at level 12 aggregate these to get the other series

    preds12['id'] = preds12['item_id'] + "_" + preds12['store_id'] + "_" + q.replace('_','.') + "_validation"

    # level 11: Unit sales of product x, aggregated for each State: 9,147

    preds11 = preds12.groupby(['date','item_id','state_id'], as_index=False)[q].sum()

    preds11['id'] = preds11['state_id'] + "_" + preds11['item_id'] + "_" + q.replace('_','.') + "_validation"

    # level 10: Unit sales of product x, aggregated for all stores/states: 3,049

    preds10 = preds11.groupby(['date','item_id'], as_index=False)[q].sum()

    preds10['id'] = preds10['item_id'] + "_X_" + q.replace('_','.') + "_validation"

    # level 9: Unit sales of all products, aggregated for each store and department: 70

    preds09 = preds12.groupby(['date', 'store_id', 'dept_id'], as_index=False)[q].sum()

    preds09['id'] = preds09['store_id'] + "_" + preds09['dept_id'] + "_" + q.replace('_','.') + "_validation"

    # level 8: Unit sales of all products, aggregated for each store and category: 30

    preds08 = preds12.groupby(['date', 'store_id', 'cat_id'], as_index=False)[q].sum()

    preds08['id'] = preds08['store_id'] + "_" + preds08['cat_id'] + "_" + q.replace('_','.') + "_validation"

    # level 7: Unit sales of all products, aggregated for each State and department: 21

    preds07 = preds12.groupby(['date', 'state_id', 'dept_id'], as_index=False)[q].sum()

    preds07['id'] = preds07['state_id'] + "_" + preds07['dept_id'] + "_" + q.replace('_','.') + "_validation"

    # level 6: Unit sales of all products, aggregated for each State and category: 9

    preds06 = preds12.groupby(['date', 'state_id', 'cat_id'], as_index=False)[q].sum()

    preds06['id'] = preds06['state_id'] + "_" + preds06['cat_id'] + "_" + q.replace('_','.') + "_validation"

    # level 5: Unit sales of all products, aggregated for each department: 7

    preds05 = preds12.groupby(['date', 'dept_id'], as_index=False)[q].sum()

    preds05['id'] = preds05['dept_id'] + "_X_" + q.replace('_','.') + "_validation"

    # level 4: Unit sales of all products, aggregated for each category: 3

    preds04 = preds12.groupby(['date', 'cat_id'], as_index=False)[q].sum()

    preds04['id'] = preds04['cat_id'] + "_X_" + q.replace('_','.') + "_validation"

    # level 3: Unit sales of all products, aggregated for each store: 10

    preds03 = preds12.groupby(['date', 'store_id'], as_index=False)[q].sum()

    preds03['id'] = preds03['store_id'] + "_X_" + q.replace('_','.') + "_validation"

    # level 2: Unit sales of all products, aggregated for each State: 3

    preds02 = preds12.groupby(['date', 'state_id'], as_index=False)[q].sum()

    preds02['id'] = preds02['state_id'] + "_X_" + q.replace('_','.') + "_validation"

    # level 1: Unit sales of all products, aggregated for all stores/states: 1

    preds01 = preds12.groupby(['date'], as_index=False)[q].sum()

    preds01['id'] = 'Total_X_' + q.replace('_','.') + '_validation'

    preds = pd.concat([preds01, preds02, preds03,

                                        preds04, preds05, preds06,

                                        preds07, preds08, preds09,

                                        preds10, preds11, preds12],

                                       ignore_index=True)

    return preds
predictions_0_005 = agg_series(y_pred_q,'0_005')[['date', 'id', '0_005']]

predictions_0_005 = predictions_0_005.pivot(index = 'id', columns = 'date', values = '0_005').reset_index()

predictions_0_005.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]



predictions_0_025 = agg_series(y_pred_q,'0_025')[['date', 'id', '0_025']]

predictions_0_025 = predictions_0_025.pivot(index = 'id', columns = 'date', values = '0_025').reset_index()

predictions_0_025.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]



predictions_0_165 = agg_series(y_pred_q,'0_165')[['date', 'id', '0_165']]

predictions_0_165 = predictions_0_165.pivot(index = 'id', columns = 'date', values = '0_165').reset_index()

predictions_0_165.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]



predictions_0_250 = agg_series(y_pred_q,'0_250')[['date', 'id', '0_250']]

predictions_0_250 = predictions_0_250.pivot(index = 'id', columns = 'date', values = '0_250').reset_index()

predictions_0_250.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]



predictions_0_500 = agg_series(y_pred_q,'0_500')[['date', 'id', '0_500']]

predictions_0_500 = predictions_0_500.pivot(index = 'id', columns = 'date', values = '0_500').reset_index()

predictions_0_500.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]



predictions_0_750 = agg_series(y_pred_q,'0_750')[['date', 'id', '0_750']]

predictions_0_750 = predictions_0_750.pivot(index = 'id', columns = 'date', values = '0_750').reset_index()

predictions_0_750.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]



predictions_0_835 = agg_series(y_pred_q,'0_835')[['date', 'id', '0_835']]

predictions_0_835 = predictions_0_835.pivot(index = 'id', columns = 'date', values = '0_835').reset_index()

predictions_0_835.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]



predictions_0_975 = agg_series(y_pred_q,'0_975')[['date', 'id', '0_975']]

predictions_0_975 = predictions_0_975.pivot(index = 'id', columns = 'date', values = '0_975').reset_index()

predictions_0_975.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]



predictions_0_995 = agg_series(y_pred_q,'0_995')[['date', 'id', '0_995']]

predictions_0_995 = predictions_0_995.pivot(index = 'id', columns = 'date', values = '0_995').reset_index()

predictions_0_995.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
predictions = pd.concat([predictions_0_005, predictions_0_025, predictions_0_165, predictions_0_250, predictions_0_500

    , predictions_0_750, predictions_0_835, predictions_0_975, predictions_0_995])

predictions = pd.concat([predictions, predictions])

predictions['id'][-385560:] = predictions['id'][-385560:].str.replace('_validation','_evaluation')

predictions.fillna(0,inplace=True)

predictions.to_csv("submission.csv",index=False)