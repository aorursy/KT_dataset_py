import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import datetime
import gc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.stats import boxcox
from scipy import stats
import numpy as np
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
train = pd.read_csv("../input/train.csv")
train = reduce_mem_usage(train)
test = pd.read_csv("../input/test.csv")
test = reduce_mem_usage(test)
print("The deimesnsions for test are", test.shape)
print(train.shape)
historical_transactions = pd.read_csv("../input/historical_transactions.csv")
historical_transactions = reduce_mem_usage(historical_transactions)
historical_transactions = historical_transactions.sample(frac= 0.2, replace=False)
historical_transactions['category_3'] = historical_transactions['category_3'].fillna(
                                            historical_transactions['category_3'].mode()[0])
historical_transactions['category_2'] = historical_transactions['category_2'].fillna(
                                            historical_transactions['category_2'].mode()[0])
historical_transactions['merchant_id'] = historical_transactions['merchant_id'].fillna(
                                            historical_transactions['merchant_id'].mode()[0])
merchants = pd.read_csv("../input/merchants.csv")
merchants = reduce_mem_usage(merchants)
new_merchant_transactions = pd.read_csv("../input/new_merchant_transactions.csv")
new_merchant_transactions = reduce_mem_usage(new_merchant_transactions)
new_merchant_transactions['category_3'] = new_merchant_transactions['category_3'].fillna(
                                            new_merchant_transactions['category_3'].mode()[0])
new_merchant_transactions['category_2'] = new_merchant_transactions['category_2'].fillna(
                                            new_merchant_transactions['category_2'].mode()[0])
new_merchant_transactions['merchant_id'] = new_merchant_transactions['merchant_id'].fillna(
                                            new_merchant_transactions['merchant_id'].mode()[0])

from sympy import log
plt.hist(historical_transactions['month_lag'], range=[-15, 0.025], align='mid')
train1 =  historical_transactions[['purchase_amount', 'month_lag', 'installments']]
#['purchase_amount = **0.5', 'month_lag', 'installments']
train1['purchase_amount'] = (train1['purchase_amount'])**0.5
print((train1['purchase_amount']).corr(train['target']))
print((historical_transactions['purchase_amount']).corr(train['target']))
#plt.hist(train1['purchase_amount'], range=[0, 1], facecolor='gray', align='mid')
for df in [historical_transactions,new_merchant_transactions]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0})
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']
def create_features(df1, df2, column1,variables, statistics):
    start = 0
    mydict_names= {"card_id":"card_id"}
    mydict_stats = {}
    if len(variables):
        for i in statistics:
            count = 1
            mynames = list(df1)
            if not variables[0] + '_' + i in mynames:
                mydict_names[variables[0]] = variables[0] + '_' + i
            else:
                mydict_names[variables[0]] = variables[0] + '_' + i + str(count)
            mydict_stats[variables[0]] = i
            if i != "mode":
                df3 = (df2.groupby('card_id', as_index=False).agg(mydict_stats).rename
                       (columns=mydict_names))
            else:
                df3 = (df2.groupby('card_id',as_index=False).agg(
                       lambda x: stats.mode(x)[0][0]).rename(columns=mydict_names))
            df1 = pd.merge(df1, df3, on='card_id', how='left')
        del variables[0]
        return create_features(df1, df2, column1,variables, statistics)
    else:
        print(list(df1))
        del statistics
        return df1
historical_transactions['purchase_amount'] = (historical_transactions['purchase_amount'])**0.5
variables = ["purchase_amount", "month_lag", "installments", 'month_diff']
statistics = ["sum", "mean", "max", "min", "var", "median"]
train = create_features(train, historical_transactions, "card_id", variables, statistics)
variables = ["purchase_amount", "month_lag", "installments", 'month_diff']
statistics = ["sum", "mean", "max", "min", "var", "median"]
train = create_features(train, new_merchant_transactions, "card_id", variables, statistics)
variables = ["purchase_amount", "month_lag", "installments", "month_diff"]
statistics = ["sum", "mean", "max", "min", "var", "median"]
test = create_features(test, historical_transactions, "card_id", variables, statistics)
variables = ["purchase_amount", "month_lag", "installments", "month_diff"]
statistics = ["sum", "mean", "max", "min", "var", "median"]
test = create_features(test, new_merchant_transactions, "card_id", variables, statistics)
def drop_columns(df, columns):
    try:
        for i in columns:
            df.drop(str(i), axis=1, inplace=True)
    except KeyError:
        print("column named ", i, " is missing in the data frame")
variables = ["purchase_date"]
statistics = ["max", "min"]
train = create_features(train, historical_transactions, "card_id", variables, statistics)
variables = ["purchase_date"]
statistics = ["max", "min"]
test = create_features(test, historical_transactions, "card_id", variables, statistics)
variables = ["subsector_id", "city_id", "state_id"]
def recode_variables(df, variables):
    for var in variables:
        df[var] = pd.factorize(df[var])[0] + 1
        #df = pd.get_dummies(df, columns=[str(var)])
recode_variables(historical_transactions, variables)
print("done")
def convert_dates(df, converts):
    for i in converts:
        df[i] = pd.to_datetime(df[i])
convert_dates(train, ["first_active_month"])
convert_dates(test, ["first_active_month"])
convert_dates(historical_transactions, ["purchase_date"])
holidays = ["1 Jan", "22 Feb","23 Feb","24 Feb","25 Feb","26 Feb","20 Mar","10 Apr","12 Apr",
            "21 Apr","1 May","11 Jun","12 Jun","20 Jun","9 Aug","7 Sep","22 Sep","12 Oct",
            "15 Oct","28 Oct","2 Nov","15 Nov","20 Nov","21 Dec","24 Dec","25 Dec","31 Dec"]
holidays = [datetime.datetime.strptime(i,'%d %b') for i in holidays]
def create_date_features(df, columns, holidays=[]):
    df[column + "month"] = df[column].dt.month
    df[column + "purchase_time"]=pd.cut(df[column].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening'])
    df[column + "week_day"] = df[column].dt.day_name() #  Not important feature
    print("done")   
convert_dates(train, ["purchase_date_min", "purchase_date_max"])
convert_dates(test, ["purchase_date_min", "purchase_date_max"])
def convert_deltas(df, converts):
    for i in converts:
        df[i] =  pd.to_datetime(df[i], format='%Y%d%b:%H:%M:%S.%f')
convert_deltas(train, ["purchase_date_min", "purchase_date_max"])
convert_deltas(test, ["purchase_date_min", "purchase_date_max"])
train['max_date_diff'] = (train['purchase_date_max'] - train['first_active_month']).dt.days
train['min_date_diff'] = (train['purchase_date_min'] - train['first_active_month']).dt.days
test['max_date_diff'] = (test['purchase_date_max'] - test['first_active_month']).dt.days
test['min_date_diff'] = (test['purchase_date_min'] - test['first_active_month']).dt.days
train['purchase_date_max'] = train['purchase_date_max'].astype(int)
train['purchase_date_min'] = train['purchase_date_min'].astype(int)
test['purchase_date_max'] = test['purchase_date_max'].astype(int)
test['purchase_date_min'] = test['purchase_date_min'].astype(int)
avoid = ['month_lag_sum', 'month_lag_mean', 'month_lag_max', 'month_lag_min', 'month_lag_var',
         'month_lag_median', 'month_diff_sum', 'month_diff_mean', 'month_diff_max', 'month_diff_min',
         'month_diff_var', 'month_diff_median', 'month_lag_sum_1', 'month_lag_mean_1', 
         'month_lag_max_1', 'month_lag_min_1', 'month_lag_var_1','month_lag_median_1',
         'month_diff_sum_1', 'month_diff_mean_1', 'month_diff_max_1', 'month_diff_min_1',
         'month_diff_var_1', 'month_diff_median_1'     
]
def create_categories(df,lengths = []):
    for i in list(df):
        if len(df[str(i)].value_counts()) < 100 and i not in avoid:    
            lengths.append(i)
    for col in lengths:
        df.sort_values(by=[col])
        df[col] = df[col].astype('category')
    print("done")   
#create_categories(train)
#create_categories(test)
def drop_correlated_feat(df):
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    return df
train = drop_correlated_feat(train)
test = drop_correlated_feat(test)
print("done")
columns_to_drop = ['card_id', 'first_active_month']
drop_columns(train, columns_to_drop)
drop_columns(test, columns_to_drop)
target = train["target"]
train.drop('target',  axis=1, inplace=True)
print("done")
X_train, X_test, y_train, y_test = train_test_split(train,
                                                    target,
                                                    test_size=0.2,
                                                    shuffle = True)
#del train
#del target
gc.collect()
import lightgbm as lgb
cats = list(X_train.select_dtypes(['category']))
d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cats, free_raw_data=False)
d_test = lgb.Dataset(X_test, label=y_test, categorical_feature=cats, free_raw_data=False)
params = {'num_leaves': round(0.6*(pow(2,9))),
         'min_data_in_leaf': 149, 
         'objective':'regression',
         'max_depth': 9,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2634,
         "random_state": 1033,
         "seed": 42,
         "verbosity": -1}
model = lgb.train(params, d_train, 1000, valid_sets=[d_test], early_stopping_rounds=100, verbose_eval=100)
fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()
["purchase_amount_max1", "month_diff_mean", "month_lag_var", "purchase_date_max", "min_date_diff",
]
y_pred = model.predict(test, num_iteration=model.best_iteration)
submission = pd.read_csv("../input/test.csv")
submission = pd.DataFrame({"card_id":submission["card_id"].values})
submission["target"] = y_pred
submission.to_csv("submission.csv", index=False)