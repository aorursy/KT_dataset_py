import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

import scipy.stats as stats

import xgboost as xgb

import datetime as dt

from IPython.display import display, HTML



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



macro = pd.read_csv('../input/macro.csv')

houses = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#pd.options.display.max_rows = 1000

#pd.options.display.max_columns = 1000



import warnings

warnings.filterwarnings('ignore')



def error(actual, predicted):

    return np.sqrt(np.sum(np.square(np.log1p(actual)-np.log1p(predicted)))/len(actual))



houses['timestampS'] = houses['timestamp']

houses['timestamp'] = pd.to_datetime(houses['timestamp'])
prev = pd.options.display.max_rows

pd.options.display.max_rows = 1000

display(pd.DataFrame({'houses column': houses.columns}))

display(pd.DataFrame({'macro column': macro.columns}))

display(pd.DataFrame({'district': houses['sub_area'].unique()}))

pd.options.display.max_rows = prev
numeric = houses.select_dtypes(exclude=['object', 'datetime'])

prices = numeric['price_doc']

correlations = list(map(lambda feature: prices.corr(numeric[feature], 'spearman'), numeric.columns))

display = pd.DataFrame({'feature': numeric.columns, 'correlation': correlations}).sort_values(by='correlation')

plt.figure(figsize=(8, 40))

plt.figure(1)

sb.barplot(data=display, orient='h', x='correlation', y='feature')



features = list(display['feature'].values)

correlatedTop = features[:20] + features[-20:]

plt.figure(2)

sb.heatmap(numeric[correlatedTop].corr())

correlatedTop2 = features[20:40] + features[-40:-20]

plt.figure(3)

sb.heatmap(numeric[correlatedTop2].corr())

correlatedTop3 = features[40:60] + features[-60:-40]

plt.figure(4)

sb.heatmap(numeric[correlatedTop3].corr())
pricesWithTimestamp = houses[['timestampS', 'price_doc']].groupby('timestampS').mean()

pricesWithTimestamp['timestampS'] = houses['timestampS'].unique()

macroWithPrices = macro.copy()

macroWithPrices = macroWithPrices.merge(pricesWithTimestamp, left_on='timestamp', right_on='timestampS')

#macroWithPrices['timestamp'] = pd.to_datetime(macroWithPrices['timestamp'])

macroWithPrices.drop('timestamp', axis=1, inplace=True)



prices = macroWithPrices['price_doc']

correlations = list(map(lambda feature: prices.corr(macroWithPrices[feature], 'spearman'), macroWithPrices.columns))

display = pd.DataFrame({'feature': macroWithPrices.columns, 'correlation': correlations}).sort_values(by='correlation')

plt.figure(figsize=(8, 40))

plt.figure(1)

sb.barplot(data=display, orient='h', x='correlation', y='feature')



features = list(display['feature'].values)

correlatedTop = features[:20] + features[-20:]

plt.figure(2)

sb.heatmap(macroWithPrices[correlatedTop].corr())

correlatedTop2 = features[20:40] + features[-40:-20]

plt.figure(3)

sb.heatmap(macroWithPrices[correlatedTop2].corr())

correlatedTop3 = features[40:60] + features[-60:-40]

plt.figure(4)

sb.heatmap(macroWithPrices[correlatedTop3].corr())
from dateutil.relativedelta import relativedelta

import datetime



start = datetime.date(2011, 8, 20)

end = datetime.date(2015, 6, 30)

periods = []

current = start

while current <= end:

    periods.append(current)

    current += relativedelta(months=1)



tmpHouses = houses.copy()

print(tmpHouses)

tmpHouses.set_index('timestamp', inplace=True)

tsDict = {}

tsDict['period'] = periods[1:]

for district in houses['sub_area'].unique():

    prices = []

    for i in range(1, len(periods)):

        monthMean = houses[(tmpHouses.index >= periods[i-1]) 

                           & (tmpHouses.index <= periods[i]) 

                           & (tmpHouses['sub_area'] == district)]['price_doc'].mean(skipna=True)

        prices.append(monthMean)

    tsDict[district] = prices



pricesTimeseries = pd.DataFrame(tsDict)

pricesTimeseries.set_index('period')
macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",

"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",

"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]



df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)



df_train.head()



ylog_train_all = np.log1p(df_train['price_doc'].values)

id_test = df_test['id']



df_train.drop(['id', 'price_doc'], axis=1, inplace=True)

df_test.drop(['id'], axis=1, inplace=True)



# Build df_all = (df_train+df_test).join(df_macro)

num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')

print(df_all.shape)





# Add month-year

month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)



# Add month and day-of-week

df_all['month'] = df_all.timestamp.dt.month

df_all['dow'] = df_all.timestamp.dt.dayofweek



# Other feature engineering

df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)

df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)



# Remove timestamp column (may overfit the model in train)

df_all.drop(['timestamp'], axis=1, inplace=True)



df_numeric = df_all.select_dtypes(exclude=['object'])

df_obj = df_all.select_dtypes(include=['object']).copy()



for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]



df_values = pd.concat([df_numeric, df_obj], axis=1)



X_all = df_values.values

print(X_all.shape)



# Create a validation set, with last 20% of data

num_val = int(num_train * 0.2)



X_train_all = X_all[:num_train]

X_train = X_all[:num_train-num_val]

X_val = X_all[num_train-num_val:num_train]

ylog_train = ylog_train_all[:-num_val]

ylog_val = ylog_train_all[-num_val:]



X_test = X_all[num_train:]



df_columns = df_values.columns



dtrain_all = xgb.DMatrix(X_train_all, ylog_train_all, feature_names=df_columns)

dtrain = xgb.DMatrix(X_train, ylog_train, feature_names=df_columns)

dval = xgb.DMatrix(X_val, ylog_val, feature_names=df_columns)

dtest = xgb.DMatrix(X_test, feature_names=df_columns)



xgb_params = {

    'eta': 0.05,

    'max_depth': 6,

    'subsample': 1.0,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



# Uncomment to tune XGB `num_boost_rounds`

partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],

                       early_stopping_rounds=20, verbose_eval=20)



num_boost_round = partial_model.best_iteration



model = xgb.train(dict(xgb_params, silent=0), dtrain_all, num_boost_round=num_boost_round)



ylog_pred = model.predict(dtrain_all)

y_pred = np.exp(ylog_pred) - 1

print(error(y_pred, np.exp(ylog_train_all)))