import pandas as pd

import numpy as np

import datetime

from dateutil.relativedelta import relativedelta



import xgboost as xgb

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import mean_squared_error



import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
df = pd.read_csv("../input/electricity/train_electricity.csv")

df_test = pd.read_csv("../input/electricity/test_electricity.csv")



print("Dataset has", len(df), "entries.")



print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")

for col_name in df.columns:

    col = df[col_name]

    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")
## 2. Adding some datetime related features



def add_datetime_features(df):

    features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",

                "Hour", "Minute",] # ,"Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start"

    one_hot_features = []#["Month", "Dayofweek"]



    datetime = pd.to_datetime(df.Date * (10 ** 9))



    df['Datetime'] = datetime  # We won't use this for training, but we'll remove it later



    for feature in features:

        new_column = getattr(datetime.dt, feature.lower())

        if feature in one_hot_features:

            df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)

        df[feature] = new_column

    return df



df = add_datetime_features(df)

df_test = add_datetime_features(df_test)

df.columns
plt.plot(df['Consumption_MW'])

plt.ylim(3000, 11000)
plt.plot(df.groupby(['Year', 'Month'])['Consumption_MW'].median().reset_index(drop=True))

plt.ylim(3000, 11000)

for l in range(9):

    plt.vlines(l*12, 0, 15000, linestyles='--', color='grey')
df_full = pd.concat([df, df_test]).reset_index(drop=True)

df_full.shape
def killer(df):

    start = df['Datetime'].min() + relativedelta(months=12) + relativedelta(hours=24*15+1)

    periods = [16, 60, 60*12, 60*24*15]

    target = df['Consumption_MW']

    for n_minutes in periods:

        lst = []

        for dt in tqdm(df.loc[df['Datetime'] > start, 'Datetime'].tolist()):

            mask = ((df['Datetime'] <= dt + relativedelta(months=-12) + relativedelta(minutes=n_minutes)) & 

                    (df['Datetime'] >= dt + relativedelta(months=-12) + relativedelta(minutes=-n_minutes)))

            lst.append(target.loc[mask].median())

        df.loc[df['Datetime'] > start, 'killer_' + str(n_minutes)] = lst

        df.loc[df['Datetime'] > start, 'killer_' + str(n_minutes)] = df.loc[df['Datetime'] > start, 'killer_' +

                                                                         str(n_minutes)].fillna(target.mean())

    return df        
# # # killer(df.iloc[:472806 - 419995 + 1000].copy())



# df_full = killer(df_full)

# # df = df_full.loc[df.index].dropna()

# # df_test = df_full.loc[~df_full.index.isin(df.index)].reset_index(drop=True).drop(columns=['Consumption_MW'])



# df_full.to_csv('train_time_killers_full_y2.csv')
# # # df_init = pd.read_csv('train_time_killer.csv', index_col='Unnamed: 0')

# # # df_init['Datetime'] = pd.to_datetime(df_init['Datetime'])

# # # df = df_init.dropna()



# df_full = pd.read_csv('train_time_killers_full.csv', index_col='Unnamed: 0')

df_full = pd.read_csv('../input/electricityfull/train_time_killers_full.csv', index_col='Unnamed: 0')

# df_full2 = pd.read_csv('../input/electricityfull/train_time_killers_full_y2.csv', index_col='Unnamed: 0')



# df_full = df_full.merge(df_full2[['killer_16_y2', 'killer_60_y2', 'killer_720_y2', 'killer_21600_y2']], 

#                         how='inner', left_index=True, right_index=True)

# del df_full2



# df_full['killer_16_y2'].fillna(df_full['killer_16'], inplace=True)

# df_full['killer_60_y2'].fillna(df_full['killer_60'], inplace=True)

# df_full['killer_720_y2'].fillna(df_full['killer_720'], inplace=True)

# df_full['killer_21600_y2'].fillna(df_full['killer_21600'], inplace=True)



# df_full['killer_16'] = df_full['killer_16'] / 2 + df_full['killer_16_y2'] / 2

# df_full['killer_60'] = df_full['killer_60'] / 2 + df_full['killer_60_y2'] / 2

# df_full['killer_720'] = df_full['killer_720'] / 2 + df_full['killer_720_y2'] / 2

# df_full['killer_21600'] = df_full['killer_21600'] / 2 + df_full['killer_21600_y2'] / 2

# df_full.drop(columns=['killer_16_y2', 'killer_60_y2', 'killer_720_y2', 'killer_21600_y2'], inplace=True)



df_full['Datetime'] = pd.to_datetime(df_full['Datetime'])

df = df_full.loc[df.index]

df_test = df_full.loc[~df_full.index.isin(df.index)].reset_index(drop=True).drop(columns=['Consumption_MW'])

df.shape, df_test.shape
df['DatetimeRound'] = df.set_index('Datetime').index.round('H')

df_test['DatetimeRound'] = df_test.set_index('Datetime').index.round('H')
weather = pd.read_csv('../input/weather/weather.csv', index_col='Unnamed: 0')

weather = weather[['Local time in Bucharest / Filaret']]#, 'T', 'Po', 'Pa', 'DD', 'VV']]

weather = weather.rename(columns={'Local time in Bucharest / Filaret': 'Temp'})

cols = weather.columns



idx = pd.date_range('03-01-2010', '04-01-2019', freq='H')

weather['DatetimeRound'] = pd.to_datetime(weather.index)

weather.drop_duplicates(subset='DatetimeRound', inplace=True)

weather.index = pd.DatetimeIndex(weather.index)

weather = weather.reindex(idx, fill_value=np.NaN)

weather['DatetimeRound'] = pd.to_datetime(weather.index)

weather.reset_index(drop=True, inplace=True)



weather.fillna(method='ffill', inplace=True)





# weather['Day'] = weather['DatetimeRound'].dt.day

# weather['Month'] = weather['DatetimeRound'].dt.month

# weather_stats = weather.copy()

# weather_stats.drop(columns=['DatetimeRound'], inplace=True)

# weather_stats = weather_stats.groupby(['Month', 'Day']).mean()

# for col in weather_stats.columns:

#     weather_stats.rename(columns={col: col + '_'}, inplace=True)



# weather = weather.merge(weather_stats, on=['Month', 'Day'])

# for col in cols:

#     weather[col] = weather[[col, col + '_']].fillna(weather[col + '_'])[col]

# weather.drop(columns=weather_stats.columns, inplace=True)





print(weather.shape, weather.dropna().shape)

weather.head()
def weather_killer(df, weather):

    df = df.merge(weather, how='left', on='DatetimeRound').drop('DatetimeRound', axis=1)

    return df
df = weather_killer(df, weather)

df_test = weather_killer(df_test, weather)
df.dropna(inplace=True)

target = df.pop('Consumption_MW')
## 3. Split data into train / validation (leaving the last six months for validation)



eval_from = df['Datetime'].max() + relativedelta(months=-12)  # Here we set the 6 months threshold

train_df = df[df['Datetime'] < eval_from]

valid_df = df[df['Datetime'] >= eval_from + relativedelta(days=5) + relativedelta(hours=-3)]



train_target = target[df['Datetime'] < eval_from]

valid_target = target[df['Datetime'] >= eval_from + relativedelta(days=5) + relativedelta(hours=-3)]



print(f"Train data: {train_df['Datetime'].min()} -> {train_df['Datetime'].max()} | {len(train_df)} samples.")

print(f"Valid data: {valid_df['Datetime'].min()} -> {valid_df['Datetime'].max()} | {len(valid_df)} samples.")
(np.sqrt(mean_squared_error(valid_target, valid_df['killer_16'])),

 np.sqrt(mean_squared_error(valid_target, valid_df['killer_60'])),

 np.sqrt(mean_squared_error(valid_target, valid_df['killer_720'])),

 np.sqrt(mean_squared_error(valid_target, valid_df['killer_21600'])),

 )

# (np.sqrt(mean_squared_error(valid_target, valid_df['killer_16_y2'])),

#  np.sqrt(mean_squared_error(valid_target, valid_df['killer_60_y2'])),

#  np.sqrt(mean_squared_error(valid_target, valid_df['killer_720_y2'])),

#  np.sqrt(mean_squared_error(valid_target, valid_df['killer_21600_y2'])),

#  )
feat_to_drop = ['Date', 'Datetime']

# model = xgb.XGBRegressor(n_estimators=5000, max_depth=8, learning_rate=0.1, min_child_weight=10,

#                          subsample=0.75, colsample_bylevel=1, reg_lambda=10, gamma=5,

#                          n_jobs=-1, random_state=42)

# model = model.fit(

#     train_df.drop(feat_to_drop, axis=1),

#     train_target,

#     eval_set=[(valid_df.drop(feat_to_drop, axis=1), valid_target)],

#     eval_metric='rmse',

#     early_stopping_rounds=100

# )

# xgb_2 389 at 1200

# xgb_1 393 at 420
# pd.Series(model.feature_importances_, train_df.drop(feat_to_drop, axis=1).columns).sort_values()
model = xgb.XGBRegressor(n_estimators=1300, max_depth=8, learning_rate=0.1, min_child_weight=10,

                         subsample=0.75, colsample_bylevel=1, reg_lambda=10, gamma=5,

                         n_jobs=-1, random_state=42)

model = model.fit(

    df.drop(feat_to_drop, axis=1),

    target,

    eval_metric='rmse'

)
pred = model.predict(df_test.drop(feat_to_drop, axis=1))

pred = pd.Series(pred, df_test.Date).rename('Consumption_MW').to_frame()

pred.to_csv('xgb_2_temp_server.csv')