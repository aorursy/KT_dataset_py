# pip install pandas
# pip install 
# pip install scikit-learn
# pip install matplotlib
# pip install seaborn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
dfs = {}
for name in ['train', 'test']:
    df = pd.read_csv('../input/bike-sharing-demand/%s.csv' % name)
    df['_data'] = name
    dfs[name] = df
# combine train and test data into one df
df = dfs['train'].append(dfs['test'])

# lowercase column names
df.columns = map(str.lower, df.columns)

# parse datetime colum & add new time related columns
dt = pd.DatetimeIndex(df['datetime'])
df.set_index(dt, inplace=True)
df.tail(20)
df.info()
df.isnull().sum() 
#corelation matrix.
cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
df[['count','casual','registered']].hist(figsize=(20,10), grid=False, layout=(1, 3), bins = 30)
# logarithmic transformation of dependent cols
# (adding 1 first so that 0 values don't become -inf)
for col in ['casual', 'registered', 'count']:
    df['%s_log' % col] = np.log(df[col] + 1)
df[['count_log','casual_log','registered_log']].hist(figsize=(20,10), grid=False, layout=(1, 3), bins = 30)
df['date'] = dt.date
df['day'] = dt.day
df['month'] = dt.month
df['year'] = dt.year
df['hour'] = dt.hour
df['dow'] = dt.dayofweek
df['woy'] = dt.weekofyear
df.head()
# add a count_season column using join
by_season = df[df['_data'] == 'train'].groupby('season')[['count']].agg(sum)
by_season.columns = ['count_season']
df = df.join(by_season, on='season')

print(by_season)
#sns.factorplot(x='season',data=df,kind='count',size=5,aspect=1)
sns.factorplot(x='season',data=df,kind='count',size=5,aspect=1.5)
def get_day(day_start):
    day_end = day_start + pd.offsets.DateOffset(hours=23)
    return pd.date_range(day_start, day_end, freq="H")
#considering one entire day
#H= hourly frequency
# tax day
df.loc[get_day(pd.datetime(2011, 4, 15)), "workingday"] = 1
df.loc[get_day(pd.datetime(2012, 4, 16)), "workingday"] = 1
# thanksgiving friday
df.loc[get_day(pd.datetime(2011, 11, 25)), "workingday"] = 0
df.loc[get_day(pd.datetime(2012, 11, 23)), "workingday"] = 0
# tax day
df.loc[get_day(pd.datetime(2011, 4, 15)), "holiday"] = 0
df.loc[get_day(pd.datetime(2012, 4, 16)), "holiday"] = 0

# thanksgiving friday
df.loc[get_day(pd.datetime(2011, 11, 25)), "holiday"] = 1
df.loc[get_day(pd.datetime(2012, 11, 23)), "holiday"] = 1

#storms
df.loc[get_day(pd.datetime(2012, 5, 21)), "holiday"] = 1
#tornado
df.loc[get_day(pd.datetime(2012, 6, 1)), "holiday"] = 1
# rentals by hour, split by working day (or not)
by_hour = df[df['_data'] == 'train'].copy().groupby(['hour', 'workingday'])['count'].agg('sum').unstack()

by_hour.plot(kind='bar', figsize=(8,4), width=0.8);
df['peak'] = df[['hour', 'workingday']].apply(lambda x: (0, 1)[(x['workingday'] == 1 and  ( x['hour'] == 8 or 17 <= x['hour'] <= 18 or x['hour'] == 12)) or (x['workingday'] == 0 and  10 <= x['hour'] <= 19)], axis = 1)
df[(df['year'] == 2012) & (df['month'] == 12) & (df['day']==25)]
#sandy
df['holiday'] = df[['month', 'day', 'holiday', 'year']].apply(lambda x: (x['holiday'], 1)[x['year'] == 2012 and x['month'] == 10 and (x['day'] in [30])], axis = 1)

#christmas day and others
df['holiday'] = df[['month', 'day', 'holiday']].apply(lambda x: (x['holiday'], 1)[x['month'] == 12 and (x['day'] in [24, 26, 31])], axis = 1)
df['workingday'] = df[['month', 'day', 'workingday']].apply(lambda x: (x['workingday'], 0)[x['month'] == 12 and x['day'] in [24, 31]], axis = 1)
df['ideal'] = df[['temp', 'windspeed']].apply(lambda x: (0, 1)[x['temp'] > 27 and x['windspeed'] < 30], axis = 1)
df['sticky'] = df[['humidity', 'workingday']].apply(lambda x: (0, 1)[x['workingday'] == 1 and x['humidity'] >= 60], axis = 1)
# can also be visulaized using histograms for all the continuous variables.
df.temp.unique()
fig,axes=plt.subplots(2,2)
axes[0,0].hist(x="temp",data=df,edgecolor="black",linewidth=2,color='#ff4125')
axes[0,0].set_title("Variation of temp")
axes[0,1].hist(x="atemp",data=df,edgecolor="black",linewidth=2,color='#ff4125')
axes[0,1].set_title("Variation of atemp")
axes[1,0].hist(x="windspeed",data=df,edgecolor="black",linewidth=2,color='#ff4125')
axes[1,0].set_title("Variation of windspeed")
axes[1,1].hist(x="humidity",data=df,edgecolor="black",linewidth=2,color='#ff4125')
axes[1,1].set_title("Variation of humidity")
fig.set_size_inches(10,10)
windspeed = df[df['_data'] == 'train'].copy().groupby(['windspeed', 'workingday'])['count'].agg('sum').unstack()
windspeed.plot(kind='bar', figsize=(8,4), width=0.8);
tempera = df[df['_data'] == 'train'].copy().groupby(['temp', 'workingday'])['count'].agg('sum').unstack()
tempera.plot(kind='bar', figsize=(8,4), width=0.8);
humid = df[df['_data'] == 'train'].copy().groupby(['humidity', 'workingday'])['count'].agg('sum').unstack()
humid.plot(kind='bar', figsize=(15,4), width=0.8);
#corelation matrix.
cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
def getrmsle(y_pred, y_actual):
    diff = np.log(y_pred + 1) - np.log(y_actual + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)


def get_data():
    data = df[df['_data'] == 'train'].copy()
    return data


def train_test_split_custom(data, midway_day=15):
    train = data[data['day'] <= midway_day]
    val = data[data['day'] > midway_day]

    return train, val


def prepare_distributed_data(data, input_cols):
    X = data[input_cols].values
    y_registered = data['registered_log'].values
    y_casual = data['casual_log'].values

    return X, y_registered, y_casual


def predict_on_validation_set(model, input_cols):
    data = get_data()

    train, val = train_test_split_custom(data)

    X_train, y_train_registered, y_train_casual = prepare_distributed_data(train, input_cols)
    X_val, y_val_registered, y_val_casual = prepare_distributed_data(val, input_cols)

    model_registered = model.fit(X_train, y_train_registered)
    y_pred_registered = np.exp(model_registered.predict(X_val)) - 1

    model_casual = model.fit(X_train, y_train_casual)
    y_pred_casual = np.exp(model_casual.predict(X_val)) - 1

    y_pred_combined = np.round(y_pred_registered + y_pred_casual)
    y_pred_combined[y_pred_combined < 0] = 0

    y_val_combined = np.exp(y_val_registered) + np.exp(y_val_casual) - 2

    score = getrmsle(y_pred_combined, y_val_combined)
    return (y_pred_combined, y_val_combined, score)

df_test = df[df['_data'] == 'test'].copy()

# predict on test set & transform output back from log scale
def predict_on_test_data(model, x_cols):
    # prepare training set
    df_train = df[df['_data'] == 'train'].copy()
    X_train = df_train[x_cols].values
    y_train_cas = df_train['casual_log'].values
    y_train_reg = df_train['registered_log'].values

    # prepare test set
    X_test = df_test[x_cols].values

    casual_model = model.fit(X_train, y_train_cas)
    y_pred_cas = casual_model.predict(X_test)
    y_pred_cas = np.exp(y_pred_cas) - 1
    registered_model = model.fit(X_train, y_train_reg)
    y_pred_reg = registered_model.predict(X_test)
    y_pred_reg = np.exp(y_pred_reg) - 1
    # add casual & registered predictions together
    return y_pred_cas + y_pred_reg
params = {'n_estimators': 500, 'max_depth': 100, ' min_samples_leaf': 2, 'min_samples_split' : 5, 'n_jobs': -1,'bootstrap':True,'max_features': 'auto'}
rf_model = RandomForestRegressor(n_estimators =500,
 min_samples_split = 5,
 min_samples_leaf = 2,
 max_features= 'auto',
 max_depth = 100,
 bootstrap = True)
rf_cols = [
    'weather', 'temp', 'atemp', 'windspeed',
    'workingday', 'season', 'holiday', 'sticky',
    'hour', 'dow', 'woy', 'peak'
    ]

(rf_p, rf_val, rf_score) = predict_on_validation_set(rf_model, rf_cols)
print (rf_score)
df[rf_cols].corr()
params = {'n_estimators': 166, 'max_depth': 5, 'random_state': 0, 'min_samples_leaf' : 10, 'learning_rate': 0.1, 'subsample': 0.7, 'loss': 'ls'}
gbm_model = GradientBoostingRegressor(**params)
gbm_cols = [
    'weather', 'temp', 'atemp', 'humidity', 'windspeed',
    'holiday', 'workingday', 'season',
    'hour', 'dow', 'year', 'ideal', 'count_season',
]

(gbm_p, gbm_val, gbm_score) = predict_on_validation_set(gbm_model, gbm_cols)
print (gbm_score)
df[gbm_cols].corr()
# the blend gives a better score on the leaderboard, even though it does not on the validation set
y_p = np.round(.2*rf_p + .8*gbm_p)
print (getrmsle(y_p, rf_val))
rf_pred = predict_on_test_data(rf_model, rf_cols)
gbm_pred = predict_on_test_data(gbm_model, gbm_cols)
y_pred = np.round(.2*rf_pred + .8*gbm_pred)
# output predictions for submission
df_test['count'] = y_pred
final_df = df_test[['datetime', 'count']].copy()

final_df.to_csv('submit.csv', index=False)