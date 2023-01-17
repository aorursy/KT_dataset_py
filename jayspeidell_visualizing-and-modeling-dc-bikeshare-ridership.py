import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime as dt
from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('../input/train.csv')
df.head()
def null_percentage(column):
    df_name = column.name
    nans = np.count_nonzero(column.isnull().values)
    total = column.size
    frac = nans / total
    perc = int(frac * 100)
    print('%d%% of values or %d missing from %s column.' % (perc, nans, df_name))

def check_null(df, columns):
    for col in columns:
        null_percentage(df[col])
        
check_null(df, df.columns)
def process_features(df):
    
    # Get month, day of month, and time of day. 
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']
    df['month'] = df.datetime.apply(lambda x: months[int(x[5:7]) - 1])
    df['day'] = df.datetime.apply(lambda x: x[8:10]).astype(int)
    df['hour'] = df.datetime.apply(lambda x: x[11:13]).astype(int)
    
    def get_season(m):
        if m in ['January', 'February', 'December']:
            return 'Winter'
        elif m in [ 'March', 'April', 'May']:
            return 'Spring'
        elif m in ['June', 'July','August']:
            return 'Summer'
        else:
            return 'Fall'
        
    df['real_seasons'] = df.month.apply(lambda x: get_season(x))
    
    # Change "feels like" temperature to deviation from the mean of 24, which is a comfortable temperature. 
    median_temp = df.atemp.median()
    df['temp_dev'] = df.atemp.apply(lambda x: x - median_temp)
    
    # Create a date object and use it to extract day of week. 
    df['date'] = df.datetime.apply(lambda x: dt.strptime(x, "%Y-%m-%d %H:%M:%S").date())
    weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df['day_of_week'] = df.date.apply(lambda x: weekdays[x.weekday()])
    df['weekend'] = df.day_of_week.apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    
    df = df.drop(['date', 'datetime'], axis=1)
    
    print(df.columns)
    
    return df
df = process_features(pd.read_csv('../input/train.csv'))
df.head()
#Ridership by Month

plt.figure('Daily rides by Day of Week', figsize=(10, 26))
plt.suptitle('Daily Rides by Day of Week', fontsize=20)
plt.subplot(311)
sns.boxplot(x='day_of_week', y='count', data=df)
plt.title('All Riders', fontsize=16)

plt.subplot(312)
sns.boxplot(x='day_of_week', y='casual', data=df)
plt.title('Casual Riders', fontsize=16)

plt.subplot(313)
sns.boxplot(x='day_of_week', y='registered', data=df)
plt.title('Registered Riders', fontsize=16)

plt.show()
#Ridership by Season

plt.figure('Daily rides by Season', figsize=(10, 20))
plt.suptitle('Daily Rides by Season', fontsize=20)
plt.subplot(311)
sns.boxplot(x='real_seasons', y='count', hue='weekend', data=df)
plt.title('All Riders', fontsize=16)

plt.subplot(312)
sns.boxplot(x='real_seasons', y='casual', hue='weekend', data=df)
plt.title('Casual Riders', fontsize=16)

plt.subplot(313)
sns.boxplot(x='real_seasons', y='registered', hue='weekend', data=df)
plt.title('Registered Riders', fontsize=16)

plt.show()
#Ridership by Season

plt.figure('Daily rides by Month', figsize=(10, 20))
plt.suptitle('Daily Rides by Month', fontsize=20)
plt.subplot(311)
sns.boxplot(x='month', y='count', hue='weekend', data=df)
plt.title('All Riders', fontsize=16)

plt.subplot(312)
sns.boxplot(x='month', y='casual', hue='weekend', data=df)
plt.title('Casual Riders', fontsize=16)

plt.subplot(313)
sns.boxplot(x='month', y='registered', hue='weekend', data=df)
plt.title('Registered Riders', fontsize=16)

plt.show()
plt.figure('Ridership v Feels-Like Temp Deviation', figsize=(10, 15))
plt.suptitle('Ridership v Feels-Like Temp Deviation', fontsize=20)
plt.subplot(311)
sns.regplot(x='temp_dev', y='count', data=df, x_bins=10, order=2)
plt.title('All Riders')
plt.subplot(312)
sns.regplot(x='temp_dev', y='casual', data=df, x_bins=10, order=2)
plt.title('Casual Riders')
plt.subplot(313)
sns.regplot(x='temp_dev', y='registered', data=df, x_bins=10, order=2)
plt.title('Registered Riders')
plt.show()
plt.figure('Ridership v Actual Temp', figsize=(10, 15))
plt.suptitle('Ridership v Actual Temp', fontsize=20)
plt.subplot(311)
sns.regplot(x='temp', y='count', data=df, x_bins=10, order=2)
plt.title('All Riders', fontsize=14)
plt.subplot(312)
sns.regplot(x='temp', y='casual', data=df, x_bins=10, order=2)
plt.title('Casual Riders', fontsize=14)
plt.subplot(313)
sns.regplot(x='temp', y='registered', data=df, x_bins=10, order=2)
plt.title('Registered Riders', fontsize=14)
plt.show() 
plt.figure('Ridership v Humidity', figsize=(10, 15))
plt.suptitle('Ridership v Humidity', fontsize=20)
plt.subplot(311)
sns.regplot(x='humidity', y='count', data=df, x_bins=10, order=2)
plt.title('All Riders', fontsize=14)
plt.subplot(312)
sns.regplot(x='humidity', y='casual', data=df, x_bins=10, order=2)
plt.title('Casual Riders', fontsize=14)
plt.subplot(313)
sns.regplot(x='humidity', y='registered', data=df, x_bins=10, order=2)
plt.title('Registered Riders', fontsize=14)
plt.show() 
plt.figure('Ridership v Wind', figsize=(10, 15))
plt.suptitle('Ridership v Wind', fontsize=20)
plt.subplot(311)
sns.regplot(x='windspeed', y='count', data=df, x_bins=20, order=3)
plt.title('All Riders', fontsize=14)
plt.subplot(312)
sns.regplot(x='windspeed', y='casual', data=df, x_bins=20, order=3)
plt.title('Casual Riders', fontsize=14)
plt.subplot(313)
sns.regplot(x='windspeed', y='registered', data=df, x_bins=20, order=3)
plt.title('Registered Riders', fontsize=14)
plt.show() 
plt.figure('Wind by month')
sns.boxplot(x='month', y='windspeed', data=df)
plt.title('Windspeed by Month', fontsize=20)
plt.show()
df.weather.value_counts()
plt.figure('Weather and Ridership', figsize=(10, 20))
plt.suptitle('Weather and Ridership', fontsize=20)
plt.subplot(311)
sns.boxplot(x='weather', y='count', data=df)
plt.title('All Riders', fontsize=14)
plt.subplot(312)
sns.boxplot(x='weather', y='casual', data=df)
plt.title('Casual Riders', fontsize=14)
plt.subplot(313)
sns.boxplot(x='weather', y='registered', data=df)
plt.title('Registered Riders', fontsize=14)
plt.show()
df.loc[df['weather'] == 4, 'weather'] = 3
def corr_heatmap(df, title):
    plt.figure('heatmap', figsize=(15,15))
    plt.suptitle(plt.title(title, fontsize=30))
    df_corr = df.corr()
    sns.heatmap(df_corr, vmax=0.6, square=True, annot=False, cmap='Blues')
    plt.yticks(rotation = 0)
    plt.xticks(rotation = 90)
    plt.show()
    
corr_heatmap(pd.get_dummies(df), 'Correlation Matrix of All Features')
hour_map = df.groupby(["hour","day_of_week"],sort=True)["count"].mean().unstack()
hour_map = hour_map[['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']]
hour_map.describe()
plt.figure('Hour and day', figsize=(25,25))
plt.title('Heatmap of ridership by time and day of week', fontsize=30)
sns.heatmap(hour_map, square=False, annot=True, cmap='coolwarm', fmt='.0f',annot_kws={"size":20})
plt.show()
import pandas as pd

df = process_features(pd.read_csv('../input/train.csv'))
df_submit = process_features(pd.read_csv('../input/test.csv'))

def clean_weather(df):
    df.loc[df['weather'] == 4, 'weather'] = 3
    return df

df = clean_weather(df)
df_test = clean_weather(df_submit)
remove_columns = ['season', 
                  #'holiday', 
                  'workingday', 
                  #'weather', 
                  #'temp', 
                  'atemp',
                  #'humidity', 
                  'windspeed', 
                  #'month',
                  'day', 
                  #'hour', 
                  #'real_seasons', 
                  #'temp_dev', 
                  #'day_of_week', 
                  'weekend'
                 ]

# Going to make this a multi-label ensemble problem and let make these three 
# predictions into features that feed into an overall model.

target_labels = ['casual', 'registered', 'count']
# Strip unwanted features
df_train = df.drop(remove_columns, axis=1)

df_targets = df_train[target_labels]
df_train = df_train.drop(target_labels, axis=1)

df_submit = df_test.drop(remove_columns, axis=1)
print(df_train.columns)
print(df_submit.columns)

df_train = pd.get_dummies(df_train)
df_submit = pd.get_dummies(df_submit)
print(df_train.shape[1] == df_submit.shape[1])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_train = scaler.fit_transform(df_train)
df_submit = scaler.transform(df_submit)

np_train = np.array(df_train)
np_targets = np.array(df_targets)
np_submit = np.array(df_submit)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(np_train, np_targets, test_size=0.15)
print(X_train.shape)
print(y_train.shape)
def rmsle(y_true,y_pred):
   assert len(y_true) == len(y_pred)
   return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=5, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred[y_pred < 0] = 1

print('RMSLE of predicting total count: %.4f' % rmsle(y_test[:,2], y_pred[:,2]))
print('RMSLE combining casual and registered predictions: %.4f' % rmsle(y_test[:,2], np.sum(y_pred[:,0:2], axis=1)))
count0 = 0
count1 = 0
for (a,b) in zip(np.sum(y_pred[:,0:2].astype(int), axis=1), y_pred[:,2].astype(int)):
    #print(a, b)
    if abs(a - b) == 0:
        count0 +=1
    if abs(a - b) <= 1:
        count1 +=1
print('Exact: %d' % count0)
print('Within one: %d ' % count1)
print('Total: %d ' % y_pred.shape[0])
print('Sum of registered and casual rider predictions is exactly the total count \nprediction %d%% of the time and within one 100%% of the time.' % int((count0 / y_pred.shape[0])*100))
import lightgbm as lgb
X_t, X_e, y_t, y_e = train_test_split(X_train, y_train[:,2], test_size=0.15)
print(y_t.shape)
print(y_e.shape)
lgb_train = lgb.Dataset(X_t, y_t)
lgb_eval = lgb.Dataset(X_e, y_e, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'l2_root',
    'num_leaves': 43,
    'max_depth': 16

}

gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                verbose_eval=0,
                early_stopping_rounds=5
               )

y_pred = gbm.predict(X_test)
y_pred[y_pred < 0] = 1

rmsle(y_pred, y_test[:,2])
submission = pd.read_csv('../input/sampleSubmission.csv')
submission['count'] = np.array(rf.predict(df_submit))[:,2]
print(submission.head())
submission.to_csv('submission.csv', index=False)