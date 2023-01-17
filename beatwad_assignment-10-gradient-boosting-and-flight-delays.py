import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, anneal, Trials
train = pd.read_csv('../input/flight_delays_train.csv')
test = pd.read_csv('../input/flight_delays_test.csv')
train.head()
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())
train.info()
print(len(train.Month.unique()))
print(len(train.DayofMonth.unique()))
print(len(train.DayOfWeek.unique()))
print(len(train.UniqueCarrier.unique()))
print(len(train.Origin.unique()))
print(len(train.Dest.unique()))
def transform_to_int(value):
    return int(value[2:])

train['Month'] = train['Month'].apply(transform_to_int)
train['DayofMonth'] = train['DayofMonth'].apply(transform_to_int)
train['DayOfWeek'] = train['DayOfWeek'].apply(transform_to_int)
test['Month'] = test['Month'].apply(transform_to_int)
test['DayofMonth'] = test['DayofMonth'].apply(transform_to_int)
test['DayOfWeek'] = test['DayOfWeek'].apply(transform_to_int)
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax1 = sns.boxplot(train.loc[train['dep_delayed_15min']=='Y','Distance'], ax=ax[0])
ax2 = sns.boxplot(train.loc[train['dep_delayed_15min']=='N','Distance'], ax=ax[1])
train_del_dow = train.loc[train['dep_delayed_15min']=='Y', 'DayOfWeek'].value_counts().sort_index()/train['DayOfWeek'].value_counts().sort_index()*100

ax = train_del_dow.sort_index().plot(kind='bar', title='Delayed fligths by day of week', figsize=(10, 6))
ax.set_xlabel('\nDay of Week')
ax.set_ylabel('%')
train_del_month = train.loc[train['dep_delayed_15min']=='Y', 'Month'].value_counts().sort_index()/train['Month'].value_counts().sort_index()*100

ax = train_del_month.sort_index().plot(kind='bar', title='Percent of delayed fligths by month', figsize=(12, 6))
ax.set_ylabel('%')
ax.set_xlabel('Month')
train_del_dom = (train.loc[train['dep_delayed_15min']=='Y', 'DayofMonth'].value_counts(normalize=True)*100).sort_index()

ax = train_del_dom.sort_index().plot(kind='bar', title='Delayed fligths by day of month', figsize=(18, 6))
ax.set_ylabel('%')
ax.set_xlabel('Day of Month')
train_carrier = (train['UniqueCarrier'].value_counts())

ax = train_carrier.sort_index().plot(kind='bar', title='Number of flights by unique carrier', figsize=(18, 6))
ax.set_ylabel('Number of Flights')
ax.set_xlabel('Unique Carrier')
train_del_carrier = (train.loc[train['dep_delayed_15min']=='Y', 'UniqueCarrier'].value_counts().sort_index()/train['UniqueCarrier'].value_counts().sort_index())*100

ax = train_del_carrier.plot(kind='bar', title='Percent of delayed fligths by unique carrier', figsize=(18, 6))
ax.set_ylabel('%')
ax.set_xlabel('Unique Carrier')
train['Flight'] = train['Origin'] + '-' + train['Dest']
test['Flight'] = train['Origin'] + '-' + train['Dest']
train['DepTime'].describe()
def fix_dep_time(value):
    if value >= 2400:
        return value-2400
    return value

train['DepTime'] = train['DepTime'].apply(fix_dep_time)
test['DepTime'] = test['DepTime'].apply(fix_dep_time)
def fetch_hour_data(value):
    value_str = str(value)
    if len(value_str) <= 2:
        hour = 0
        minutes = int(value_str)
    elif len(value_str) == 3:  
        hour = int(value_str[:1])
        minutes = int(value_str[1:])
    else:
        hour = int(value_str[:2])
        minutes = int(value_str[2:])
    if minutes < 15:
        minutes = 0
    elif minutes in range(15, 45):
        minutes = 0.5
    else:
        minutes = 1
    if hour + minutes == 24:
        return 0.0
    return hour + minutes

def fetch_minutes_data(value):
    value_str = str(value)
    if len(value_str) <= 2:
        minutes = int(value_str)
    elif len(value_str) == 3:  
        minutes = int(value_str[1:])
    else:
        minutes = int(value_str[2:])
    return minutes

train['Hour'] = train['DepTime'].apply(fetch_hour_data).astype('str')
test['Hour'] = test['DepTime'].apply(fetch_hour_data).astype('str')

train['HourSQ'] = (train['Hour'].astype('float')**2)
test['HourSQ'] = (test['Hour'].astype('float')**2)
train['HourSQ2'] = (train['Hour'].astype('float')**4)
test['HourSQ2'] = (test['Hour'].astype('float')**4)

train['Minutes'] = train['DepTime'].apply(fetch_minutes_data)
test['Minutes'] = test['DepTime'].apply(fetch_minutes_data)
train_del_hour = train.loc[train['dep_delayed_15min']=='Y', 'Hour'].value_counts().sort_index()/train['Hour'].value_counts().sort_index()*100
# new_index = [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 4, 5, 6, 7, 8, 9]
# train_del_dom.index = new_index

ax = train_del_hour.sort_index().plot(kind='bar', title='Percent of delayed fligths by hour', figsize=(18, 6))
ax.set_ylabel('%')
ax.set_xlabel('Hour')
train.info()
def add_season(value):
    if value > 11 or value < 3:
        return 'Winter'
    elif value < 6:
        return 'Spring'
    elif value < 9:
        return 'Summer'
    else:
        return 'Autumn'
    
def add_weekend(value):
    if value > 5 :
        return 'Y'
    else:
        return 'N'

train['Season'] = train['Month'].apply(add_season)
train['Weekend'] = train['DayOfWeek'].apply(add_weekend)
test['Season'] = test['Month'].apply(add_season)
test['Weekend'] = test['DayOfWeek'].apply(add_weekend)
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

dr = pd.date_range(start='2011-01-01', end='2012-12-31')
df_hol = pd.DataFrame()
df_hol['Date'] = dr

cal = calendar()
holidays = cal.holidays(start=dr.min(), end=dr.max())

df_hol = df_hol[df_hol['Date'].isin(holidays)]
df_hol['DayofMonth'] = df_hol['Date'].dt.day
df_hol['DayofWeek'] = df_hol['Date'].dt.dayofweek + 1
df_hol['Month'] = df_hol['Date'].dt.month
df_hol
%%time

holiday_list = list()

def get_holiday_dates(row):
    month_code = row['Month']
    day_of_month_code = row['DayofMonth']
    day_of_week_code = row['DayofWeek']
    holiday_date = str(month_code) + '-' + str(day_of_month_code) + '-' + str(day_of_week_code)
    holiday_list.append(holiday_date)

df_hol.apply(get_holiday_dates, axis=1)

def find_holiday(row):
    month_code = str(row['Month'])[2:]
    day_of_month_code = str(row['DayofMonth'])[2:]
    day_of_week_code = str(row['DayOfWeek'])[2:]
    holiday_date = month_code + '-' + day_of_month_code + '-' + day_of_week_code
    if holiday_date in holiday_list:
        return 'Y'
    else:
        return row['Weekend']
        
train['Weekend'] = train.apply(find_holiday, axis=1)
test['Weekend'] = test.apply(find_holiday, axis=1)
def add_dist_label(value):
    if value <= 500:
        return 'vshort'
    elif value <= 1000:
        return 'short'
    elif value <= 1500:
        return 'middle'
    elif value <= 2000:
        return 'vmiddle'
    elif value <= 2500:
        return 'long'
    else:
        return 'vlong'
    
train['DistLabel'] = train['Distance'].apply(add_dist_label)
test['DistLabel'] = test['Distance'].apply(add_dist_label)
train_del_dist = (train.loc[train['dep_delayed_15min']=='Y', 'DistLabel'].value_counts().sort_index()/train['DistLabel'].value_counts().sort_index()*100).sort_values()

ax = train_del_dist.plot(kind='bar', title='Delayed fligths by distance', figsize=(18, 6))
ax.set_ylabel('%')
ax.set_xlabel('Distance')
def add_part_of_day(value):
    value = float(value)
    if value < 6:
        return 'Night'
    elif value < 12:
        return 'Morning'
    elif value < 18:
        return 'Day'
    else:
        return 'Evening'
    
train['PartOfDay'] = train['Hour'].apply(add_part_of_day)
test['PartOfDay'] = test['Hour'].apply(add_part_of_day)
train['Month'] = train['Month'].astype(str)
test['Month'] = test['Month'].astype(str)

train['DayofMonth'] = train['DayofMonth'].astype(str)
test['DayofMonth'] = test['DayofMonth'].astype(str)

train['DayOfWeek'] = train['DayOfWeek'].astype(str)
test['DayOfWeek'] = test['DayOfWeek'].astype(str)

train['OriginHour'] = train['Origin'] + '-' + train['Hour']
test['OriginHour'] = test['Origin'] + '-' + test['Hour']

train['DestHour'] = train['Dest'] + '-' + train['Hour']
test['DestHour'] = test['Dest'] + '-' + test['Hour']

train['CarrierOrigin'] = train['UniqueCarrier'] + '-' + train['Origin']
test['CarrierOrigin'] = test['UniqueCarrier'] + '-' + test['Origin']

train['CarrierDest'] = train['UniqueCarrier'] + '-' + train['Dest']
test['CarrierDest'] = test['UniqueCarrier'] + '-' + test['Dest']

train['CarrierHour'] = train['UniqueCarrier'] + '-' + train['Hour']
test['CarrierHour'] = test['UniqueCarrier'] + '-' + test['Hour']

train['CarrierOriginHour'] = train['UniqueCarrier'] + '-' + train['Origin'] + '-' + train['Hour']
test['CarrierOriginHour'] = test['UniqueCarrier'] + '-' + test['Origin'] + '-' + train['Hour']

train['CarrierDestHour'] = train['UniqueCarrier'] + '-' + train['Dest'] + '-' + train['Hour']
test['CarrierDestHour'] = test['UniqueCarrier'] + '-' + test['Dest'] + '-' + test['Hour']
def prepare_data(train, features_to_drop=[]):
    X_test = test.drop(features_to_drop, axis=1)
    features_to_drop.append('dep_delayed_15min')
    X_train = train.drop(features_to_drop, axis=1)
    y_train = train['dep_delayed_15min'].map({'Y': 1, 'N': 0})
    return X_train, y_train, X_test

def write_submission_to_file(predicted_labels, out_file, target='dep_delayed_15min', index_label='id'):
    predicted_df = pd.DataFrame(predicted_labels, columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

def train_predict_and_submit(train, test, filename, features_to_drop=[]):
    # prepare data
    X_train, y_train, X_test = prepare_data(train, features_to_drop)
    # make train and validation set
    X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=17)
    # determine categorical features
    categ_feat_idx = np.where(X_train.dtypes == 'object')[0]
    # train and test model 
    ctb = CatBoostClassifier(random_seed=17)
    ctb.fit(X_train_part, y_train_part, cat_features=categ_feat_idx)
    ctb_valid_pred = ctb.predict_proba(X_valid)[:, 1]
    # make predicitons and write them to file
    ctb_test_pred = ctb.predict_proba(X_test)[:, 1]
    write_submission_to_file(ctb_test_pred, filename)
    print(roc_auc_score(y_valid, ctb_valid_pred))
    feat_imp = zip(list(X_train.columns), list(ctb.feature_importances_))
    for i, j in feat_imp:
        print(i,'--->', j)
    return ctb
ctb = train_predict_and_submit(train, test, 'submission.csv', features_to_drop=['Distance', 'DepTime'])