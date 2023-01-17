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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

from catboost import Pool, CatBoostRegressor, cv
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp

%matplotlib inline
path = '/kaggle/input/hse-pml-2/'
data_train = pd.read_csv(path+'train_resort.csv',sep=',', decimal='.')
data_test = pd.read_csv(path+'test_resort.csv',sep=',', decimal='.')
data_train.head()
def train_test_NAN_compare(df_train, df_test):
    
    train_c = pd.DataFrame(np.sum(df_train.isnull(),axis=0))    
    test_c = pd.DataFrame(np.sum(df_test.isnull(),axis=0))
    
    train_c.reset_index(level=0, inplace=True)
    test_c.reset_index(level=0, inplace=True)
    train_c.columns = ['Var','NAN']
    test_c.columns = ['Var','NAN']
    
    train_c['NAN_share'] = train_c['NAN']/df_train.shape[0]
    test_c['NAN_share'] = test_c['NAN']/df_test.shape[0]
    
    return train_c.merge(test_c, left_on='Var', right_on='Var',
          suffixes=('_train', '_test'))
train_test_NAN_compare(data_train,data_test)
def train_test_Nunique_compare(df_train, df_test):
    
    train_c = pd.DataFrame(df_train.nunique())    
    test_c = pd.DataFrame(df_test.nunique())
    
    train_c.reset_index(level=0, inplace=True)
    test_c.reset_index(level=0, inplace=True)
    train_c.columns = ['Var','Nunique']
    test_c.columns = ['Var','Nunique']
    
    train_c['Nunique_share'] = train_c['Nunique']/df_train.shape[0]
    test_c['Nunique_share'] = test_c['Nunique']/df_test.shape[0]
    
    return train_c.merge(test_c, left_on='Var', right_on='Var',
          suffixes=('_train', '_test'))
train_test_Nunique_compare(data_train,data_test)
#Спасибо Дмитрию :)
def check_coverage(x1, x2):
    return len(set(x1) & set(x2)) / len(set(x1))
for col in data_train.drop(columns=['amount_spent_per_room_night_scaled']).columns:
    print('Column: {}, coverage: {}'.format(col, check_coverage(data_train[col].fillna(-999),
                                                                data_test[col].fillna(-999))))
#change format for date
data_train['checkout_date']=pd.to_datetime(data_train['checkout_date'], format='%Y-%m-%d')
data_train['booking_date']=pd.to_datetime(data_train['booking_date'], format='%Y-%m-%d')
data_train['checkin_date']=pd.to_datetime(data_train['checkin_date'], format='%Y-%m-%d')

data_test['checkout_date']=pd.to_datetime(data_test['checkout_date'], format='%Y-%m-%d')
data_test['booking_date']=pd.to_datetime(data_test['booking_date'], format='%Y-%m-%d')
data_test['checkin_date']=pd.to_datetime(data_test['checkin_date'], format='%Y-%m-%d')
data_train['reservation_diff_D'] = (data_train['checkin_date']-data_train['booking_date'])/np.timedelta64(1,'D')
data_train['trip_duration_D'] = (data_train['checkout_date']-data_train['checkin_date'])/np.timedelta64(1,'D')
data_train['reservation_diff_M'] = (data_train['checkin_date']-data_train['booking_date'])/np.timedelta64(1,'M')
data_train['reservation_diff_W'] = (data_train['checkin_date']-data_train['booking_date'])/np.timedelta64(1,'W')

data_test['reservation_diff_D'] = (data_test['checkin_date']-data_test['booking_date'])/np.timedelta64(1,'D')
data_test['trip_duration_D'] = (data_test['checkout_date']-data_test['checkin_date'])/np.timedelta64(1,'D')
data_test['reservation_diff_M'] = (data_test['checkin_date']-data_test['booking_date'])/np.timedelta64(1,'M')
data_test['reservation_diff_W'] = (data_test['checkin_date']-data_test['booking_date'])/np.timedelta64(1,'W')

data_train['booking_date_dayofyear']=data_train.booking_date.dt.dayofyear
data_train['booking_date_dayofweek']=data_train.booking_date.dt.dayofweek
data_train['booking_date_month']=data_train.booking_date.dt.month
data_train['booking_date_year']=data_train.booking_date.dt.year

data_train['checkout_date_dayofyear']=data_train.checkout_date.dt.dayofyear
data_train['checkout_date_dayofweek']=data_train.checkout_date.dt.dayofweek
data_train['checkout_date_month']=data_train.checkout_date.dt.month
data_train['checkout_date_year']=data_train.checkout_date.dt.year

data_train['checkin_date_dayofyear']=data_train.checkin_date.dt.dayofyear
data_train['checkin_date_dayofweek']=data_train.checkin_date.dt.dayofweek
data_train['checkin_date_month']=data_train.checkin_date.dt.month
data_train['checkin_date_year']=data_train.checkin_date.dt.year


data_test['booking_date_dayofyear']=data_test.booking_date.dt.dayofyear
data_test['booking_date_dayofweek']=data_test.booking_date.dt.dayofweek
data_test['booking_date_month']=data_test.booking_date.dt.month
data_test['booking_date_year']=data_test.booking_date.dt.year

data_test['checkout_date_dayofyear']=data_test.checkout_date.dt.dayofyear
data_test['checkout_date_dayofweek']=data_test.checkout_date.dt.dayofweek
data_test['checkout_date_month']=data_test.checkout_date.dt.month
data_test['checkout_date_year']=data_test.checkout_date.dt.year

data_test['checkin_date_dayofyear']=data_test.checkin_date.dt.dayofyear
data_test['checkin_date_dayofweek']=data_test.checkin_date.dt.dayofweek
data_test['checkin_date_month']=data_test.checkin_date.dt.month
data_test['checkin_date_year']=data_test.checkin_date.dt.year
def train_test_compare(df_train, df_test, target, ):
    df_train['train_f']=1
    df_test['train_f']=0
    df_all = df_train.drop(columns=[target]).append(df_test)
    df_all = df_all._get_numeric_data()
    
    fig, ax = plt.subplots(int(np.ceil(df_all.columns.size/2)),2, figsize=(10,60))
    ax = ax.flatten()
    for i, feature in enumerate(df_all.columns.values):
        kde = (round(ks_2samp(df_all[df_all['train_f']==1][feature],
                       df_all[df_all['train_f']==0][feature])[1],5))
        sns.kdeplot(df_all[df_all['train_f']==1][feature], bw=0.5, ax=ax[i],legend=True)    
        sns.kdeplot(df_all[df_all['train_f']==0][feature], bw=0.5, ax=ax[i],legend=True)
        ax[i].set_title(feature+' /KDE='+str(kde))
        ax[i].legend(['Train', 'Test'])

    plt.show()
#If the KS statistic is small or the p-value is high, then we cannot reject the hypothesis that the distributions of the two samples are the same.
train_test_compare(data_train,data_test, target='amount_spent_per_room_night_scaled')
corr = data_train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(22, 18))
sns.heatmap(data_train.corr(), mask=mask, cmap='viridis', center=0, square=True, cbar_kws={"shrink": .5})
plt.show()
data_train[['roomnights', 'trip_duration_D', 'checkout_date', 'checkin_date']]
data_train[data_train['roomnights'] > data_train['trip_duration_D']].shape, \
data_train[data_train['roomnights'] > data_train['trip_duration_D']].shape[0]/data_train.shape[0]
data_test[data_test['roomnights'] > data_test['trip_duration_D']].shape, \
data_test[data_test['roomnights'] > data_test['trip_duration_D']].shape[0]/data_test.shape[0]
fig, ax = plt.subplots(1,2, figsize=(10, 5))
data_train.plot.scatter(x='roomnights',
                      y='trip_duration_D',
                      c='DarkBlue', ax=ax[0])
data_test.plot.scatter(x='roomnights',
                      y='trip_duration_D',
                      c='DarkBlue', ax=ax[1])
ax[0].set_title('Train')
ax[1].set_title('Test')
data_train[data_train['roomnights']<=0]
corr = data_test.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(22, 18))
sns.heatmap(data_test.corr(), mask=mask, cmap='viridis', center=0, square=True, cbar_kws={"shrink": .5})
plt.show()
data_train.plot.scatter(x='roomnights',
                      y='trip_duration_D',
                      c='DarkBlue')
data_test.plot.scatter(x='roomnights',
                      y='trip_duration_D',
                      c='DarkBlue')
fig, ax = plt.subplots(3,2, figsize=(10,8))
ax = ax.flatten()
sns.countplot(y="booking_date_year", data=data_train, ax=ax[0])
sns.countplot(y="booking_date_year", data=data_test, ax=ax[1])
sns.countplot(y="checkin_date_year", data=data_train, ax=ax[2])
sns.countplot(y="checkin_date_year", data=data_test, ax=ax[3])
sns.countplot(y="checkout_date_year", data=data_train, ax=ax[4])
sns.countplot(y="checkout_date_year", data=data_test, ax=ax[5])

print(sum(data_test['checkin_date_year']==2012))
data_test[data_test['checkin_date_year']==2012]
def var_boxplot(df_train ):
    fig, ax = plt.subplots(int(np.ceil(df_train.columns.size/2)),2, figsize=(10,100))
    ax = ax.flatten()
    for i, feature in enumerate(df_train.columns.values):
        sns.boxplot(df_train[feature], ax=ax[i])
        ax[i].set_title(feature)
        
        
    plt.show()
#train
var_boxplot(data_train._get_numeric_data())
#test
var_boxplot(data_test._get_numeric_data())
sns.distplot(data_train["amount_spent_per_room_night_scaled"])
sns.distplot(np.log(data_train["amount_spent_per_room_night_scaled"]))
sns.distplot(np.exp(data_train["amount_spent_per_room_night_scaled"]))
sns.boxplot(data_train["amount_spent_per_room_night_scaled"])
def target_boxplot(df_train, target, ):
    fig, ax = plt.subplots(int(np.ceil(df_train.columns.size/2)),2, figsize=(10,100))
    ax = ax.flatten()
    for i, feature in enumerate(df_train.columns.values):
        if feature!=target:
            sns.boxplot(x=feature, y=target, ax=ax[i],
                     data=df_train)
            ax[i].set_title(feature)
        
        
    plt.show()
target_boxplot(data_train[['channel_code', 'main_product_code', 'numberofadults',
       'numberofchildren', 'persontravellingid', 'resort_region_code',
                            'resort_type_code', 'room_type_booked_code', 'roomnights',
       'season_holidayed_code', 'state_code_residence', 'state_code_resort',
                           'total_pax', 'member_age_buckets', 'booking_type_code', 
       'cluster_code', 'reservationstatusid_code',
                           'trip_duration_D', 'reservation_diff_M', 
                           'booking_date_dayofweek','checkout_date_dayofweek',
                           'checkin_date_dayofweek'
                           ,"amount_spent_per_room_night_scaled"]],"amount_spent_per_room_night_scaled")
sns.boxplot(x="member_age_buckets", y="amount_spent_per_room_night_scaled",# hue="smoker",
            order=['A', 'B','C', 'D','E', 'F', 'G', 'H','I','J'],
                 data=data_train)
