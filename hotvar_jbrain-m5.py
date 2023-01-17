import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from datetime import datetime
from datetime import timedelta
import calendar
import gc
import os
import time
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
df_stv = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
print(f'Item length : {len(df_stv)}')

df_sp = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
print(f'Item length : {len(df_sp)}')

df_calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
df_calendar['date'] = pd.to_datetime(df_calendar['date'], format='%Y-%m-%d')
print(f'calendar length : {len(df_calendar)}')

# externel dataset
df_holiday = pd.read_csv('../input/federal-holidays-usa-19662020/usholidays.csv', index_col=0)
df_holiday.columns = ['date', 'holiday']
df_holiday['date'] = pd.to_datetime(df_holiday['date'], format='%Y-%m-%d')

# oil prices dataset
df_oil_US = pd.read_csv('../input/m5-external-data/Weekly_US_Gasoline_Diesel_Prices.csv', header=6)
df_oil_CA = pd.read_csv('../input/m5-external-data/Weekly_California_Gasoline_Prices.csv', header=6)
df_oil_WI = pd.read_csv('../input/m5-external-data/Weekly_Minnesota_Gasoline_Prices.csv', header=6)
df_oil_TX = pd.read_csv('../input/m5-external-data/Weekly_Texas_Gasoline_Prices.csv', header=6)
df_oil_US.columns = ['date', 'US_diesel', 'US_gasoline']
df_oil_CA.columns = ['date', 'gasoline']
df_oil_WI.columns = ['date', 'gasoline']
df_oil_TX.columns = ['date', 'gasoline']
df_oil_US['date'] = pd.to_datetime(df_oil_US['date'], format='%m/%d/%Y')
df_oil_CA['date'] = pd.to_datetime(df_oil_CA['date'], format='%m/%d/%Y')
df_oil_WI['date'] = pd.to_datetime(df_oil_WI['date'], format='%m/%d/%Y')
df_oil_TX['date'] = pd.to_datetime(df_oil_TX['date'], format='%m/%d/%Y')
sample_submission = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
sample_submission.set_index('id', inplace=True)
sample_submission
df_calendar
df_holiday
def get_preproc_data(stv, calendar, sp, start_day):
    start = time.time()
    gc.collect()
    
    global df_oil_US, df_oil_CA, df_oil_WI, df_oil_TX
    
    for i in range(1, 57):
        stv['d_'+str(i+1913)] = np.zeros(len(stv))
    
    # 1913일 판매 데이터를 row-wise로 변환
    data_ = pd.melt(stv,
                   id_vars=stv.columns[:6],
                   value_vars=stv.columns[5+start_day:],
                   var_name='d',
                   value_name='solds')
    
    
    def shift_with_copy(df, back_shift=-7):
        a = df.fillna('').copy()
        for n in range(-1, back_shift-1, -1):
            b = df.shift(n).fillna('')
            for i in range(len(a)):
                a[i] += b[i]
        return a
    
    # event정보 처리
    calendar['event_name_1'] = shift_with_copy(calendar['event_name_1'], -7)
    calendar['event_type_1'] = shift_with_copy(calendar['event_type_1'], -7)
    calendar['event_name_2'] = shift_with_copy(calendar['event_name_2'], -7)
    calendar['event_type_2'] = shift_with_copy(calendar['event_type_2'], -7)
    
    # holiday 추가 / 처리
    calendar = calendar.merge(df_holiday, on='date', how='left')
    calendar['holiday'] = shift_with_copy(calendar['holiday'], -7)
    
    # melting을 위해 SNAP정보를 뒤로
    c = list(calendar.columns)
    calendar = calendar[c[:11]+[c[-1]]+c[11:14]]
    
    
    # SNAP 정보 row-wise 변환
    calendar = pd.melt(calendar,
                   id_vars=calendar.columns[:12],
                   value_vars=calendar.columns[12:],
                   var_name='state_id',
                   value_name='SNAP'
                  )

    # state를 KEY값으로 활용할 수 있도록 처리
    calendar['state_id'] = calendar['state_id'].apply(lambda x: x[-2:])
    # SNAP값 binarize
    calendar['SNAP'] = calendar['SNAP'].apply(lambda x: x==1)
    
    # calendar 매핑
    data_ = data_.merge(calendar, on=['d', 'state_id'], how='inner', copy=False)
    
    # df_sp 매핑
    data_ = data_.merge(sp, on=['store_id', 'item_id', 'wm_yr_wk'], how='left', copy=False)
    
    # 비어있는 sell_price -> 재고 없음
    # 재고없음은 0으로 처리
    data_['sell_price'] = data_['sell_price'].fillna(0)
    
    
    # 유가 4주 shift
    df_oil_US['US_diesel'] = df_oil_US['US_diesel'].shift(4)
    df_oil_US['US_gasoline'] = df_oil_US['US_gasoline'].shift(4)
    df_oil_CA['gasoline'] = df_oil_CA['gasoline'].shift(4)
    df_oil_WI['gasoline'] = df_oil_WI['gasoline'].shift(4)
    df_oil_TX['gasoline'] = df_oil_TX['gasoline'].shift(4)

    # calendar와 병합을 위해 날짜 동기화
    start_date = calendar.loc[0, 'date']
    end_date = calendar.loc[len(calendar)-1, 'date'] + timedelta(days=1)
    df_oil_US = df_oil_US[(df_oil_US['date'] >= start_date) & (df_oil_US['date'] <= end_date)]
    df_oil_CA = df_oil_CA[(df_oil_CA['date'] >= start_date) & (df_oil_CA['date'] <= end_date)]
    df_oil_WI = df_oil_WI[(df_oil_WI['date'] >= start_date) & (df_oil_WI['date'] <= end_date)]
    df_oil_TX = df_oil_TX[(df_oil_TX['date'] >= start_date) & (df_oil_TX['date'] <= end_date)]
    df_oil_US['date'] = df_oil_US['date'] - timedelta(days=1)
    df_oil_CA['date'] = df_oil_CA['date'] - timedelta(days=1)
    df_oil_WI['date'] = df_oil_WI['date'] - timedelta(days=1)
    df_oil_TX['date'] = df_oil_TX['date'] - timedelta(days=1)

    df_oil_CA['state_id'] = 'CA'
    df_oil_WI['state_id'] = 'WI'
    df_oil_TX['state_id'] = 'TX'
    df_oil_state = pd.concat([df_oil_CA, df_oil_TX, df_oil_WI], axis=0)

    # 유가정보의 date를 calendar의 wm_yr_wk로 변환
    df_oil_US['wm_yr_wk'] = df_oil_US['date'].apply(lambda x:calendar[calendar['date']==x].iloc[0, 1])
    df_oil_state['wm_yr_wk'] = df_oil_state['date'].apply(lambda x:calendar[calendar['date']==x].iloc[0, 1])
    
    # US, state별 전체 유가정보를 data와 병합
    data_ = data_.merge(df_oil_US.drop(columns='date'), on='wm_yr_wk', how='left')
    data_ = data_.merge(df_oil_state.drop(columns='date'), on=['wm_yr_wk', 'state_id'], how='left')
    
    # 유가 정보를 28days fore-shift
                        
    # 필요없는 컬럼을 지워보자
    data_ = data_.drop(columns=['weekday', 'year'])

    # 컬럼 타입변환
    data_['d'] = data_['d'].apply(lambda x: x[2:]).astype(np.uint16)
    dict_convert = {'item_id':'category',
                    'dept_id':'category',
                    'cat_id':'category',
                    'store_id':'category',
                    'state_id':'category',
                    'solds':np.uint16,
                    'wday':'category',
                    'month':'category',
                    'date':'datetime64',
                    'SNAP':'category',
                    'sell_price':np.float32,
                    'event_name_1':'category',
                    'event_type_1':'category',
                    'event_name_2':'category',
                    'event_type_2':'category',
                    'holiday':'category'
                   }

    display(data_)
    for c, t in dict_convert.items():
        data_[c] = data_[c].astype(t)
    
    print('전처리 종료')
    print(f'Preprocessing --> running time : {str(timedelta(seconds=time.time() - start))}')
    
    return reduce_mem_usage(data_)
def m5_fe(data_):
    start = time.time()
    # feature engineering
    # sell_price
    data_.sort_values(by=['id', 'd'], inplace=True)
    
    # solds feature
    for s in [28]:
        for r in [7, 14, 28]:
            data_[f'mean_solds_{s}s_{r}r'] = data_.groupby(['id'])['solds'].shift(s).rolling(r, min_periods=1).mean().reset_index(0, drop=True)
    
    # sell_price feature
    for r in [7, 14, 28]:
        data_[f'mean_sp_{r}r'] = data_.groupby(['id'])['sell_price'].rolling(r, min_periods=1).mean().reset_index(0, drop=True)
            
    # oil prices feature
    for r in [7, 14, 28]:
        data_[f'mean_US_diesel_{r}r'] = data_.groupby(['id'])['US_diesel'].rolling(r, min_periods=1).mean().reset_index(0, drop=True)
        data_[f'mean_US_gasoline_{r}r'] = data_.groupby(['id'])['US_gasoline'].rolling(r, min_periods=1).mean().reset_index(0, drop=True)
        data_[f'mean_gasoline_{r}r'] = data_.groupby(['id'])['gasoline'].rolling(r, min_periods=1).mean().reset_index(0, drop=True)
    
    # release-day feature
    def get_rd(sp_per_id):
        ret = np.zeros(len(sp_per_id))
        cur_id = sp_per_id.iloc[0, 0]
        day_cnt = 0
        for i, (_id, sp) in enumerate(zip(sp_per_id['id'], sp_per_id['sell_price'])):
            if _id != cur_id:
                cur_id = _id
                day_cnt = 0
            if sp != 0:
                ret[i] = day_cnt
                day_cnt += 1
            else:
                day_cnt = 0
        return ret
    
    data_['rd'] = get_rd(data_[['id', 'sell_price']])
    print('rd features')

    # price-steady-period feature
    def get_psp(sp_per_id):
        ret = np.zeros(len(sp_per_id))
        cur_id = sp_per_id.iloc[0, 0]
        cur_sp = sp_per_id.iloc[0, 1]
        day_cnt = 0
        for i , (_id, sp) in enumerate(zip(sp_per_id['id'], sp_per_id['sell_price'])):
            if _id != cur_id:
                cur_id = _id
                day_cnt = 0
            if sp != cur_sp or sp == 0:
                cur_sp = sp
                day_cnt = 0
            else:
                day_cnt += 1
                ret[i] = day_cnt
        return ret
 
    data_['psp'] = get_psp(data_[['id', 'sell_price']])
    print('price steady period')
    
    display(data_)
    display(data_.dtypes)
    
    print(f'Feature engineering --> running time : {str(timedelta(seconds=time.time() - start))}')
    
    return reduce_mem_usage(data_.dropna())
START_DAY = 1000
data = get_preproc_data(df_stv, df_calendar, df_sp, START_DAY)
data = m5_fe(data)
cat_features = [f for f in data.columns if data[f].dtype.name == 'category']

#del df_sp, df_stv, df_calendar
gc.collect()
data.dtypes
# Bayesian hyperparameter optimization
def bayesian_opt(data_, start_val=1800, init_iter=5, n_iter=10, num_iterations=1000):
    L_RATE = 0.04
    train_set = data_[data_['d']<start_val]
    valid_set = data_[(data_['d']>=start_val) & (data_['d']<=1913)]

    # make lgb.Dataset
    train_data = lgb.Dataset(train_set[train_set.columns.drop(['id', 'solds', 'date'])],
                                 label=train_set['solds'],
                                 categorical_feature=cat_features,
                                 free_raw_data=False)
    valid_data = lgb.Dataset(valid_set[valid_set.columns.drop(['id', 'solds', 'date'])],
                             label=valid_set['solds'],
                             categorical_feature=cat_features,
                             free_raw_data=False)
    def hyp_lgbm(num_leaves, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
        # default params
        params = {'application':'regression',
                 'num_iterations':num_iterations,
                 'learning_rate':0.01,
                 'early_stopping_round':10,
                 'metric':'rmse'}
        # modifying
        params['num_leaves'] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['lambda_l1'] = lambda_l1
        params['lambda_l2'] = lambda_l2
        
        # fit
        model = lgb.train(params,
                     train_set=train_data,
                     valid_sets=[valid_data],
                     verbose_eval=None)
        return -1 * model.best_score['valid_0']['rmse']
    
    # hyperparams' range
    pds = {'num_leaves':(30, 200),
          'feature_fraction':(0.5, 1),
          'bagging_fraction':(0.2, 0.8),
          'lambda_l1': (0.0, 0.95),
          'lambda_l2': (0.0, 0.95)
          }
    
    # surrogate model
    optimizer = BayesianOptimization(hyp_lgbm, pds, random_state=42)
    
    # optimize
    optimizer.maximize(init_points=init_iter, n_iter=n_iter)
    
    return optimizer
    
opt = bayesian_opt(data, 1800)
opt
opt.max
####################################
# Split train-valid dataset
# d_0 ~ d_1913 : train+validation
# d_1914 ~ d_1969 : test
####################################
L_RATE = 0.01
NUM_ITER = 5000
start_val = 1913

params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'regression',
        'n_jobs': -1,
        'seed': 236,
        'learning_rate': L_RATE,
        'bagging_fraction': 0.2,
        'feature_fraction': 1.0,
        'max_depth':-1,
        'lambda_l1':0.95,
        'lambda_l2':0.0,
        'num_leaves':256,
        'verbose':1}

train_set = data[data['d']<=start_val]
valid_set = data[(data['d']>1800)&(data['d']<=1913)]
X_test = data[(data['d']>1913)&(data['d']<=1941)].drop(columns=['solds', 'date'])
del data
gc.collect()

# make lgb.Dataset
train_data = lgb.Dataset(train_set[train_set.columns.drop(['id', 'solds', 'date'])],
                             label=train_set['solds'],
                             categorical_feature=cat_features)

valid_data = lgb.Dataset(valid_set[valid_set.columns.drop(['id', 'solds', 'date'])],
                         label=valid_set['solds'],
                         categorical_feature=cat_features)

del train_set, valid_set
gc.collect()

# fit
model = lgb.train(params,
                 train_set=train_data,
                 valid_sets=[valid_data],
                 num_boost_round=NUM_ITER,
                 early_stopping_rounds=250,
                 verbose_eval=100)

model
plt.figure(figsize=(10, 10))
plt.xticks(color='y')
plt.yticks(color='y')
f_importance = pd.DataFrame(data={'fname':model.feature_name(), 'fval':model.feature_importance()})
f_importance.sort_values(by='fval', ascending=False, inplace=True)
sns.barplot(x='fval', y='fname', data=f_importance)
X_test['pred_solds'] = model.predict(X_test.drop(columns=['id']))
submission = X_test[['id', 'd', 'pred_solds']].pivot(index='id', columns='d').iloc[:, :28]
submission.columns = ['F'+str(i) for i in range(1, 29)]
my_submission = sample_submission + submission
my_submission = my_submission.loc[sample_submission.index, :].fillna(value=0.)
my_submission = np.maximum(0, my_submission)
my_submission.to_csv('submission.csv')
my_submission