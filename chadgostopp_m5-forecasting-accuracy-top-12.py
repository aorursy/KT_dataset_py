from  datetime import datetime, timedelta
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn import preprocessing
import os, sys, gc, time, warnings, pickle, psutil, random
from multiprocessing import Pool
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 300

val_start=1942
val_end=1969
TARGET='sale'
predict_len=1941-100
def dtypes_opt(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    obj_type=['object']
    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col=='id':
            df[col]=df[col].astype('object')

        elif col_type in numerics:
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
        elif col_type in obj_type:
            df[col] = df[col].astype('category')
    return 
def train_format(predict=False):
    raw=pd.read_csv('sales_train_evaluation.csv')

        
    for i in range(val_start,val_end+1):
        col_name='d_'+str(i)
        raw[col_name]=np.nan
        
    df=pd.melt(raw,id_vars=['id','item_id','dept_id','cat_id','store_id','state_id']
               ,value_vars=[ 'd_'+str(d) for d in range(1,val_end+1)],
               var_name='d', 
               value_name='sale')
    df.d=df.d.apply(lambda x: x.split('_')[1]).astype(np.int16)
    
    if predict:
        df=df[df['d']>=predict_len]
    
    for col in list(df):
        if col!='id' and df[col].dtypes=='object':
            le = preprocessing.LabelEncoder()
            le.fit(raw[col])
            df[col]=le.transform(df[col])

    dtypes_opt(df)                            
    return df 

def calendar_fe() :
    calendar=pd.read_csv('calendar.csv')
    calendar.d=calendar.d.apply(lambda x: x.split('_')[1]).astype('int16')
    
    calendar['date']=pd.to_datetime(calendar.date)
    calendar['y']=pd.to_datetime(calendar.date).dt.year
    calendar['qua']=pd.to_datetime(calendar.date).dt.quarter
    calendar['mon']=pd.to_datetime(calendar.date).dt.month
    calendar['wy']=pd.to_datetime(calendar.date).dt.weekofyear
    calendar['dw']=pd.to_datetime(calendar.date).dt.dayofweek
    calendar['dm']=pd.to_datetime(calendar.date).dt.day
    calendar['weekend']=calendar.dw.apply(lambda x: 1 if x>=5 else 0 ).astype('int8')   
    
    # print('calendar:') 
    del calendar['month'],calendar['wday'],calendar['year'],calendar['weekday'],calendar['date']
    dtypes_opt(calendar)
    
    
    for col in list(calendar):
        if calendar[col].dtype.name=='category':
            calendar[col] = calendar[col].cat.codes.astype('int')
    dtypes_opt(calendar)
    return calendar




def price_fe():
    price=pd.read_csv('sell_prices.csv')
    price['id']=price['item_id']+'_'+price['store_id']+'_evaluation'

    # Price encode
    price['price_max']=price.groupby(['id'])['sell_price'].transform('max')
    price['price_min']=price.groupby(['id'])['sell_price'].transform('min')
    price['price_enc_1']=price['sell_price']/price['price_max']
    price['price_enc_2']=price['sell_price']/price['price_min']
    
    del price['price_max'],price['price_min'],price['store_id'],price['item_id']
    
    dtypes_opt(price)
    return price

def sale_fe(df,predict=False):
    sale_fe=df[['id','d','sale']]
    shift_days=[7,28]
    windows=[7,28]
    
    sale_fe['shift_1_sale_fe']=sale_fe[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(1)).astype('float16')
    sale_fe['1rmean_7days_fe']=sale_fe[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(1).rolling(7).mean()).astype('float16')
#     sale_fe['1rmax_7days_fe']=sale_fe[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(1).rolling(7).max()).astype('float16')
#     sale_fe['1rmin_7days_fe']=sale_fe[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(1).rolling(7).min()).astype('float16')
#     sale_fe['1rstd_7days_fe']=sale_fe[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(1).rolling(7).std()).astype('float16')
    
    for day in shift_days:
        start_time=time.time()
        print('Shift Day:',day)
        col_name='shift_'+str(day)+'_sale'+'_fe'
        sale_fe[col_name]=sale_fe[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(day)).astype('float16')
        for window in windows:
            print('Window',window)
            sale_fe[str(day)+'rmean_'+str(window)+'days'+'_fe']=sale_fe[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(day).rolling(window).mean()).astype('float16')
            sale_fe[str(day)+'rmax_'+str(window)+'days'+'_fe']=sale_fe[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(day).rolling(window).max()).astype('float16')
            sale_fe[str(day)+'rmin_'+str(window)+'days'+'_fe']=sale_fe[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(day).rolling(window).min()).astype('float16')
            sale_fe[str(day)+'rstd_'+str(window)+'days'+'_fe']=sale_fe[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(day).rolling(window).std()).astype('float16')            
        print('Time:',time.time()-start_time)
    if not predict:
        sale_fe.dropna(inplace=True,subset=[col for col in sale_fe.columns if '_fe' in col])
    dtypes_opt(sale_fe)
    return sale_fe

df=train_format()
calendar=calendar_fe() 
df=df.merge(calendar,on='d',copy=False)

dummy_state=pd.get_dummies(df.state_id).rename(columns={0:'CA',1:'TX',2:'WI'})
df=pd.concat([df,dummy_state],axis=1)
df['snap']=df['snap_CA']*df['CA']+df['snap_TX']*df['TX']+df['snap_WI']*df['WI']
remove_feat=['snap_CA','snap_TX','snap_WI','CA','TX','WI']
for feat in remove_feat: 
    del df[feat]
dtypes_opt(df)    
del dummy_state,calendar;gc.collect() 

price=price_fe()
df=df.merge(price,on=['id','wm_yr_wk'],copy=False)
del df['wm_yr_wk'],price;gc.collect()

sale=sale_fe(df)
sale.to_pickle('sale.pkl')

df=df.drop('sale',axis=1).merge(sale,on=['id','d'])
del sale;gc.collect()
train=df.query("d<{}".format(val_start))
# training after 50 days
train_shift=50
train['flag']=train[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(train_shift)).astype('float16')
train.dropna(subset=['flag'],inplace=True)
del train['flag']

remove_col=['id','d','sale']
train_col=[col for col in list(train) if col not in remove_col]
X=train[train_col]
Y=train[TARGET]


np.random.seed(42)
val_index = np.random.choice(X.index.values, int(X.shape[0]/20), replace = False)
train_index = np.setdiff1d(X.index.values, val_index)
cat_feats=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
           'event_name_1','event_type_1', 'event_name_2', 'event_type_2','snap',
           'y', 'qua', 'mon', 'wy', 'dw', 'dm', 'weekend']
           
X_train = lgb.Dataset(X.loc[train_index] , label = Y.loc[train_index], 
                      categorical_feature=cat_feats, free_raw_data=False)
X_val = lgb.Dataset(X.loc[val_index], label = Y.loc[val_index],
                    categorical_feature=cat_feats,free_raw_data=False)

del  X, Y ; gc.collect()
params = {
            'boosting_type': 'gbdt',
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'metric': 'rmse',
            'subsample': 0.5,
            'subsample_freq': 1,
            'learning_rate': 0.03,
            'num_leaves': 2**11-1,
            'min_data_in_leaf': 2**12-1,
            'feature_fraction': 0.5,
            'max_bin': 100,
            'n_estimators': 1500,
            'boost_from_average': False,
            'verbose': 20,
            'early_stopping_round':50
            } 
m_lgb = lgb.train(params, X_train, valid_sets = [X_train,X_val], verbose_eval=20) 
m_lgb.save_model("final_model.lgb")
pred=train_format(predict=True)
calendar=calendar_fe() 

pred=pred.merge(calendar,on='d',copy=False)

dummy_state=pd.get_dummies(pred.state_id).rename(columns={0:'CA',1:'TX',2:'WI'})
pred=pd.concat([pred,dummy_state],axis=1)
pred['snap']=pred['snap_CA']*pred['CA']+pred['snap_TX']*pred['TX']+pred['snap_WI']*pred['WI']
remove_feat=['snap_CA','snap_TX','snap_WI','CA','TX','WI']
for feat in remove_feat: 
    del pred[feat]
dtypes_opt(pred)    
del dummy_state,calendar;gc.collect() 

price=price_fe()
pred=pred.merge(price,on=['id','wm_yr_wk'],copy=False)
del pred['wm_yr_wk'],price;gc.collect()

pred_sale=sale_fe(pred,predict=True)
pred=pred.drop('sale',axis=1).merge(pred_sale,on=['id','d'])
def str_fun(x):
    x=x[:-10]
    x=x+'validation'
    return x
val=pred[(pred.d>=1914)&(pred.d<=1941)]
val.id=val.id.apply(str_fun)
estimator = lgb.Booster(model_file='final_model.lgb')
val['pred_sale']=estimator.predict(val.drop(['id','d','sale'],axis=1))

pred_val=val[['id','d','pred_sale']]

roll_fe=pd.read_pickle('roll_fe.pkl')
tmp_pred=pred[pred.d>=1949]
tmp_pred=tmp_pred.merge(roll_fe,on='id',how='left')

tmp_pred['7rmax_7days_fe']=tmp_pred['tmp_7rmax_7days_fe']
tmp_pred['7rmin_7days_fe']=tmp_pred['tmp_7rmin_7days_fe']
tmp_pred['7rstd_7days_fe']=tmp_pred['tmp_7rstd_7days_fe']
tmp_pred['7rmax_28days_fe']=tmp_pred['tmp_7rmax_28days_fe']
tmp_pred['7rmin_28days_fe']=tmp_pred['tmp_7rmin_28days_fe']
tmp_pred['7rstd_28days_fe']=tmp_pred['tmp_7rstd_28days_fe']

tmp_pred=tmp_pred.drop(['tmp_7rmax_7days_fe','tmp_7rmin_7days_fe','tmp_7rstd_7days_fe','tmp_7rmax_28days_fe','tmp_7rmin_28days_fe','tmp_7rstd_28days_fe'],axis=1)
pred=pd.concat([pred[pred.d<1949],tmp_pred])
pred=pred.sort_values(by=['id','d']).reset_index(drop=True)
pred[pred.id=='FOODS_3_825_WI_3_evaluation']
all_preds = pd.DataFrame()
day=7
windows=[7,28]
TARGET='sale'
main_time = time.time()

for PREDICT_DAY in range(1,29):    
    print('Predict | Day:', PREDICT_DAY)
    start_time = time.time()

    grid_df = pred.copy()
    grid_df['shift_1_sale_fe']=grid_df[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(1)).astype('float16')
    grid_df['1rmean_7days_fe']=grid_df[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(1).rolling(7).mean()).astype('float16')    
    print('Shift:',day)
    col_name='shift_'+str(day)+'_sale'+'_fe'
    grid_df[col_name]=grid_df[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(day)).astype('float16')
    for window in windows:
        print('Window:',window)        
        grid_df[str(day)+'rmean_'+str(window)+'days'+'_fe']=grid_df[['id','d','sale']].groupby(['id'])['sale'].transform(lambda x: x.shift(day).rolling(window).mean()).astype('float16')
  

    print('Sale Feature Time:',time.time()-start_time)
        

        
    estimator = lgb.Booster(model_file='final_model.lgb')
    day_mask = pred['d']==(1941+PREDICT_DAY)

    mask = day_mask
    pred[TARGET][mask] = estimator.predict(grid_df[mask].drop(['id','d','sale'],axis=1))


    temp_df = pred[day_mask][['id',TARGET]]
    temp_df.columns = ['id','F'+str(PREDICT_DAY)]
    if 'id' in list(all_preds):
        all_preds = all_preds.merge(temp_df, on=['id'], how='left')
    else:
        all_preds = temp_df.copy()
        
    print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                  ' %0.2f min total |' % ((time.time() - main_time) / 60),
                  ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
    del temp_df
    
pred_eva = all_preds.reset_index(drop=True)