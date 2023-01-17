import os
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor
import joblib
from sklearn.metrics import mean_squared_error
sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
sales.name = 'sales'
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
calendar.name = 'calendar'
prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
prices.name = 'prices'
for d in range(1942,1970):
    col = 'd_' + str(d)
    sales[col] = 0
    sales[col] = sales[col].astype(np.int16)
def downcast(df):
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i,t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == np.object:
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    return df  

sales = downcast(sales)
prices = downcast(prices)
calendar = downcast(calendar)
df = pd.melt(sales, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='d', value_name='sold').dropna()
df = pd.merge(df, calendar, on='d', how='left')
df = pd.merge(df, prices, on=['store_id','item_id','wm_yr_wk'], how='left')
d_id = dict(zip(df.id.cat.codes, df.id))
d_item_id = dict(zip(df.item_id.cat.codes, df.item_id))
d_dept_id = dict(zip(df.dept_id.cat.codes, df.dept_id))
d_cat_id = dict(zip(df.cat_id.cat.codes, df.cat_id))
d_store_id = dict(zip(df.store_id.cat.codes, df.store_id))
d_state_id = dict(zip(df.state_id.cat.codes, df.state_id))
list1=['event_name_1','event_type_1','event_name_2','event_type_2']
from sklearn.preprocessing import LabelEncoder
for i in list1:
    df[i] = df[i].cat.add_categories("nan").fillna("nan")
    df[i]=LabelEncoder().fit_transform(df[i]).astype(np.int8)

df.d = df['d'].apply(lambda x: x.split('_')[1]).astype(np.int16)
cols = df.dtypes.index.tolist()
types = df.dtypes.values.tolist()
for i,type in enumerate(types):
    if type.name == 'category':
        df[cols[i]] = df[cols[i]].cat.codes
# extracting day_of_week has shown some memory errors, therefore I have dropped the date, 
# it is a good feature though, try to incorporate it and let me know the reason for error 
# df['date'] = df['date'].apply(lambda x: x.strftime('%d')).astype(np.int8)
df.drop(['date'],axis=1, inplace=True)
#make datatype of event as category
for i in list1:
    df[i]=df[i].astype('category')
# I have used 4 lags here in intervals of 7, all showed a good value of feature importance
lags = [28,35,42,49]
for lag in lags:
    df['sold_lag_'+str(lag)] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)['sold'].shift(lag).astype(np.float16)
# I have added 4 days surrounding the event as features
lags2 = [-2,-1,1,2]
for lag in lags2:
    df['event1_lag_'+str(lag)] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)['event_name_1'].shift(lag).astype(np.float16)
    df['event1_lag_'+str(lag)].fillna(100, inplace=True)
    df['event1_lag_'+str(lag)]=df['event1_lag_'+str(lag)].astype(np.int8)
    df['event1_lag_'+str(lag)]=df['event1_lag_'+str(lag)].astype('category')
# event type didn't showed a good feature importance, opposite to event itself
# for lag in lags2:
#     df['eventtype1_lag_'+str(lag)] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)['event_type_1'].shift(lag).astype(np.float16).fillna(100, inplace=True)
#     df['eventtype1_lag_'+str(lag)].fillna(100, inplace=True)
#     df['eventtype1_lag_'+str(lag)]=df['eventtype1_lag_'+str(lag)].astype(np.int8)
#     df['eventtype1_lag_'+str(lag)]=df['eventtype1_lag_'+str(lag)].astype('category')
df['item_sold_avg'] = df.groupby('item_id')['sold'].transform('mean').astype(np.float16)    
df['state_sold_avg'] = df.groupby('state_id')['sold'].transform('mean').astype(np.float16)    #total 3 unique values, 1 for each state
df['store_sold_avg'] = df.groupby('store_id')['sold'].transform('mean').astype(np.float16)  #10 unique values
df['cat_sold_avg'] = df.groupby('cat_id')['sold'].transform('mean').astype(np.float16)
df['dept_sold_avg'] = df.groupby('dept_id')['sold'].transform('mean').astype(np.float16)
df['cat_dept_sold_avg'] = df.groupby(['cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
df['store_item_sold_avg'] = df.groupby(['store_id','item_id'])['sold'].transform('mean').astype(np.float16)
df['cat_item_sold_avg'] = df.groupby(['cat_id','item_id'])['sold'].transform('mean').astype(np.float16)
df['dept_item_sold_avg'] = df.groupby(['dept_id','item_id'])['sold'].transform('mean').astype(np.float16)
df['state_store_sold_avg'] = df.groupby(['state_id','store_id'])['sold'].transform('mean').astype(np.float16)
df['state_store_cat_sold_avg'] = df.groupby(['state_id','store_id','cat_id'])['sold'].transform('mean').astype(np.float16)
df['store_cat_dept_sold_avg'] = df.groupby(['store_id','cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
df['wm_yr_wk_linear']=LabelEncoder().fit_transform(df['wm_yr_wk'].values).astype(np.int16)

df.drop(['wm_yr_wk'], axis=1, inplace=True)
df['price_lag'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)['sell_price'].shift(7).astype(np.float16)
df['price-diff']=df['price_lag']-df['sell_price']
df.drop(['price_lag'], axis=1, inplace=True)
df['sell_price'].fillna(-1,inplace=True)
df['decimal']=df['sell_price'].apply(lambda x: 100*(x-int(x))).astype(np.int16)
df['sell_price'].replace(-1,np.nan,inplace=True)
df['expanding_price_mean'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sell_price'].transform(lambda x: x.expanding(2).mean()).astype(np.float16)
df['diff_moving_mean']=df['expanding_price_mean']-df['sell_price']
df.drop(['expanding_price_mean'], axis=1, inplace=True)
df['price-diff']=df['price-diff'].astype(np.float16)
df.drop(['wday'], axis=1, inplace=True)
df['decimal']=df['decimal'].astype(np.int8)
df['year']=LabelEncoder().fit_transform(df['year']).astype(np.int8)
df['daily_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','d'])['sell_price'].transform('mean').astype(np.float16)
df['avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sell_price'].transform('mean').astype(np.float16)
df['selling_trend'] = (df['daily_avg_sold'] - df['avg_sold']).astype(np.float16)
df.drop(['daily_avg_sold','avg_sold'],axis=1,inplace=True)
list3=['cat_id','state_id']
for i in list3:
    df.drop([i], axis=1, inplace=True)
df = df[df['d']>=49]


df.to_pickle('data.pkl')
del df, sales, prices, calendar
gc.collect()
data = pd.read_pickle('data.pkl')
valid = data[(data['d']>=1599) & (data['d']<1942)][['id','d','sold']]  
valid_csv=data[(data['d']>=1914) & (data['d']<1942)][['id','d','sold']]
test = data[data['d']>=1942][['id','d','sold']]
eval_preds = test['sold']
valid_preds = valid['sold']
valid_preds_csv=valid_csv['sold']
cat_column=[]
for i in data.columns:
    if(str(data.dtypes[i])=='category'):
        cat_column.append(i)
for store in d_store_id:
    df = data[data['store_id']==store]
    
    #Split the data
    X_train, y_train = df[df['d']<1914].drop('sold',axis=1), df[df['d']<1914]['sold']
    # Uncover if you want to use last year for train
    #X_valid, y_valid = df[(df['d']>=1599) & (df['d']<1942)].drop('sold',axis=1), df[(df['d']>=1599) & (df['d']<1942)]['sold']
    X_valid_csv, y_valid_csv = df[(df['d']>=1914) & (df['d']<1942)].drop('sold',axis=1), df[(df['d']>=1914) & (df['d']<1942)]['sold']
    X_test = df[df['d']>=1942].drop('sold',axis=1)
    
    #Train and validate
    model = LGBMRegressor(
        learning_rate= 0.05,
        subsample=0.6,
        feature_fraction=0.6,
        num_iterations = 1200,
        max_bin=350,
        num_leaves= 100,
        lambda_l2=0.003,
        max_depth=200,
        min_data_in_leaf= 80,
        force_row_wise= True,
    )
    print('*****Prediction for Store: {}*****'.format(d_store_id[store]))
    model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_valid_csv,y_valid_csv)],
             eval_metric='rmse',  verbose=100, early_stopping_rounds=20,categorical_feature=cat_column)
    valid_preds_csv[X_valid_csv.index] = model.predict(X_valid_csv)
    eval_preds[X_test.index] = model.predict(X_test)
    filename = 'model'+str(d_store_id[store])+'.pkl'
    # save model
    joblib.dump(model, filename)
    del model, X_train, y_train, X_valid_csv, y_valid_csv
    gc.collect()
feature_importance_df = pd.DataFrame()
features = [f for f in data.columns if f != 'sold']
for filename in os.listdir('/kaggle/working/'):
    if 'model' in filename:
        # load model
        model = joblib.load(filename)
        store_importance_df = pd.DataFrame()
        store_importance_df["feature"] = features
        store_importance_df["importance"] = model.feature_importances_
        store_importance_df["store"] = filename[5:9]
        feature_importance_df = pd.concat([feature_importance_df, store_importance_df], axis=0)
    
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (averaged over store predictions)')
    plt.tight_layout()
    
display_importances(feature_importance_df)
#Get the actual validation results
valid_csv['sold'] = valid_preds_csv
validation = valid_csv[['id','d','sold']]
validation = pd.pivot(validation, index='id', columns='d', values='sold').reset_index()
validation.columns=['id'] + ['F' + str(i + 1) for i in range(28)]
validation.id = validation.id.map(d_id).str.replace('evaluation','validation')

#Get the evaluation results
test['sold'] = eval_preds
evaluation = test[['id','d','sold']]
evaluation = pd.pivot(evaluation, index='id', columns='d', values='sold').reset_index()
evaluation.columns=['id'] + ['F' + str(i + 1) for i in range(28)]
#Remap the category id to their respective categories
evaluation.id = evaluation.id.map(d_id)

#Prepare the submission
submit = pd.concat([validation,evaluation]).reset_index(drop=True)
# submit.memory_usage().sum()
submit.to_csv('submission.csv',index=False)