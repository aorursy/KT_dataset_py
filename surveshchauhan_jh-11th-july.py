import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error#,roc_curve,auc
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from datetime import datetime as dt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))
#dateparse = lambda x: pd.datetime.strptime(x, '%d/%-m/Y')
df_train = pd.read_csv("/kaggle/input/train_0irEZ2H.csv",parse_dates=['new_date'])
df_test = pd.read_csv("/kaggle/input/test_nfaJ3J5.csv",parse_dates=['new_date'])
sample_submission = pd.read_csv("/kaggle/input/sample_submission.csv")

df_train.head()
df_train.head()
df_test.head()
df_train.isna().sum()
def create_sales_agg_monthwise_features(df, gpby_cols, target_col, agg_funcs):
    '''
    Creates various sales agg features with given agg functions  
    '''
    gpby = df.groupby(gpby_cols)
    newdf = df[gpby_cols].drop_duplicates().reset_index(drop=True)
    for agg_name, agg_func in agg_funcs.items():
        aggdf = gpby[target_col].agg(agg_func).reset_index()
        aggdf.rename(columns={target_col:target_col+'_'+agg_name}, inplace=True)
        newdf = newdf.merge(aggdf, on=gpby_cols, how='left')
    return newdf

def create_sales_lag_feats(df, gpby_cols, target_col, lags):
    gpby = df.groupby(gpby_cols)
    for i in lags:
        df['_'.join([target_col, 'lag', str(i)])] = \
                gpby[target_col].shift(i).values #+ np.random.normal(scale=1.6, size=(len(df),))
    return df
df_train.total_price.fillna(df_train.total_price.median(),inplace=True)
df_train['class']=1
df_test['class']=2
df_train.sort_values(by=['store_id','sku_id','class','new_date'],inplace=True)
df_test.sort_values(by=['store_id','sku_id','class','new_date'],inplace=True)
df_train['units_sold'] = np.log1p(df_train['units_sold'])

combined= pd.concat([df_train,df_test],axis=0) 
combined['weekday'] = combined['new_date'].dt.weekofyear
combined['QTR'] = combined['new_date'].apply(lambda x: x.quarter)
combined['month'] = combined['new_date'].apply(lambda x: x.month)
combined['year'] = combined['new_date'].dt.year


combined['total_price'] = np.log1p(combined['total_price'])
combined['base_price'] = np.log1p(combined['base_price'])
combined['diff'] = combined['total_price'] - combined['base_price']
combined['relative_diff_base'] = combined['diff']/combined['base_price']
combined['relative_diff_total'] = combined['diff']/combined['total_price']

# combined['discount_avail'] = np.where(combined['total_price'] > combined['base_price']
#                                       ,'1'
#                                       ,np.where(combined['base_price'] > combined['total_price']
#                                                ,'2'
#                                                ,'3')
#                                       )
# combined['discount_rate'] = np.where(combined['base_price'] > combined['total_price']
#                                       ,round(100*(combined['base_price'] - combined['total_price'])
#                                              /combined['base_price'],2)
#                                       ,0
#                                       )

# combined['surcharge'] = np.where(combined['total_price'] > combined['base_price']
#                                       ,round(100*(combined['total_price'] - combined['base_price'])
#                                              /combined['total_price'],2)
#                                       ,0
#                                       )


month_smry= (
        (combined.groupby(['month']).agg([np.nanmean]).units_sold - np.nanmean(combined.units_sold) ) / 
        np.nanmean(combined.units_sold)
).rename(columns={'nanmean':'month_mod'})
combined=combined.join(month_smry,how='left',on='month')

qtr_smry= (
        ( combined.groupby(['QTR']).agg([np.nanmean]).units_sold - np.nanmean(combined.units_sold) ) / 
        np.nanmean(combined.units_sold)
).rename(columns={'nanmean':'QTR_mod'})
combined=combined.join(qtr_smry,how='left',on='QTR')



# medians = pd.DataFrame({'Average price' :combined[combined['class']==1].groupby(by=['store_id','sku_id','QTR','year'])['total_price'].mean()}).reset_index()
# combined = combined.merge(medians, how = 'outer', on = ['store_id','sku_id','QTR','year'])
# medians = pd.DataFrame({'Monthly Sales' :combined[combined['class']==1].groupby(by=['store_id','sku_id','QTR','year'])['units_sold'].mean()}).reset_index()
# combined = combined.merge(medians, how = 'outer', on = ['store_id','sku_id','QTR','year'])

combined.sort_values(by=['store_id','sku_id','class','new_date'],inplace=True)
combined = create_sales_lag_feats(combined, gpby_cols=['store_id','sku_id'], target_col='units_sold', 
                               lags=[12,13,14,15,16,17,18,19,20,21,22,23,24])
combined = create_sales_lag_feats(combined, gpby_cols=['store_id','sku_id'], target_col='total_price', 
                               lags=[12,13,14,15,16,17,18,19,20,21,22,23,24])
combined = create_sales_lag_feats(combined, gpby_cols=['store_id','sku_id'], target_col='base_price', 
                                lags=[12,13,14,15,16,17,18,19,20,21,22,23,24])
# combined = create_sales_lag_feats(combined, gpby_cols=['store_id','sku_id'], target_col='is_featured_sku', 
#                            lags=[12,16])
# combined = create_sales_lag_feats(combined, gpby_cols=['store_id','sku_id'], target_col='is_display_sku', 
#                            lags=[12,16])
#total_price	base_price
combined
df_train = combined[combined['class']==1]
df_test  = combined[combined['class']==2]
df_train.columns
target = 'units_sold'
predictors=list(combined.columns)
predictors.remove("record_ID")
predictors.remove("week")
predictors.remove("new_date")
predictors.remove("units_sold")
#predictors.remove('is_featured_sku')
#predictors.remove('is_display_sku')
predictors.remove('class')
df_train.dropna(inplace=True)
X = df_train[predictors]
#df_train[target]= np.log1p(df_train[target])
Y=df_train[target]

val_size = 0.10
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size,random_state=7)

# X_val = df_train[(df_train.year==2013) & (df_train.month.isin([6,7]))][predictors]
# X_train = df_train[(df_train.year.isin([2011,2012,2013])) & (~df_train.month.isin([6,7]))][predictors]
# Y_val = df_train[(df_train.year==2013) & (df_train.month.isin([6,7]))][target]
# Y_train = df_train[(df_train.year.isin([2011,2012,2013])) & (~df_train.month.isin([6,7]))][target]

predictors
clf = RandomForestRegressor(n_jobs=-1,random_state=7,n_estimators=100,verbose=True)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_val)
#y_pred = np.exp(y_pred)
rms = rmsle(np.exp(Y_val),np.exp(y_pred))
print(1000*rms)
# Create prediciton test set
X_react = combined[combined['class']==2][predictors]
y_pred_test = clf.predict(X_react)
submission_df = df_test.copy()
submission_df['units_sold'] = np.round(np.exp(y_pred_test),0)
submission_df = submission_df[['record_ID', 'units_sold']]
submission_df.to_csv('submission_rf.csv', index=False)
params = {
    'boosting_type': 'dart',
    'objective': 'regression',
    'metric': 'rmsle',
    'max_depth': 6, 
    'learning_rate': .1,
    #'colsample_bytree': [0.8],
    'verbose': 0#, 
    #'early_stopping_round': 20
}
cat_columns = ['store_id','sku_id','weekday','QTR','month','year','is_display_sku','is_featured_sku']
#,'is_featured_sku_lag_12','is_featured_sku_lag_16','is_display_sku_lag_12','is_display_sku_lag_16' ]
for c in cat_columns:
    print(c)
    X[c] = X[c].astype('category')

for c in cat_columns:
    df_test[c] = df_test[c].astype('category')  
n_estimators = 1000

n_iters = 10
preds_buf = []
err_buf = []
for i in range(n_iters): 
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.10, random_state=i)
    #print(x_train,y_train)
    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_valid, label=y_valid)
    watchlist = [d_valid]

    model = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=1)

    preds = model.predict(x_valid)
    err = 1000 * rmsle(np.expm1(y_valid), np.expm1(preds))
    err_buf.append(err)
    print('RMSLE = ' + str(err))
    
    preds = model.predict(df_test[predictors])
    preds_buf.append(np.expm1(preds))

print('Mean RMSLE = ' + str(np.mean(err_buf)) + ' +/- ' + str(np.std(err_buf)))
# Average predictions
preds = np.mean(preds_buf, axis=0)

# Prepare submission
subm = pd.DataFrame()
subm['record_ID'] = df_test.record_ID.values
subm['units_sold'] = preds
subm.to_csv('submission_lgbm.csv', index=False)



