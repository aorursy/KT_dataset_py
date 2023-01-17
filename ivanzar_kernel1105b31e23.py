import numpy as np
import pandas as pd 
pd.set_option('display.max_rows', 2500)
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_columns', 500)
date_columns = ['delivery_status_log.5.datetime', 
       'preorder_time', 
       'delivery_status_log.8.datetime', 
       'delivery_status_log.9.datetime', 
       'delivery_status_log.0.datetime',
       'delivery_status_log.15.datetime', 
       'delivery_status_log.18.datetime', 
       'delivery_status_log.19.datetime', 
       'delivery_status_log.4.datetime', 
       'delivery_status_log.13.datetime',
       'time_received',
       'delivery_status_log.16.datetime', 
       'delivery_status_log.10.datetime',
       'delivery_status_log.11.datetime', 
       'delivery_status_log.12.datetime',
       'delivery_status_log.14.datetime',
       'time_delivered',
       'delivery_status_log.3.datetime',
       'client_pre_estimate',
       'delivery_status_log.17.datetime', 
       'delivery_status_log.2.datetime',
       'pickup_eta',
       'delivery_status_log.1.datetime',
       'delivery_status_log.7.datetime', 
       'delivery_status_log.6.datetime']
df_trans = pd.read_csv('/kaggle/input/purchases-1/purchases.csv', index_col=['_id'], parse_dates=date_columns)
#Churn column calculation
#Churn definition: if after an order customer didn't use service for more than 180 days they are considered churned in this period
df_trans['next_date_time_received'] = df_trans.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['time_received'].shift(-1)
df_trans['days_to_next_purch'] = (df_trans['next_date_time_received'] - df_trans['time_received']) 
df_trans['if_churned'] = (df_trans['days_to_next_purch'] > pd.Timedelta(days=90)) | (pd.isnull(df_trans.next_date_time_received) & (df_trans.time_received < pd.to_datetime('2018-01-01').tz_localize('Europe/Helsinki')))
df_trans.to_csv('churned_result.csv')
#Feature engineering
#Feature 1. Review rating

review_rating = pd.get_dummies(df_trans['review.rating'], drop_first=True, prefix="rating")

for i in range(5):
    if 'review_rating_av' in locals():
        n = df_trans.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['review.rating'].shift(1+i)
        n.name = ('Review_rating_t-'+str(i+1))
        review_rating_av = pd.concat([review_rating_av, n], axis=1)
    else:
        review_rating_av = pd.DataFrame(df_trans.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['review.rating'].shift(1+i))
        review_rating_av = review_rating_av.rename(columns={'review.rating': 'Review_rating_t-1'})
        
review_rating_av = review_rating_av.sum(axis=1, skipna=True)/review_rating_av.count(axis=1)

review_rating_av[review_rating_av<0] = 0
review_rating_av = pd.cut(review_rating_av, 5, labels=['less1', '1to2', '2to3', '3to4', '4to5'])
review_rating_av.name = 'RatingLast5Av'
review_rating_av = pd.get_dummies(review_rating_av, prefix="last_5_av_rating")
review_rating = pd.concat([review_rating, review_rating_av], axis=1)
review_rating
#Feature 2. Whether the ordered was delayed or delivered faster than estimated: last order and average of last 5 orders

import re
def classify_estimate(n):
    n = str(n)
    n = re.sub("\+", "", n)
    
    if re.search('-', n) == None:
        return float(n)

    else:
        return float(n.split("-")[1])
  
client_pre_estimate = df_trans['client_pre_estimate'].apply(classify_estimate)
delivery_duration = df_trans['time_delivered']-df_trans['time_received']
delivery_duration = delivery_duration.apply(lambda x: x.seconds/60)
overperform_delay = delivery_duration - client_pre_estimate

overperform_delay.name = 'overperform_delay'
df_od  = pd.concat([df_trans[['from_field.id', 'time_received']], overperform_delay], axis=1)
for i in range(5):
    if 'overperform_delay_av' in locals():
        n = df_od.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['overperform_delay'].shift(1+i)
        n.name = ('od_t-'+str(i+1))
        overperform_delay_av = pd.concat([overperform_delay_av, n], axis=1)
    else:
        overperform_delay_av = pd.DataFrame(df_od.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['overperform_delay'].shift(1+i))
        overperform_delay_av = overperform_delay_av.rename(columns={'overperform_delay': 'od_t-1'})


overperform_delay_av['od_cum'] = overperform_delay_av.sum(axis=1, skipna=True)
overperform_delay_av['od_count'] = overperform_delay_av.loc[:,'od_t-1':'od_t-5'].count(axis=1)
overperform_delay_av['overperf_delay_l5_av'] =  overperform_delay_av['od_cum']/overperform_delay_av['od_count']
overperform_delay = pd.concat([overperform_delay, overperform_delay_av['overperf_delay_l5_av']], axis=1)
overperform_delay = overperform_delay.fillna(overperform_delay.median())
overperform_delay
#Feature 3. Status of delivery, whether the order was delivered last 5 times, wheter the order was rejected last 5 times
status = pd.get_dummies(df_trans['status'], drop_first=True, prefix="status")

if 'delivered_stats' in locals():
    del delivered_stats

for i in range(10):
    if 'delivered_stats' in locals():
        n = df_trans.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['status'].shift(1+i)
        n.name = ('status_t-'+str(i+1))
        delivered_stats = pd.concat([delivered_stats, n], axis=1)
    else:
        delivered_stats = pd.DataFrame(df_trans.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['status'].shift(1+i))
        delivered_stats = delivered_stats.rename(columns={'status': 'status_t-1'})

dcum = pd.DataFrame(index = delivered_stats.index)
dcum['delivered_count'] = 0
for i in range(10):
    d = (delivered_stats['status_t-'+str(i+1)]=='delivered').astype('int')
    dcum['delivered_count'] = dcum['delivered_count'] + d
rcum = pd.DataFrame(index = delivered_stats.index)
rcum['rejected_count'] = 0
for i in range(10):
    r = (delivered_stats['status_t-'+str(i+1)]=='rejected').astype('int')
    rcum['rejected_count'] = rcum['rejected_count'] + r
    
status = pd.concat([status, dcum, rcum], axis=1)
status
#Feature 4. Average delivery time
df_with_dd = df_trans
df_with_dd['delivery_duration'] = df_trans['time_delivered']-df_trans['time_received']
df_with_dd['delivery_duration'] = df_with_dd['delivery_duration'].apply(lambda x: x.seconds)/60

for i in range(5):
    if 'delivery_duration_av' in locals():
        n = df_trans.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['delivery_duration'].shift(1+i)
        n.name = ('Delivery duration_t-'+str(i+1))
        delivery_duration_av = pd.concat([delivery_duration_av, n], axis=1)
    else:
        delivery_duration_av = pd.DataFrame(df_trans.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['delivery_duration'].shift(1+i))
        delivery_duration_av = delivery_duration_av.rename(columns={'delivery_duration': 'Delivery duration_t-1'})
delivery_duration_av = delivery_duration_av.sum(axis=1, skipna=True)/delivery_duration_av.count(axis=1)
delivery_duration_av = delivery_duration_av.fillna(delivery_duration_av.median())
delivery_duration_av.name = 'delivery_duration_last_5'
delivery_duration_av
#Feature 5. Delivery method. Share of homedelivery during the last 10 visits
if 'delivery_method' in locals():
    del delivery_method

for i in range(10):
    if 'delivery_method' in locals():
        n = df_trans.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['delivery_method'].shift(1+i)
        n.name = ('Delivery method_t-'+str(i+1))
        delivery_method = pd.concat([delivery_method, n], axis=1)
    else:
        delivery_method = pd.DataFrame(df_trans.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['delivery_method'].shift(1+i))
        delivery_method = delivery_method.rename(columns={'delivery_method': 'Delivery method_t-1'})

mcum = pd.DataFrame(index = delivery_method.index)
mcum['hd_count'] = 0
for i in range(10):
    m = (delivery_method['Delivery method_t-'+str(i+1)]=='homedelivery').astype('int')
    mcum['hd_count'] = mcum['hd_count'] + m
n_orders = delivery_method.loc[:,'Delivery method_t-1':'Delivery method_t-10'].count(axis=1)
n_orders.name = 'n_orders'
delivery_method = pd.concat([delivery_method, mcum, n_orders], axis=1)
home_delivery_share = delivery_method['hd_count']/delivery_method['n_orders']
home_delivery_share = home_delivery_share.fillna(0)
home_delivery_share.name = 'home_delivery_share'
home_delivery_share
#Feature 6. Usage of credits. Number of uses in last 10 orders

if 'credits_use_n' in locals():
    del credits_use_n


for i in range(10):
    if 'credits_use_n' in locals():
        n = df_trans.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['credits_used_amount'].shift(1+i)
        n.name = ('credits_used_t-'+str(i+1))
        credits_use_n = pd.concat([credits_use_n, n], axis=1)
    else:
        credits_use_n = pd.DataFrame(df_trans.sort_values(by=['from_field.id', 'time_received']).groupby('from_field.id')['credits_used_amount'].shift(1+i))
        credits_use_n = credits_use_n.rename(columns={'credits_used_amount': 'credits_used_t-1'})

mcum = pd.DataFrame(index = credits_use_n.index)
mcum['credits_count'] = 0
for i in range(10):
    m = (credits_use_n['credits_used_t-'+str(i+1)]>0).astype('int')
    mcum['credits_count'] = mcum['credits_count'] + m
credits_use_n = mcum
credits_use_n.name = 'credits_used_last10orders'
credits_use_n
df_model_input = pd.concat([df_trans['if_churned'], review_rating, overperform_delay, status, delivery_duration_av, home_delivery_share, credits_use_n], axis=1)
df_model_input_na = df_model_input[df_model_input.isna().any(axis=1)]
df_model_input_na 
#Test and train test split
sample_size = round(df_model_input.shape[0]*0.20)
df_test = df_model_input.sample(sample_size)
df_train = df_model_input.drop(df_test.index)

df_train[df_train['if_churned'] == np.inf]
df_train.columns
factors = ['rating_0.0', 'rating_1.0', 'rating_2.0', 'rating_3.0',
       'rating_4.0', 'rating_5.0', 'last_5_av_rating_less1',
       'last_5_av_rating_1to2', 'last_5_av_rating_2to3',
       'last_5_av_rating_3to4', 'last_5_av_rating_4to5', 'overperform_delay',
       'overperf_delay_l5_av', 'status_delivered',
       'status_pending_transaction', 'status_process_payment_failed',
       'status_production', 'status_ready', 'status_received',
       'status_refunded', 'status_rejected', 'delivered_count',
       'rejected_count', 'delivery_duration_last_5', 'home_delivery_share',
       'credits_count']
df_x = pd.DataFrame(data=df_train, columns=factors)
df_y = pd.DataFrame(data=df_train, columns=['if_churned'])
df_x_test = pd.DataFrame(data=df_test, columns=factors)
df_y_test = pd.DataFrame(data=df_test, columns=['if_churned'])
df_x
import statsmodels.api as sm
from sklearn import metrics
from matplotlib import pyplot
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
logit_model=sm.Logit(df_y,df_x)
result=logit_model.fit(method='bfgs')
print(result.summary2())
#Testing the logistic regression model
test_output = result.predict(df_x_test)
fpr, tpr, thresholds = metrics.roc_curve(df_y_test, test_output)
print("AUC: {}".format(metrics.auc(fpr, tpr)))
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
xg_reg = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(df_x,df_y)
xg_reg
#Testing XGBoost
output_xgboost = xg_reg.predict(df_x_test)
fpr_xgb, tpr_xgb, thresholds_xgb = metrics.roc_curve(df_y_test, output_xgboost)
print("AUC: {}".format(metrics.auc(fpr_xgb, tpr_xgb)))
pyplot.plot(fpr_xgb, tpr_xgb, marker='.', label='XGBoost')
xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [100, 50]
plt.show()
xgb.plot_importance(xg_reg)
plt.show()
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(df_x, df_y)
predictions_rf = rf.predict(df_x_test)
fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(df_y_test, predictions_rf)
print("AUC: {}".format(metrics.auc(fpr_rf, tpr_rf)))
pyplot.plot(fpr_rf, tpr_rf, marker='.', label='RandomForest')
importances = rf.feature_importances_
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(df_x.columns, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
errors = abs(predictions_rf - np.array(df_y_test))
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = 100 * errors/np.array(df_y_test)
np.mean(mape)
#accuracy = 100 - np.mean(mape)
#print('Accuracy:', round(accuracy, 2), '%.')

conf_mat = confusion_matrix(df_y_test, predictions_rf>0.2)
conf_mat

