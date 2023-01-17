# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Source: https://www.kaggle.com/nictosi/m5-forecasting-starter-data-exploration/edit

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from itertools import cycle
pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
# RSME
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
# Read in the data
INPUT_DIR = '../input/m5-forecasting-accuracy'
cal = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
stv = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')
ss = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
sellp = pd.read_csv(f'{INPUT_DIR}/sell_prices.csv')
ss = ss.set_index('id')
stv['item_store_id'] = stv['item_id'] + '_'+  stv['store_id']
stv = stv.set_index('item_store_id')
d_cols = [c for c in stv.columns if 'd_' in c] # sales data columns
past_sales = stv[d_cols] \
    .T \
    .merge(cal.set_index('d')['date'],
           left_index=True,
           right_index=True,
            validate='1:1') \
    .set_index('date')
def plot_single_product(item_store_id):
    d_cols = [c for c in stv.columns if 'd_' in c] # sales data columns

    # Below we are chaining the following steps in pandas:
    stv.loc[item_store_id] \
        [d_cols[-365:]] \
        .T \
        .plot(figsize=(15, 5),
              title=item_store_id  + ' sales by "d" number',
              color=next(color_cycle))
    plt.legend('')
    plt.show()
    
plot_single_product('FOODS_3_120_CA_3')
feat = stv.copy()
feat['Mean'] = np.mean(feat,axis=1)
feat['Std'] = np.std(feat,axis=1)
feat['Total'] = np.sum(feat,axis=1)
feat = feat.sort_values(['Total'], ascending=False)
#feat = feat.set_index('item_store_id')
feat.head()
fig, ax = plt.subplots(figsize=(40,10))  
plt.scatter(feat['Mean'],np.log(feat['Std']/feat['Mean']),marker = 'd', s=3)
plt.ylabel('log(Std/Mean)')
plt.xlabel('Mean')
ax.set_xlim(0,20)
for i in stv['cat_id'].unique():
    items_col = [c for c in past_sales.columns if i in c]
    past_sales[items_col] \
        .sum(axis=1) \
        .plot(figsize=(15, 5),
              alpha=0.8,
              title='Total Sales by Item Category')
plt.legend(stv['cat_id'].unique())
plt.show()
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

model = Ridge()
def ridge_predict2(data, label, predicted_days = 30, n_days_back=60, verbose = False):
    
    if (verbose):
        print('Fitting ' + label + " ...")
    
    # format ucdata and extend dates to the 
    ucdata = pd.DataFrame(data.copy())
    ucdata['date'] = pd.to_datetime(ucdata.index)
    
    ucdata = ucdata.append(pd.DataFrame({'date': pd.date_range(start=ucdata.index[-1], periods=predicted_days, freq='D', closed='right')}), sort=True)
    ucdata = ucdata.set_index('date')
    
    # compute shifts
    shifts = np.arange(predicted_days,predicted_days+n_days_back)
    for i in shifts:
        ucdata['lag_{}'.format(i)] = ucdata[label].shift(i).fillna(0)
    
    Y = ucdata[label]
    ucdata = ucdata.drop(label,1)
    
    ## fit ridge regression
    X_train = ucdata[:-predicted_days]
    X_test = ucdata[-predicted_days:]
    y_train = Y[:-predicted_days]
    y_test = Y[-predicted_days:]
    ground_truth = Y[-predicted_days:]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred[pred<0] = 0
    
    
    if(verbose):
        fig, ax = plt.subplots(figsize=(40, 5))
        plt.title("Prediction of " + label + " with Ridge regression looking " + str(n_days_back) + " days back")
        plt.plot(X_train.index, y_train, color='b', label='train set')
        plt.plot(X_train.index, model.predict(X_train), color='c', label='prediction on train set')
        plt.plot(X_test.index, pred, color='r', label='future prediction')
        plt.legend()
        plt.xticks(rotation=90)
    
    
    
    return pred
 
def avg_predict(data, label, predicted_days = 30, n_days_back=60, verbose = False):
        pred = np.ones(predicted_days) * np.nanmean(data[-28:]) # preding with avg
        
        if(verbose):
            fig, ax = plt.subplots(figsize=(40, 5))
            plt.title("Prediction of " + label + " with Ridge regression looking " + str(n_days_back) + " days back")
            plt.plot(data[:-predicted_days].index, data[:-predicted_days].values, color='b', label='train set')
            plt.plot(data[-predicted_days:].index, pred, color='r', label='future prediction')
            plt.legend()
            plt.xticks(rotation=90)
            
        return pred
# timer test
import time

n = 10
timelist = []
for i in np.arange(0,n):
    tic = time.perf_counter()  
    item = feat.index[i]
    print(item)
    ucdata_raw = past_sales[item][-365:].interpolate('linear')
    ridge_predict2(ucdata_raw, item, 28,120,False)
    toc = time.perf_counter()
    timelist.append(toc-tic)    
    
avg_time = np.mean(timelist) 
print(f"average time: {avg_time:0.3f}s")    
#save submission for d_1914 - d_1941
from datetime import timedelta  
from datetime import datetime
import math

days_to_predict = 28
items_to_predict = 3 # set to inf if you want to predict the whole dataset
zero_threshold = 0.3 # percentage of zero values in the train set to activate the baseline prediction
sub = pd.DataFrame(ss.copy())
#sub = sub.set_index('id')

items = [i.replace('_validation','') for i in sub.index[sub.index.str.contains("validation")]]

tot_items = np.min([items_to_predict, len(items)])
print('predicting: ' + str(tot_items) + ' items')
count = 0
count_avg = 0
count_ridge = 0
for prod in items[0: tot_items]:    
    if(count % 5000 == 0):
        print('Predicting ... ' + str(count) + " / " + str(items_to_predict) + ' completed')
        print(f'ETA: {avg_time * (items_to_predict - count):0.4f}s')
       
    try:
        ucdata_raw = past_sales[prod][-365:]
        if((ucdata_raw.values == 0).sum() > zero_threshold * len(ucdata_raw)):
            pred = avg_predict(ucdata_raw, prod, days_to_predict,120, True)
            count_avg += 1
        else:        
            # predict next month
            ucdata = ucdata_raw.interpolate('linear')
            pred = ridge_predict2(ucdata, prod, days_to_predict,120, True) 
            count_ridge += 1
        
        for i in np.arange(0,days_to_predict):
            sub.loc[prod + '_validation','F' + str(i+1)] = pred[i]        
    except:
          print("issue with item " + prod)
       
    count += 1

print(str(count_ridge) + ' items predicted with Ridge')
print(str(count_avg) + ' items predicted with avg baseline')

filename = 'submission_ridge' + str(datetime.now()) + "_zero_th_" + str(zero_threshold) + '.csv'
print('saving ' + filename)
sub.to_csv(filename, index=True)
# quantify error and order by error
import random

verbose = False

items_to_predict = 2
print('predicting: ' + str(items_to_predict) + ' items')

items = [i for i in feat.index]
random.shuffle(items)

res = pd.DataFrame(columns=['prod' , 'ridge_rmse', 'baseline_rmse', 'zeros_perc' , 'mean' , 'std'])

count = 0
for prod in items[0:items_to_predict]:
    if(count % 1000 == 0):
        print('Comparing items ... ' + str(count) + " / " + str(items_to_predict) + ' completed') 
        
    d = past_sales[prod][-365:-28]
    actual = past_sales[prod][-28:]
        
    # predict next month
    baseline = np.ones(days_to_predict) * np.mean(d[-28:])
    pred = ridge_predict2(d, prod, days_to_predict,120, False)   
    zeros_perc = (d.values == 0).sum()/ len(d)
    avg = np.nanmean(d.values)
    std = np.nanstd(d.values)
    
    rms = rmse(pred, actual)
    blrms = rmse(baseline, actual)   
    res = res.append({'prod' : prod , 'ridge_rmse' : rms, 'baseline_rmse' : blrms, 'zeros_perc' : zeros_perc, 'mean' : avg, 'std' : std} , ignore_index=True)
    count +=1
    
res['ridge-baseline'] = res['ridge_rmse'] - res['baseline_rmse']    
print('done!')

sp = sellp.copy()
sp = sp[sp['wm_yr_wk'] == np.max(sp['wm_yr_wk'])]
sp['item_store_id'] = sp['item_id'] +'_' +  sp['store_id']
res = pd.merge(res,sp[['item_store_id','sell_price']], how='inner', left_on='prod', right_on = 'item_store_id' )
res.to_csv('error_analysis.csv', index=True)
# plot error analysis
fig, ax = plt.subplots(figsize=(40, 20))
plt.scatter(res['ridge_rmse'], res['sell_price_x']*res['mean'],marker='o')
def evaluate_wrmsse(y_train, y_test, y_pred, sell_prices):
    y_train_weight = y_train.iloc[:,-30:].sum(axis = 1)
    sell_prices_weight = sell_prices[sell_prices.wm_yr_wk == 11621] # imprecise, I'm assuming same price last 5 weeks
    weights = y_train_weight.to_numpy() * sell_prices_weight.sell_price.to_numpy()
    weights_sum = sum(weights)
    weights = [x/weights_sum for x in weights]
    numerators = [np.sum((y_test.iloc[i,:].to_numpy() - y_pred.iloc[i,:].to_numpy())**2) for i in range(y_test.shape[0])]
    rmsse = [np.sqrt(1/30 * numerators[i] / denominators[i]) for i in range(len(numerators))]
    wrmsse = np.sum(np.array(weights) * np.array(rmsse))
    return wrmsse

