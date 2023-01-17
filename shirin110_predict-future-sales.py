import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

#from tqdm import tqdm

from tqdm.auto import tqdm

from xgboost import XGBRegressor

from xgboost import plot_importance

from sklearn.feature_selection import SelectFromModel

import pickle

from sklearn.metrics import mean_absolute_error

import math

from itertools import product

import seaborn as sns

#Load data



print('Loading Data..')

#rawSales = pd.read_csv( 'sales_train_v2.csv', sep=",")

rawSales = pd.read_csv( '../input/pfsextracted/sales_train_v2.csv', sep=",")



#rawItems = pd.read_csv( 'items.csv', sep=",")

rawItems = pd.read_csv( '../input/pfsextracted/items.csv', sep=",")

print('Converting to Date..')

rawSales['date'] = pd.to_datetime(rawSales['date'], format="%d.%m.%Y")

print('Creating month_year..')



#get month-year

#rawSales['month_year'] = pd.to_datetime(rawSales['date']).dt.to_period('M')

#rawSales['month_year'] = [adate.replace(day = 1) for adate in rawSales['date']]

rawSales = rawSales.sort_values(by='date')

print(rawSales.shape, min(rawSales['item_cnt_day']), max(rawSales['item_cnt_day']))

rawSales.tail()
#total number of items sold each block



result = rawSales.groupby(by = ['date_block_num']).agg({'item_cnt_day':'sum'})

result = result.reset_index()

result

plt.plot(result['date_block_num'], result['item_cnt_day'])

plt.xlabel('date_block_num', fontsize=10)

plt.ylabel('item_cnt_day', fontsize=10)

plt.xticks(result['date_block_num'], result['date_block_num'], fontsize=9, rotation=30)

plt.show()
#total sales each block



result = rawSales.loc[:, ['item_price','item_cnt_day','date_block_num']]#[['item_price', 'item_cnt_day', 'date_block_num']]

result['item_price'] = result['item_price'].fillna(0)

result['totsales'] = result['item_price']*result['item_cnt_day']

result = result.groupby(by = ['date_block_num']).agg({'totsales':'sum'})

result = result.reset_index()

result

plt.plot(result['date_block_num'], result['totsales'])

plt.xlabel('date_block_num', fontsize=10)

plt.ylabel('sales', fontsize=10)

plt.xticks(result['date_block_num'], result['date_block_num'], fontsize=9, rotation=30)

plt.show()
viewtrend1 = rawSales[(rawSales['item_id'] == 22167)]

viewtrend1 = viewtrend1.groupby(['date_block_num'], as_index=False).agg({"item_cnt_day": np.sum})

plt.plot(viewtrend1['date_block_num'], viewtrend1['item_cnt_day'])

plt.xlabel('date_block_num', fontsize=10)

plt.ylabel('item_cnt_day', fontsize=10)

plt.xticks(viewtrend1['date_block_num'], viewtrend1['date_block_num'], fontsize=9, rotation=30)

plt.show()



viewtrend2 = rawSales[(rawSales['item_id'] == 21377)]

viewtrend2 = viewtrend2.groupby(['date_block_num'], as_index=False).agg({"item_cnt_day": np.sum})

plt.plot(viewtrend2['date_block_num'], viewtrend2['item_cnt_day'])

plt.xlabel('date_block_num', fontsize=10)

plt.ylabel('item_cnt_day', fontsize=10)

plt.xticks(viewtrend2['date_block_num'], viewtrend2['date_block_num'], fontsize=9, rotation=30)

plt.show()



viewtrend3 = rawSales[(rawSales['item_id'] == 21386) ]

viewtrend3 = viewtrend3.groupby(['date_block_num'], as_index=False).agg({"item_cnt_day": np.sum})

plt.plot(viewtrend3['date_block_num'], viewtrend3['item_cnt_day'])

plt.xlabel('date_block_num', fontsize=10)

plt.ylabel('item_cnt_day', fontsize=10)

plt.xticks(viewtrend3['date_block_num'], viewtrend3['date_block_num'], fontsize=9, rotation=30)

plt.show()
result = rawItems.groupby(by = ['item_category_id']).agg({'item_id':'count'})

result = result.reset_index()

result



from matplotlib.pyplot import figure

figure(num=None, figsize=(14, 5), dpi=80, facecolor='w', edgecolor='k')

plt.plot(result['item_category_id'], result['item_id'])

plt.xlabel('date_block_num', fontsize=10)

plt.ylabel('item_cnt_day', fontsize=10)

plt.xticks(result['item_category_id'], result['item_category_id'], fontsize=9, rotation=30)

plt.show()
plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x=rawSales['item_cnt_day'])

print('Sale volume outliers:',rawSales['item_id'][rawSales['item_cnt_day']>500].unique())



plt.figure(figsize=(10,4))

plt.xlim(rawSales['item_price'].min(), rawSales['item_price'].max())

sns.boxplot(x=rawSales['item_price'])

print('Item price outliers:',rawSales['item_id'][rawSales['item_price']>50000].unique())

plt.show()


rawSales = rawSales[(rawSales.item_price<50000) | (rawSales.date_block_num>=33)]#

rawSales = rawSales[(rawSales.item_cnt_day<501) | (rawSales.date_block_num>=33)]#



median = rawSales[(rawSales.shop_id==32)&(rawSales.item_id==2973)&(rawSales.date_block_num==4)&(rawSales.item_price>0)].item_price.median()

rawSales.loc[rawSales.item_price<0, 'item_price'] = median
#aggregate on month-year, item id, and shop id



aggSales = rawSales.groupby(['date_block_num', 'item_id', 'shop_id'], as_index=False).agg({"item_cnt_day": np.sum, "item_price": np.mean})

aggSales = aggSales.rename(index=str, columns={"item_cnt_day": "item_cnt_month_year", "item_price": "item_price_avg"})
print(aggSales.shape, min(aggSales['item_cnt_month_year']), max(aggSales['item_cnt_month_year']))
#To ensure there is data for every item and every shop for every month in time window

print(aggSales.shape)



completeSales = []

for aBlock in tqdm(rawSales['date_block_num'].unique()):

    theShops = rawSales[rawSales['date_block_num']==aBlock]['shop_id'].unique()

    theItems = rawSales[rawSales['date_block_num']==aBlock]['item_id'].unique()

    completeSales.append(np.array(list(product(*[theShops, theItems, [aBlock]])),dtype='int32'))

    

completeSales = pd.DataFrame(np.vstack(completeSales), columns = ['shop_id', 'item_id', 'date_block_num'],dtype=np.int32)



aggSales = pd.merge(completeSales,aggSales,how='left',on=['shop_id', 'item_id', 'date_block_num'])

aggSales.item_cnt_month_year = aggSales.item_cnt_month_year.fillna(0)





print(completeSales.shape)

print(aggSales.shape)
#testSales = pd.read_csv("test.csv", usecols = lambda column : column not in ["ID"])

testSales = pd.read_csv("../input/pfsextracted/test.csv", usecols = lambda column : column not in ["ID"])

#add category id

testSales['date_block_num'] = 34

testSales['item_cnt_month_year'] = 0



aggSales = pd.concat([aggSales, testSales], axis = 0)
#change field types to conserve memory usage



aggSales['date_block_num'] = aggSales['date_block_num'].astype(np.int8)

aggSales['shop_id'] = aggSales['shop_id'].astype(np.int8)

aggSales['item_id'] = aggSales['item_id'].astype(np.int16)

aggSales['item_cnt_month_year'] = aggSales['item_cnt_month_year'].astype(np.float16)

aggSales
#Add item category

#Using item category id is already label encoded



def get_item_category(salesdata, itemdata):

    salesdata = salesdata.merge(itemdata[['item_id', 'item_category_id']], on = ['item_id'], how = 'left')

    salesdata['item_category_id'] = salesdata['item_category_id'].astype(np.int8)

    return salesdata
#item mean sales prev block

#Take avg of sales made in previous x months of same item across all stores



def item_mean_sales(salesdata):

    groups = salesdata.groupby(by = ['item_id', 'date_block_num'])

    item_sales_cols = []



    #Lag of 12 months in case of year to year seasonality or shorter

    #from feature importance it is seen that lag months 1,2,3,6,12 are used most

    lagmonths = [1,2,3,6,12]

    for diff in tqdm(lagmonths):

        feature_name = 'item_sales_lag_' + str(diff)

        item_sales_cols += [feature_name]

        result = groups.agg({'item_cnt_month_year':'mean'})

        result = result.reset_index()

        result.loc[:, 'date_block_num'] += diff

        result.rename(columns={'item_cnt_month_year': feature_name}, inplace=True)

        salesdata = salesdata.merge(result, on = ['item_id', 'date_block_num'], how = 'left')

        salesdata[feature_name] = salesdata[feature_name].fillna(0)

        salesdata[feature_name] = salesdata[feature_name].astype(np.float16)

        

        

    return salesdata, item_sales_cols

#item sales from store for prev block

#Take sales made in previous x months of same item across specific stores



def item_store_sales(salesdata):

    #groups = salesdata.groupby(by = ['item_id', 'shop_id', 'date_block_num'])

    item_store_sales_cols = []



    #Lag of 12 months in case of year to year seasonality or shorter

    #from feature importance it is seen that lag months 1,2,3,6,12 are used most

    lagmonths = [1,2,3,6,12]

    for diff in tqdm(lagmonths):

        feature_name = 'item_store_sales_lag_' + str(diff)

        item_store_sales_cols += [feature_name]



        result = salesdata[['item_id', 'shop_id', 'date_block_num', 'item_cnt_month_year']]

        result.loc[:, 'date_block_num'] += diff

        result.rename(columns={'item_cnt_month_year': feature_name}, inplace=True)

        salesdata = salesdata.merge(result, on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')

        salesdata[feature_name] = salesdata[feature_name].fillna(0)

        salesdata[feature_name] = salesdata[feature_name].astype(np.float16)

        

    return salesdata, item_store_sales_cols

#Item sales for every category for prev block

#Take avg of sales made in previous x months of every category



def item_cat_mean_sales(salesdata):

    groups = salesdata.groupby(by = ['item_category_id', 'date_block_num'])

    item_cat_sales_cols = []



    #Lag of 12 months in case of year to year seasonality or shorter

    #from feature importance it is seen that lag months 1,2,3,6,12 are used most

    lagmonths = [1,2,3,6,12]

    for diff in tqdm(lagmonths):

        feature_name = 'item_cat_sales_lag_' + str(diff)

        item_cat_sales_cols += [feature_name]

        result = groups.agg({'item_cnt_month_year':'mean'})

        result = result.reset_index()

        result.loc[:, 'date_block_num'] += diff

        result.rename(columns={'item_cnt_month_year': feature_name}, inplace=True)

        salesdata = salesdata.merge(result, on = ['item_category_id', 'date_block_num'], how = 'left')

        salesdata[feature_name] = salesdata[feature_name].fillna(0)

        salesdata[feature_name] = salesdata[feature_name].astype(np.float16)

        

        

    return salesdata, item_cat_sales_cols
#item mean sales prev block

#Take avg of sales made in previous x months of same item across all stores



def item_store_cat_mean_sales(salesdata):

    groups = salesdata.groupby(by = ['item_category_id', 'shop_id', 'date_block_num'])

    item_store_cat_sales_cols = []



    #Lag of 12 months in case of year to year seasonality or shorter

    #from feature importance it is seen that lag months 1,2,3,6,12 are used most

    lagmonths = [1,2,3,6,12]

    for diff in tqdm(lagmonths):

        feature_name = 'item_store_cat_sales_lag_' + str(diff)

        item_store_cat_sales_cols += [feature_name]

        result = groups.agg({'item_cnt_month_year':'mean'})

        result = result.reset_index()

        result.loc[:, 'date_block_num'] += diff

        result.rename(columns={'item_cnt_month_year': feature_name}, inplace=True)

        salesdata = salesdata.merge(result, on = ['item_category_id', 'shop_id', 'date_block_num'], how = 'left')

        salesdata[feature_name] = salesdata[feature_name].fillna(0)

        salesdata[feature_name] = salesdata[feature_name].astype(np.float16)

        

        

    return salesdata, item_store_cat_sales_cols
#Cumulative sum

#cumulative amount of sales of each item from each store for all block provided for training



def sales_cumulative_sum(salesdata):

    

    salesdata['item_cnt_cum'] = salesdata.groupby(['item_id', 'shop_id'])['item_cnt_month_year'].cumsum()



    #Since it should be a lag feature we shift all values down 1 row

    salesdata['item_cnt_cum'] = salesdata.groupby(by = ['item_id', 'shop_id'])['item_cnt_cum'].shift(1)

    salesdata['item_cnt_cum'] = salesdata['item_cnt_cum'].fillna(0)

    salesdata['item_cnt_cum'] = salesdata['item_cnt_cum'].astype(np.float16)

    

    return salesdata
#diff sales from prev block item-shop basis



def sales_difference(salesdata):

    

    result = salesdata.loc[:,['item_id', 'shop_id', 'date_block_num', 'item_cnt_month_year']]

    result.loc[:, 'date_block_num'] += 1

    result.rename(columns={'item_cnt_month_year': 'item_cnt_month_yearprev'}, inplace=True)

    salesdata = salesdata.merge(result, on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')

    salesdata['item_cnt_month_yearprev'] = salesdata['item_cnt_month_yearprev'].fillna(0)

    salesdata['item_cnt_dff_1'] = salesdata['item_cnt_month_year'] - salesdata['item_cnt_month_yearprev']

    salesdata = salesdata.drop(columns=['item_cnt_month_yearprev'])



    #Since it should be a lag feature we shift all values down 1 row

    salesdata['item_cnt_dff_1'] = salesdata.groupby(by = ['item_id', 'shop_id'])['item_cnt_dff_1'].shift(1)

    salesdata['item_cnt_dff_1'] = salesdata['item_cnt_dff_1'].fillna(0)

    salesdata['item_cnt_dff_1'] = salesdata['item_cnt_dff_1'].astype(np.float16)

    

    return salesdata

#item mean price prev block

#Take avg of item prices in previous 12 months of same item across all stores



def item_mean_price(salesdata):

    

    groups = salesdata.groupby(by = ['item_id', 'date_block_num'])

    

    #To keep track of feature names

    item_price_cols = []



    #Lag of 12 months in case of year to year seasonality or shorter

    #from feature importance it is seen that lag months 1,2,3,6,12 are used most

    lagmonths = [1,2,3,6,12]

    for diff in tqdm(lagmonths):

        feature_name = 'item_price_lag_' + str(diff)

        item_price_cols += [feature_name]

        result = groups.agg({'item_price_avg':'mean'})

        result = result.reset_index()

        result.loc[:, 'date_block_num'] += diff

        result.rename(columns={'item_price_avg': feature_name}, inplace=True)

        salesdata = salesdata.merge(result, on = ['item_id', 'date_block_num'], how = 'left')

        salesdata[feature_name] = salesdata[feature_name].fillna(0)

        salesdata[feature_name] = salesdata[feature_name].astype(np.float16)

        

    return salesdata, item_price_cols
#item price from store for prev block

#Take prices of product from previous 12 months of same item across specific stores



def item_store_price(salesdata):



    groups = salesdata.groupby(by = ['item_id', 'shop_id', 'date_block_num'])

    item_store_price_cols = []



    #Lag of 12 months in case of year to year seasonality or shorter

    #from feature importance it is seen that lag months 1,2,3,6,12 are used most

    lagmonths = [1,2,3,6,12]

    for diff in tqdm(lagmonths):

        feature_name = 'item_store_price_lag_' + str(diff)

        item_store_price_cols += [feature_name]

        result = aggSales[['item_id', 'shop_id', 'date_block_num', 'item_price_avg']]

        result.loc[:, 'date_block_num'] += diff

        result.rename(columns={'item_price_avg': feature_name}, inplace=True)

        salesdata = salesdata.merge(result, on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')

        salesdata[feature_name] = salesdata[feature_name].fillna(0)

        salesdata[feature_name] = salesdata[feature_name].astype(np.float16)

        

    return salesdata, item_store_price_cols

    
#Calculate delta (price trend) by using forlmula: ((item price avg for month) - (item price avg))/ (item price avg)



def item_price_delta(salesdata):

    tempcols = []

    

    #Get avg price for all items

    groups = salesdata.groupby(by = ['item_id'])

    result = groups.agg({'item_price_avg':'mean'})

    result = result.reset_index()

    result.rename(columns={'item_price_avg': 'items_avg_price'}, inplace=True)

    salesdata = salesdata.merge(result, on = ['item_id'], how = 'left')

    tempcols += ['items_avg_price']

    

    #get item price based on month

    groups = salesdata.groupby(by = ['item_id', 'date_block_num'])

    lagmonths = [1,2,3,4,5,6]

    for diff in tqdm(lagmonths):

        feature_name = 'items_avg_price_month_' + str(diff)

        tempcols += [feature_name]

        result = groups.agg({'item_price_avg':'mean'})

        result = result.reset_index()

        result.loc[:, 'date_block_num'] += diff

        result.rename(columns={'item_price_avg': feature_name}, inplace=True)

        salesdata = salesdata.merge(result, on = ['item_id', 'date_block_num'], how = 'left')

    

    #calculate delta for past 6 months

    print('Getting item price delta lag -> past 6 months delta')

    for l in lagmonths:

        salesdata['item_price_delta_lag_'+str(l)] = (salesdata['items_avg_price_month_'+str(l)] - salesdata['items_avg_price'])/ salesdata['items_avg_price']

        tempcols += ['item_price_delta_lag_'+str(l)]

        

    def get_trend(row):

        for l in lagmonths:

            if row['item_price_delta_lag_'+str(l)]:

                return row['item_price_delta_lag_'+str(l)]

        return 0



    print('Getting item price delta lag -> get first valid delta')

    salesdata['price_delta_trend'] = salesdata.apply(get_trend, axis=1)

    salesdata['price_delta_trend'] = salesdata['price_delta_trend'].fillna(0)

    salesdata['price_delta_trend'] = salesdata['price_delta_trend'].astype(np.float16)

    salesdata.drop(tempcols, axis=1, inplace=True)

    return salesdata

    
def fill_prices(salesdata):



    salesdata, item_store_price_cols = item_store_price(salesdata)



    result = salesdata['item_price_avg'].values



    for index, row in salesdata.iterrows():

        if math.isnan(row['item_price_avg']):

            for aFeat in item_store_price_cols[::-1]:

                if row[aFeat] > 0:

                    result[index] = row[aFeat]



    salesdata['item_price_avg'] = result

    salesdata['item_price_avg'] = salesdata['item_price_avg'].fillna(0)



    salesdata = salesdata.drop(columns= item_store_price_cols)

    

    #Save to be reused later

    salesdata[['shop_id', 'item_id', 'date_block_num', 'item_cnt_month_year', 'item_price_avg']].to_csv("sales_filled_prices.csv")

    return salesdata



#aggSales = fill_prices(aggSales)
#clip

aggSales['item_cnt_month_year'] = aggSales['item_cnt_month_year'].clip(0,20)

print(aggSales.shape, min(aggSales['item_cnt_month_year']), max(aggSales['item_cnt_month_year']))


print('Getting item categories')

aggSales = get_item_category(aggSales, rawItems)



print('Getting item sales lag')

aggSales, item_sales_cols = item_mean_sales(aggSales)



print('Getting item sales lag for each store')

aggSales, item_store_sales_cols = item_store_sales(aggSales)



print('Getting item sales lag for each category')

aggSales, item_cat_sales_cols = item_cat_mean_sales(aggSales)



print('Getting item sales lag for each category and store')

aggSales, item_store_cat_sales_cols = item_store_cat_mean_sales(aggSales)



print('Calculating difference of item sales from last block for each store')

aggSales = sales_difference(aggSales)



#print('Getting item price lag')

#aggSales, item_price_cols = item_mean_price(aggSales)



#print('Getting item price lag for each store')

#aggSales, item_store_price_cols = item_store_price(aggSales)



print('Getting item price delta lag')

aggSales= item_price_delta(aggSales)

aggSales.drop(['item_price_avg'], axis=1, inplace=True)



print('Calculating item first sale for each shop')

aggSales['item_shop_first_sale'] = aggSales['date_block_num'] - aggSales.groupby(['item_id','shop_id'])['date_block_num'].transform('min')

aggSales['item_shop_first_sale'] = aggSales['item_shop_first_sale'].astype(np.int8)



print('Calculating item first sale')

aggSales['item_first_sale'] = aggSales['date_block_num'] - aggSales.groupby('item_id')['date_block_num'].transform('min')

aggSales['item_first_sale'] = aggSales['item_first_sale'].astype(np.int8)



#

#Maybe drop first 12 months with little lag data

#

aggSales = aggSales[aggSales['date_block_num'] >= 12]



print('Calculating cumulative sum for items in each store')

aggSales = sales_cumulative_sum(aggSales)



print('Get the month number for each date bloack number')

aggSales['month'] = aggSales['date_block_num'] % 12
aggSales.iloc[:,30:50]



aggSales[(aggSales['shop_id'] == 25 ) & (aggSales['date_block_num'] < 14)][['date_block_num', 'item_category_id', 'item_cat_sales_lag_1', 'item_cat_sales_lag_2', 'item_cat_sales_lag_3', 'item_store_cat_sales_lag_1', 'item_store_cat_sales_lag_2']] 
#Free up some memory

del rawSales

del completeSales

del rawItems

del theShops

del theItems

del result

#del testSales

print('Deleted vars')
aggSales[(aggSales['item_id'] == 22167) & (aggSales['shop_id'] == 25 )]

aggSales.info()
#Check if there are any columns with null values still (if nothing is printed then there are no null values)



for cooll in aggSales.columns:

    if (aggSales[cooll].isnull().values.any() == True):

        print(cooll)
#store the calculated features



#aggSales.to_csv("sales_train_featjul4.csv")
#aggSales = pd.read_csv("dataset_train_feat.csv", index_col=0)
aggSales.head()
X_train = aggSales[aggSales['date_block_num'] < 33].drop(['item_cnt_month_year'], axis=1).copy()

Y_train = aggSales[aggSales['date_block_num'] < 33]['item_cnt_month_year'].copy()

X_valid = aggSales[aggSales['date_block_num'] == 33].drop(['item_cnt_month_year'], axis=1).copy()

Y_valid = aggSales[aggSales['date_block_num'] == 33]['item_cnt_month_year'].copy()

X_test = aggSales[aggSales['date_block_num'] == 34].drop(['item_cnt_month_year'], axis=1).copy()
X_train.info()
del aggSales

del item_sales_cols

del item_store_sales_cols

#del item_price_cols

#del item_store_price_cols
model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 10)
#save models

pickle.dump(model, open('pred_xgbreg_feat_changes_salesF10.sav', 'wb'))
fig, ax = plt.subplots(1,1,figsize=(10,15))

plot_importance(booster=model, ax=ax)

plt.show()
Y_pred = model.predict(X_valid).clip(0, 20)

Y_test = model.predict(X_test).clip(0, 20)



submission = pd.DataFrame({"ID": testSales.index, "item_cnt_month": Y_test})

submission.to_csv('sd_submission_feat_changes_salesF10.csv', index=False)