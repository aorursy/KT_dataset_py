import numpy as np 

import pandas as pd 

import catboost

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import tqdm

import re

from sklearn.preprocessing import LabelEncoder

from itertools import product

import time

import os

import gc

import pickle

from xgboost import XGBRegressor

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

pd.set_option('display.max_columns',100)

import os



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:#tqdm(df.columns):

        col_type = df[col].dtypes



        if col_type=='object':

            df[col] = df[col].astype('category')



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



def display_status(message):

    from IPython.display import display, clear_output

    import time

    display(message) # print string

    clear_output(wait=True)

    

def unify_duplicated_shop_id(ids):    

    global train,test

    for pair in ids:

        (origin, replacement) = pair

        display_status('replace {0} with {1}'.format(origin, replacement))

        train.loc[train.shop_id == origin, "shop_id"] = replacement

        test.loc[test.shop_id == origin , "shop_id"] = replacement



def clean_and_expand_shop_data():

    global shops

    display_status('Cleaning Shop Data')

    shops.loc[ shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"'

    shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] )

    shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] )

    shops.loc[shops.city == "!Якутск", "city"] = "Якутск"



    

def name_correction(x):

    x = x.lower()

    x = x.partition('[')[0]

    x = x.partition('(')[0]

    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)

    x = x.replace('  ', ' ')

    x = x.strip()

    return x



def clean_text_and_recategorize():

    global shops,item_categories,items

    category = []

    for cat in shops.category.unique():

        print(cat, len(shops[shops.category == cat]) )

        if len(shops[shops.category == cat]) > 4:

            category.append(cat)



    shops.category = shops.category.apply( lambda x: x if (x in category) else "etc" )



    

    shops["shop_category"] = LabelEncoder().fit_transform( shops.category )

    shops["shop_city"] = LabelEncoder().fit_transform( shops.city )

    shops = shops[["shop_id", "shop_category", "shop_city"]]



    item_categories["type_code"] = item_categories.item_category_name.apply( lambda x: x.split(" ")[0] ).astype(str)

    item_categories.loc[ (item_categories.type_code == "Игровые")| (item_categories.type_code == "Аксессуары"), "category" ] = "Игры"



    category = []

    for cat in item_categories.type_code.unique():

        print(cat, len(item_categories[item_categories.type_code == cat]))

        if len(item_categories[item_categories.type_code == cat]) > 4: 

            category.append( cat )



    item_categories.type_code = item_categories.type_code.apply(lambda x: x if (x in category) else "etc")



    for cat in item_categories.type_code.unique():

        print(cat, len(item_categories[item_categories.type_code == cat]))



    item_categories.type_code = LabelEncoder().fit_transform(item_categories.type_code)

    item_categories["split"] = item_categories.item_category_name.apply(lambda x: x.split("-"))

    item_categories["subtype"] = item_categories.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

    item_categories["subtype_code"] = LabelEncoder().fit_transform( item_categories["subtype"] )

    item_categories = item_categories[["item_category_id", "subtype_code", "type_code"]]



    items["name1"], items["name2"] = items.item_name.str.split("[", 1).str

    items["name1"], items["name3"] = items.item_name.str.split("(", 1).str



    items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()

    items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()

    items = items.fillna('0')



    items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))

    items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0")



    items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )

    items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"

    items.loc[ items.type == "", "type"] = "mac"

    items.type = items.type.apply( lambda x: x.replace(" ", "") )

    items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"

    items.loc[ items.type == 'рs3' , "type"] = "ps3"

    #Add popular titles as separate type

    populars = ['alien', 'adele', 'angry', 'anno','edition', 'ultimate', 'gold', 'premium', 'assassin', 'battlefield', 

            'call of duty', 'fifa', 'football','need for speed', 'pokemon',

            'grand theft auto', 'fallout', 'star wars', 'uncharted','Kaspersky', 'Sims', 'medal of honor', 'Фирменный пакет майка',

            'mad max', 'metal gear', 'ea', 'diablo', 'warcraft', 'Доставка','Ведьмак 3'

           ]

    index = [False] * len(train)

    display_status('tagging popular items such as Star Wars')

    for word in tqdm(populars):

        idx_popular_item_sales = items['item_name'].str.contains(word, case=False)

        items.loc[ idx_popular_item_sales , "type"] = word





    group_sum = items.groupby(["type"]).agg({"item_id": "count"})

    print( group_sum.reset_index() )

    group_sum = group_sum.reset_index()





    drop_cols = []

    for cat in group_sum.type.unique():

    #     print(group_sum.loc[(group_sum.type == cat), "item_id"].values[0])

        if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:

            drop_cols.append(cat)



    items.name2 = items.name2.apply( lambda x: "etc" if (x in drop_cols) else x )

    items = items.drop(["type"], axis = 1)





    items.name2 = LabelEncoder().fit_transform(items.name2)

    items.name3 = LabelEncoder().fit_transform(items.name3)



    items.drop(["item_name", "name1"],axis = 1, inplace= True)

    items.head()

    



def lag_feature( df,lags, cols ):

    for col in cols:

        print(col)

        tmp = df[["date_block_num", "shop_id","item_id",col ]]

        for i in lags:

            shifted = tmp.copy()

            shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_"+str(i)]

            shifted.date_block_num = shifted.date_block_num + i

            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

    return df

    







for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

basepath= '../input/competitive-data-science-predict-future-sales/'



items = pd.read_csv(basepath+"items.csv")

item_categories = pd.read_csv(basepath+"item_categories.csv")

shops = pd.read_csv(basepath+"shops.csv")

train = pd.read_csv( basepath+"sales_train.csv" )

#train = train.sample(20000) #for quick Proof of Concept, sample 20000

test = pd.read_csv( basepath+"test.csv" )



unify_duplicated_shop_id(ids=[(0,57),(1,58),(11,10),(40,39)])

clean_and_expand_shop_data()

clean_text_and_recategorize()



#Now we will group the data by these 5 columns and some of their combination: "date_block_num", "shop_id","item_id",'subtype_code','shop_city'

ts = time.time()

agg_data = []

cols  = ["date_block_num", "shop_id", "item_id"]

display_status('========| Create aggregated data(1/8) |========= ~Time elapsed: {0} secs'.format(str(time.time()- ts)) )

for i in range(34):

    sales = train[train.date_block_num == i]

    agg_data.append( np.array(list( product( [i], sales.shop_id.unique(), sales.item_id.unique() ) ), dtype = np.int16) )



agg_data = pd.DataFrame( np.vstack(agg_data), columns = cols )

agg_data["date_block_num"] = agg_data["date_block_num"].astype(np.int8)

agg_data["shop_id"] = agg_data["shop_id"].astype(np.int8)

agg_data["item_id"] = agg_data["item_id"].astype(np.int16)

agg_data.sort_values( cols, inplace = True )





train["revenue"] = train["item_cnt_day"] * train["item_price"]



display_status('========| Group Aggregate based on date, shop, item(2/8) |========= ~Time elapsed: {0} secs'.format(str(time.time()- ts)) )

group = train.groupby( ["date_block_num", "shop_id", "item_id"] ).agg( {"item_cnt_day": ["sum"]} )

group.columns = ["item_cnt_month"]

group.reset_index( inplace = True)

agg_data = pd.merge( agg_data, group, on = cols, how = "left" )

agg_data["item_cnt_month"] = agg_data["item_cnt_month"].fillna(0).clip(0,20).astype(np.float16)





test["date_block_num"] = 34

test["date_block_num"] = test["date_block_num"].astype(np.int8)

test["shop_id"] = test.shop_id.astype(np.int8)

test["item_id"] = test.item_id.astype(np.int16)





agg_data = pd.concat([agg_data, test.drop(["ID"],axis = 1)], ignore_index=True, sort=False, keys=cols)

agg_data.fillna( 0, inplace = True )

agg_data





display_status('========| Merge shops, items, categoriess (3/8) |========= ~Time elapsed: {0} secs'.format(str(time.time()- ts)) )

agg_data = pd.merge( agg_data, shops, on = ["shop_id"], how = "left" )

agg_data = pd.merge(agg_data, items, on = ["item_id"], how = "left")

agg_data = pd.merge( agg_data, item_categories, on = ["item_category_id"], how = "left" )

agg_data["shop_city"] = agg_data["shop_city"].astype(np.int8)

agg_data["shop_category"] = agg_data["shop_category"].astype(np.int8)

agg_data["item_category_id"] = agg_data["item_category_id"].astype(np.int8)

agg_data["subtype_code"] = agg_data["subtype_code"].astype(np.int8)

agg_data["name2"] = agg_data["name2"].astype(np.int8)

agg_data["name3"] = agg_data["name3"].astype(np.int16)

agg_data["type_code"] = agg_data["type_code"].astype(np.int8)

time.time() - ts







agg_data = lag_feature( agg_data, [1,2,3], ["item_cnt_month"] )





display_status('========| Group & merge based on date_block_num (4/8) |========= ~Time elapsed: {0} secs'.format(str(time.time()- ts)) )

group = agg_data.groupby( ["date_block_num"] ).agg({"item_cnt_month" : ["mean"]})

group.columns = ["date_avg_item_cnt"]

group.reset_index(inplace = True)



agg_data = pd.merge(agg_data, group, on = ["date_block_num"], how = "left")

agg_data.date_avg_item_cnt = agg_data["date_avg_item_cnt"].astype(np.float16)

agg_data = lag_feature( agg_data, [1], ["date_avg_item_cnt"] )

agg_data.drop( ["date_avg_item_cnt"], axis = 1, inplace = True )









display_status('========| Group & merge based on [date_block_num, item_id] (4/8) |========= ~Time elapsed: {0} secs'.format(str(time.time()- ts)) )

group = agg_data.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_item_avg_item_cnt' ]

group.reset_index(inplace=True)



agg_data = pd.merge(agg_data, group, on=['date_block_num','item_id'], how='left')

agg_data.date_item_avg_item_cnt = agg_data['date_item_avg_item_cnt'].astype(np.float16)

agg_data = lag_feature(agg_data, [1,2,3], ['date_item_avg_item_cnt'])

agg_data.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)







display_status('========| Group & merge based on [date_block_num , shop_id] (5/8) |========= ~Time elapsed: {0} secs'.format(str(time.time()- ts)) )

group = agg_data.groupby( ["date_block_num","shop_id"] ).agg({"item_cnt_month" : ["mean"]})

group.columns = ["date_shop_avg_item_cnt"]

group.reset_index(inplace = True)



agg_data = pd.merge(agg_data, group, on = ["date_block_num","shop_id"], how = "left")

agg_data.date_avg_item_cnt = agg_data["date_shop_avg_item_cnt"].astype(np.float16)

agg_data = lag_feature( agg_data, [1,2,3], ["date_shop_avg_item_cnt"] )

agg_data.drop( ["date_shop_avg_item_cnt"], axis = 1, inplace = True )







group = agg_data.groupby( ["date_block_num","shop_id","item_id"] ).agg({"item_cnt_month" : ["mean"]})

group.columns = ["date_shop_item_avg_item_cnt"]

group.reset_index(inplace = True)



agg_data = pd.merge(agg_data, group, on = ["date_block_num","shop_id","item_id"], how = "left")

agg_data.date_avg_item_cnt = agg_data["date_shop_item_avg_item_cnt"].astype(np.float16)

agg_data = lag_feature( agg_data, [1,2,3], ["date_shop_item_avg_item_cnt"] )

agg_data.drop( ["date_shop_item_avg_item_cnt"], axis = 1, inplace = True )









group = agg_data.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_shop_subtype_avg_item_cnt']

group.reset_index(inplace=True)



agg_data = pd.merge(agg_data, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')

agg_data.date_shop_subtype_avg_item_cnt = agg_data['date_shop_subtype_avg_item_cnt'].astype(np.float16)

agg_data = lag_feature(agg_data, [1], ['date_shop_subtype_avg_item_cnt'])

agg_data.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)







group = agg_data.groupby(['date_block_num', 'shop_city']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_city_avg_item_cnt']

group.reset_index(inplace=True)



agg_data = pd.merge(agg_data, group, on=['date_block_num', "shop_city"], how='left')

agg_data.date_city_avg_item_cnt = agg_data['date_city_avg_item_cnt'].astype(np.float16)

agg_data = lag_feature(agg_data, [1], ['date_city_avg_item_cnt'])

agg_data.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)





display_status('========| This will take long time (6/8) |========= ~Time elapsed: {0} secs'.format(str(time.time()- ts)) )

group = agg_data.groupby(['date_block_num', 'item_id', 'shop_city']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_item_city_avg_item_cnt' ]

group.reset_index(inplace=True)



agg_data = pd.merge(agg_data, group, on=['date_block_num', 'item_id', 'shop_city'], how='left')

agg_data.date_item_city_avg_item_cnt = agg_data['date_item_city_avg_item_cnt'].astype(np.float16)

agg_data = lag_feature(agg_data, [1], ['date_item_city_avg_item_cnt'])

agg_data.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)









group = train.groupby( ["item_id"] ).agg({"item_price": ["mean"]})

group.columns = ["item_avg_item_price"]

group.reset_index(inplace = True)



agg_data = agg_data.merge( group, on = ["item_id"], how = "left" )

agg_data["item_avg_item_price"] = agg_data.item_avg_item_price.astype(np.float16)





group = train.groupby( ["date_block_num","item_id"] ).agg( {"item_price": ["mean"]} )

group.columns = ["date_item_avg_item_price"]

group.reset_index(inplace = True)



agg_data = agg_data.merge(group, on = ["date_block_num","item_id"], how = "left")

agg_data["date_item_avg_item_price"] = agg_data.date_item_avg_item_price.astype(np.float16)

lags = [1, 2, 3]

agg_data = lag_feature( agg_data, lags, ["date_item_avg_item_price"] )

for i in lags:

    agg_data["delta_price_lag_" + str(i) ] = (agg_data["date_item_avg_item_price_lag_" + str(i)]- agg_data["item_avg_item_price"] )/ agg_data["item_avg_item_price"]



def select_trends(row) :

    for i in lags:

        if row["delta_price_lag_" + str(i)]:

            return row["delta_price_lag_" + str(i)]

    return 0



agg_data["delta_price_lag"] = agg_data.apply(select_trends, axis = 1)

agg_data["delta_price_lag"] = agg_data.delta_price_lag.astype( np.float16 )

agg_data["delta_price_lag"].fillna( 0 ,inplace = True)



features_to_drop = ["item_avg_item_price", "date_item_avg_item_price"]

for i in lags:

    features_to_drop.append("date_item_avg_item_price_lag_" + str(i) )

    features_to_drop.append("delta_price_lag_" + str(i) )

agg_data.drop(features_to_drop, axis = 1, inplace = True)







display_status('========| This will take long time too (7/8) |========= ~Time elapsed: {0} secs'.format(str(time.time()- ts)) )

group = train.groupby( ["date_block_num","shop_id"] ).agg({"revenue": ["sum"] })

group.columns = ["date_shop_revenue"]

group.reset_index(inplace = True)



agg_data = agg_data.merge( group , on = ["date_block_num", "shop_id"], how = "left" )

agg_data['date_shop_revenue'] = agg_data['date_shop_revenue'].astype(np.float32)



group = group.groupby(["shop_id"]).agg({ "date_block_num":["mean"] })

group.columns = ["shop_avg_revenue"]

group.reset_index(inplace = True )



agg_data = agg_data.merge( group, on = ["shop_id"], how = "left" )

agg_data["shop_avg_revenue"] = agg_data.shop_avg_revenue.astype(np.float32)

agg_data["delta_revenue"] = (agg_data['date_shop_revenue'] - agg_data['shop_avg_revenue']) / agg_data['shop_avg_revenue']

agg_data["delta_revenue"] = agg_data["delta_revenue"]. astype(np.float32)



agg_data = lag_feature(agg_data, [1], ["delta_revenue"])

agg_data["delta_revenue_lag_1"] = agg_data["delta_revenue_lag_1"].astype(np.float32)

agg_data.drop( ["date_shop_revenue", "shop_avg_revenue", "delta_revenue"] ,axis = 1, inplace = True)









display_status('========| Last (8/8) |========= ~Time elapsed: {0} secs'.format(str(time.time()- ts)) )

# Adding number of days for each month

agg_data["month"] = agg_data["date_block_num"] % 12

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

agg_data["days"] = agg_data["month"].map(days).astype(np.int8)





agg_data["item_shop_first_sale"] = agg_data["date_block_num"] - agg_data.groupby(["item_id","shop_id"])["date_block_num"].transform('min')

agg_data["item_first_sale"] = agg_data["date_block_num"] - agg_data.groupby(["item_id"])["date_block_num"].transform('min')





agg_data = agg_data[agg_data["date_block_num"] > 3]

display_status('========| Completed (8/8) |========= ~Time elapsed: {0} secs'.format(str(time.time()- ts)) )











#Adding approx number of holidays 2013-2015 in Russia for each month

            # ([Jan,Feb, ... ,            Dec, ])

days = pd.Series([10,3,2,1,7,2,  1,0,2,1,1,2])

agg_data["holiday_num"] = agg_data["month"].map(days).astype(np.int8)



#Approx number of Russia holidays next month 2013-2015 for each month

               # ([Feb, Mar, ... ,     Dec,Jan])

days = pd.Series([3,2,1,7,2,  1,0,2,1,1,2,10])

agg_data["next_month_holiday_num"] = agg_data["month"].map(days).astype(np.int8)





#Approx number of Russia holidays next month 2013-2015

#month ([Jan,  ... , Dec])

days = pd.Series(['winter','etc','spring','etc','may',  'summer','summer','summer','etc','winter','winter','winter'])

agg_data["vacation_period"] = agg_data["month"].map(days).astype('category')

#mean encoding

agg_data['vacation_period'] = agg_data.groupby('vacation_period')['item_cnt_month'].transform('mean')



agg_data.to_csv('agg_data.csv',index=False,header=True)

agg_data = reduce_mem_usage(agg_data)









X_train = agg_data[agg_data.date_block_num < 33].drop(['item_cnt_month'], axis=1)

y_train = agg_data[agg_data.date_block_num < 33]['item_cnt_month']

X_val = agg_data[agg_data.date_block_num == 33].drop(['item_cnt_month'], axis=1)

y_val = agg_data[agg_data.date_block_num == 33]['item_cnt_month']



X_test = agg_data[agg_data.date_block_num == 34].drop(['item_cnt_month'], axis=1)



del agg_data

import gc

gc.collect()



X_train.fillna(0,inplace=True)

X_val.fillna(0,inplace=True)

X_test.fillna(0,inplace=True)
est = 100

max_dept = 10



xgb_filename = 'xgb_pred_'+str(est)+'_'+str(max_dept)



xgb = XGBRegressor(

    max_depth=max_dept,

    n_estimators=est,

    min_child_weight=0.5, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.1,

#     tree_method='gpu_hist',

    seed=42)



xgb.fit(

    X_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, y_train), (X_val, y_val)], 

    verbose=True, 

    early_stopping_rounds = 20)



pickle.dump(xgb, open(xgb_filename+'.model', 'wb'))





y_test = xgb.predict(X_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": y_test

})

submission.to_csv(xgb_filename+'.csv', index=False)
est = 30

max_depth = 10





rf_pred_filename = 'rf_pred_'+str(est)+'_'+str(max_dept)



from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=est,max_depth=max_depth, random_state=42, oob_score=True, verbose=10, n_jobs=-1)

rf.fit(X_train, y_train)



pickle.dump(rf, open(rf_pred_filename+'.model', 'wb'))



y_test = rf.predict(X_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": y_test

})

submission.to_csv(rf_pred_filename+'.csv', index=False)
from sklearn.linear_model import LinearRegression

meta_train = pd.DataFrame({'ID':X_train.index,'pred_1':xgb.predict(X_train).clip(0, 20),'pred_2':rf.predict(X_train).clip(0, 20)})

y_meta_train = y_train

meta_model = LinearRegression()

meta_model.fit(meta_train[['pred_1','pred_2']],y_meta_train)







#Eval Meta Model on val data

meta_val = pd.DataFrame({'ID':X_val.index,'pred_1':xgb.predict(X_val).clip(0, 20),'pred_2':rf.predict(X_val).clip(0, 20)})

X_meta_val, y_meta_val = meta_val[['pred_1','pred_2']],y_val

print(meta_model.score(X_meta_val,y_meta_val))





filename = 'metamodel.model'

pickle.dump(meta_model, open(filename, 'wb'))





#Finally predict!

try:

    if xgb_filename and rf_filename:

        pass

except Exception:

    xgb_filename = 'xgb_pred_100_10'

    rf_filename = 'rf_pred_30_10'

    

pred_1 = pd.read_csv(xgb_filename+'.csv')

pred_2 = pd.read_csv(rf_filename+'.csv')

meta_test = pd.DataFrame({'ID':pred_1.ID,'pred_1':pred_1.item_cnt_month,'pred_2':pred_2.item_cnt_month})

X_meta_test = meta_test[['pred_1','pred_2']]







basepath= '../input/competitive-data-science-predict-future-sales/'

test = pd.read_csv( basepath+"test.csv" )



y_test = meta_model.predict(X_meta_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": y_test

})

submission.to_csv('meta_model_pred.csv', index=False)

submission.to_csv('submission.csv', index=False)

print('final prediction submission.csv file has been generated.')

submission