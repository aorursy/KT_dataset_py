from tqdm import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statistics

orders={}

products={}

users={}

products_aisles={}

products_departments={}

orders_isnone={}

orders_dows={}

orders_hours={}

orders_days_since_prior_order={}

train_days_since_prior_order={}

print("Start loading products.csv")

products_df=pd.read_csv('../input/products.csv', encoding = 'UTF-8')

for index, row in tqdm(products_df.iterrows()):

    product_id=str(int(row['product_id']))

    aisles_id=str(int(row['aisle_id']))

    departments_id=str(int(row['department_id']))

    products_aisles[product_id]=aisles_id

    products_departments[product_id]=departments_id

    



print("Start loading order_products__prior.csv")

#read prior orders

fr = open("../input/order_products__prior.csv", 'r', encoding = 'UTF-8')

fr.readline()# skip header

lines=fr.readlines()

for i,line in tqdm(enumerate(lines)):

    datas=line.replace("\n","").split(",")

    order_id=(datas[0])

    product_id=(datas[1])

    reorderer=int(datas[3])

    if(order_id not in orders):

        orders[order_id]={}

    orders[order_id][product_id]=reorderer



print("Start loading order_products__train.csv")

#read train orders

fr = open("../input/order_products__train.csv", 'r', encoding = 'UTF-8')

fr.readline()# skip header

lines=fr.readlines()

for i,line in tqdm(enumerate(lines)):

    datas=line.replace("\n","").split(",")

    order_id=(datas[0])

    product_id=(datas[1])

    reorderer=int(datas[3])

    if(order_id not in orders):

        orders[order_id]={}

    orders[order_id][product_id]=reorderer



for order_id in orders:

    reorder_list=list(orders[order_id].values())

    if(sum(reorder_list)==0):

        orders_isnone[order_id]=1

    else:

        orders_isnone[order_id]=0

    

print("Start loading orders.csv")

#read orders

fr = open("../input/orders.csv", 'r', encoding = 'UTF-8')



fr.readline()# skip header

lines=fr.readlines()

for i,line in tqdm(enumerate(lines)):

    datas=line.replace("\n","").split(",")

    order_id=(datas[0])

    user_id=(datas[1])

    eval_set=(datas[2])

    order_number=(datas[3])

    order_dow=int(datas[4])

    order_hours=int(datas[5])

    orders_dows[order_id]=order_dow

    orders_hours[order_id]=order_hours

    if(user_id not in users):

        users[user_id]={}

    if(eval_set=="prior"):

        try:

            days_since_prior_order=int(float(datas[6]))

            if(user_id not in orders_days_since_prior_order):

                orders_days_since_prior_order[user_id]=[]

            orders_days_since_prior_order[user_id].append(days_since_prior_order)

        except:

            pass

        users[user_id][order_number]=order_id

    elif(eval_set=="train"):

        users[user_id]["train"]=order_id

        days_since_prior_order=int(float(datas[6]))

        train_days_since_prior_order[order_id]=days_since_prior_order

    elif(eval_set=="test"):

        users[user_id]["test"]=order_id

    elif(eval_set=="valid"):

        users[user_id]["valid"]=order_id





print("Start creating features")



user_buytime_mean={}

user_buydow_mean={}

user_products={}

user_departments={}

user_aisles={}

user_none_order_rate={}

user_total_products={}

user_order_len={}

user_overall_reorder={}



for user_id in tqdm(users):

    if('train' not in list(users[user_id].keys())):

        continue

    user_buytime_mean[user_id]=[]

    user_buydow_mean[user_id]=[]

    user_products[user_id]=[]

    user_departments[user_id]=[]

    user_aisles[user_id]=[]

    user_none_order_rate[user_id]=[]

    user_order_len[user_id]=[]

    user_total_products[user_id]=0

    user_overall_reorder[user_id]=[]

    for i,(order_number, order_id) in enumerate(users[user_id].items()):

        if(order_number in ["test","train","valid"]):

            continue

        user_buytime_mean[user_id].append(orders_hours[order_id])

        user_buydow_mean[user_id].append(orders_dows[order_id])

        if(int(order_number)!=1):

            user_none_order_rate[user_id].append(orders_isnone[order_id])

        user_order_len[user_id].append(len(orders[order_id]))

        for product_id in orders[order_id]:

            user_total_products[user_id]+=1

            if(product_id not in user_products[user_id]):

                user_products[user_id].append(product_id)

            

            aisles_id=products_aisles[product_id]

            departments_id=products_departments[product_id]

            

            if(departments_id not in user_departments[user_id]):

                user_departments[user_id].append(departments_id)

            if(aisles_id not in user_aisles[user_id]):

                user_aisles[user_id].append(aisles_id)

            user_overall_reorder[user_id].append(orders[order_id][product_id])



outcsv = open("none_train_datas.csv", 'w', encoding = 'UTF-8')





cols=[]

cols.append("user_id")

cols.append("order_id")



cols.append("total_product")

cols.append("total_order")

cols.append("total_distinct_product")

cols.append("totoal_dep")

cols.append("totoal_aisle")



cols.append("order_dow")

cols.append("order_hour")

cols.append("days_since_prior_order")



cols.append("overall_reorder_rate")

cols.append("none_order_rate")



cols.append("mean_order_dow")

cols.append("mean_order_hour")

cols.append("mean_days_since_prior_order")

cols.append("mean_basket_size")



cols.append("is_none")



outcsv.writelines(','.join(cols)+"\n")

outcsv.flush()



print("Start saving features to csv")

for user_id in tqdm(users):

    if('train' not in list(users[user_id].keys())):

        continue

    

    order_id=users[user_id]["train"]

    

    features=[]

    features.append(user_id)

    features.append(order_id)

    

    features.append(user_total_products[user_id])

    features.append(len(users[user_id])-1)

    features.append(len(user_products[user_id]))

    features.append(len(user_departments[user_id]))

    features.append(len(user_aisles[user_id]))

    

    features.append(orders_dows[order_id])

    features.append(orders_hours[order_id])

    features.append(train_days_since_prior_order[order_id])

    

    overall_reorder_rate=sum(user_overall_reorder[user_id])/len(user_overall_reorder[user_id])

    none_order_rate=sum(user_none_order_rate[user_id])/len(user_none_order_rate[user_id])

    mean_order_dow=sum(user_buydow_mean[user_id])/len(user_buydow_mean[user_id])

    mean_order_hour=sum(user_buytime_mean[user_id])/len(user_buytime_mean[user_id])

    mean_days_since_prior_order=sum(orders_days_since_prior_order[user_id])/len(orders_days_since_prior_order[user_id])

    mean_basket_size=sum(user_order_len[user_id])/len(user_order_len[user_id])

    is_none=orders_isnone[order_id]

    

    features.append(overall_reorder_rate)

    features.append(none_order_rate)

    features.append(mean_order_dow)

    features.append(mean_order_hour)

    features.append(mean_days_since_prior_order)

    features.append(mean_basket_size)

    features.append(is_none)

    

    features_str=[]

    for feature in features:

        if(isinstance(feature, float)):

            features_str.append(str(round(feature,5)))

        else:

            features_str.append(str(feature))

            

    

    outcsv.writelines(','.join(features_str)+"\n")

    outcsv.flush()

        
from tqdm import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statistics

orders={}

products={}

users={}

products_aisles={}

products_departments={}

orders_isnone={}

orders_dows={}

orders_hours={}

orders_days_since_prior_order={}

train_days_since_prior_order={}

print("Start loading products.csv")

products_df=pd.read_csv('../input/products.csv', encoding = 'UTF-8')

for index, row in tqdm(products_df.iterrows()):

    product_id=str(int(row['product_id']))

    aisles_id=str(int(row['aisle_id']))

    departments_id=str(int(row['department_id']))

    products_aisles[product_id]=aisles_id

    products_departments[product_id]=departments_id

    



print("Start loading order_products__prior.csv")

#read prior orders

fr = open("../input/order_products__prior.csv", 'r', encoding = 'UTF-8')

fr.readline()# skip header

lines=fr.readlines()

for i,line in tqdm(enumerate(lines)):

    datas=line.replace("\n","").split(",")

    order_id=(datas[0])

    product_id=(datas[1])

    reorderer=int(datas[3])

    if(order_id not in orders):

        orders[order_id]={}

    orders[order_id][product_id]=reorderer



print("Start loading order_products__train.csv")

#read train orders

fr = open("../input/order_products__train.csv", 'r', encoding = 'UTF-8')

fr.readline()# skip header

lines=fr.readlines()

for i,line in tqdm(enumerate(lines)):

    datas=line.replace("\n","").split(",")

    order_id=(datas[0])

    product_id=(datas[1])

    reorderer=int(datas[3])

    if(order_id not in orders):

        orders[order_id]={}

    orders[order_id][product_id]=reorderer



for order_id in orders:

    reorder_list=list(orders[order_id].values())

    if(sum(reorder_list)==0):

        orders_isnone[order_id]=1

    else:

        orders_isnone[order_id]=0

    

print("Start loading orders.csv")

#read orders

fr = open("../input/orders.csv", 'r', encoding = 'UTF-8')



fr.readline()# skip header

lines=fr.readlines()

for i,line in tqdm(enumerate(lines)):

    datas=line.replace("\n","").split(",")

    order_id=(datas[0])

    user_id=(datas[1])

    eval_set=(datas[2])

    order_number=(datas[3])

    order_dow=int(datas[4])

    order_hours=int(datas[5])

    orders_dows[order_id]=order_dow

    orders_hours[order_id]=order_hours

    if(user_id not in users):

        users[user_id]={}

    if(eval_set=="prior"):

        try:

            days_since_prior_order=int(float(datas[6]))

            if(user_id not in orders_days_since_prior_order):

                orders_days_since_prior_order[user_id]=[]

            orders_days_since_prior_order[user_id].append(days_since_prior_order)

        except:

            pass

        users[user_id][order_number]=order_id

    elif(eval_set=="train"):

        users[user_id]["train"]=order_id

    elif(eval_set=="test"):

        users[user_id]["test"]=order_id

        days_since_prior_order=int(float(datas[6]))

        train_days_since_prior_order[order_id]=days_since_prior_order

    elif(eval_set=="valid"):

        users[user_id]["valid"]=order_id





print("Start creating features")



user_buytime_mean={}

user_buydow_mean={}

user_products={}

user_departments={}

user_aisles={}

user_none_order_rate={}

user_total_products={}

user_order_len={}

user_overall_reorder={}



for user_id in tqdm(users):

    if('test' not in list(users[user_id].keys())):

        continue

    user_buytime_mean[user_id]=[]

    user_buydow_mean[user_id]=[]

    user_products[user_id]=[]

    user_departments[user_id]=[]

    user_aisles[user_id]=[]

    user_none_order_rate[user_id]=[]

    user_order_len[user_id]=[]

    user_total_products[user_id]=0

    user_overall_reorder[user_id]=[]

    for i,(order_number, order_id) in enumerate(users[user_id].items()):

        if(order_number in ["test","train","valid"]):

            continue

        user_buytime_mean[user_id].append(orders_hours[order_id])

        user_buydow_mean[user_id].append(orders_dows[order_id])

        if(int(order_number)!=1):

            user_none_order_rate[user_id].append(orders_isnone[order_id])

        user_order_len[user_id].append(len(orders[order_id]))

        for product_id in orders[order_id]:

            user_total_products[user_id]+=1

            if(product_id not in user_products[user_id]):

                user_products[user_id].append(product_id)

            

            aisles_id=products_aisles[product_id]

            departments_id=products_departments[product_id]

            

            if(departments_id not in user_departments[user_id]):

                user_departments[user_id].append(departments_id)

            if(aisles_id not in user_aisles[user_id]):

                user_aisles[user_id].append(aisles_id)

            user_overall_reorder[user_id].append(orders[order_id][product_id])



outcsv = open("none_test_datas.csv", 'w', encoding = 'UTF-8')





cols=[]

cols.append("user_id")

cols.append("order_id")



cols.append("total_product")

cols.append("total_order")

cols.append("total_distinct_product")

cols.append("totoal_dep")

cols.append("totoal_aisle")



cols.append("order_dow")

cols.append("order_hour")

cols.append("days_since_prior_order")



cols.append("overall_reorder_rate")

cols.append("none_order_rate")



cols.append("mean_order_dow")

cols.append("mean_order_hour")

cols.append("mean_days_since_prior_order")

cols.append("mean_basket_size")



#cols.append("is_none")



outcsv.writelines(','.join(cols)+"\n")

outcsv.flush()



print("Start saving features to csv")

for user_id in tqdm(users):

    if('test' not in list(users[user_id].keys())):

        continue

    

    order_id=users[user_id]["test"]

    

    features=[]

    features.append(user_id)

    features.append(order_id)

    

    features.append(user_total_products[user_id])

    features.append(len(users[user_id])-1)

    features.append(len(user_products[user_id]))

    features.append(len(user_departments[user_id]))

    features.append(len(user_aisles[user_id]))

    

    features.append(orders_dows[order_id])

    features.append(orders_hours[order_id])

    features.append(train_days_since_prior_order[order_id])

    

    overall_reorder_rate=sum(user_overall_reorder[user_id])/len(user_overall_reorder[user_id])

    none_order_rate=sum(user_none_order_rate[user_id])/len(user_none_order_rate[user_id])

    mean_order_dow=sum(user_buydow_mean[user_id])/len(user_buydow_mean[user_id])

    mean_order_hour=sum(user_buytime_mean[user_id])/len(user_buytime_mean[user_id])

    mean_days_since_prior_order=sum(orders_days_since_prior_order[user_id])/len(orders_days_since_prior_order[user_id])

    mean_basket_size=sum(user_order_len[user_id])/len(user_order_len[user_id])

    #is_none=orders_isnone[order_id]

    

    features.append(overall_reorder_rate)

    features.append(none_order_rate)

    features.append(mean_order_dow)

    features.append(mean_order_hour)

    features.append(mean_days_since_prior_order)

    features.append(mean_basket_size)

    #features.append(is_none)

    

    features_str=[]

    for feature in features:

        if(isinstance(feature, float)):

            features_str.append(str(round(feature,5)))

        else:

            features_str.append(str(feature))

            

    

    outcsv.writelines(','.join(features_str)+"\n")

    outcsv.flush()

        
import numpy as np 

import pandas as pd

from sklearn.preprocessing import LabelEncoder



# read datasets

train = pd.read_csv('none_train_datas.csv')

test=pd.read_csv('none_test_datas.csv')

order_id=test['order_id']



y_train = train["is_none"]

y_mean = np.mean(y_train)

       

print('Shape train: {}\n Shape test: {}'.format(train.shape,test.shape))



import xgboost as xgb



xgb_params = {

    'eta': 0.005,

    'max_depth': 6,

    'subsample': 0.8,

    'objective': 'reg:linear',

    'eval_metric': 'logloss',

    'base_score': y_mean,

    'silent': 1

}



dtrain = xgb.DMatrix(train.drop(['user_id','order_id','is_none'], axis=1), y_train)

dtest = xgb.DMatrix(test.drop(['user_id','order_id'], axis=1))

'''

cv_result = xgb.cv(xgb_params, 

                   dtrain, 

                   nfold=10,

                   num_boost_round=1500, # increase to have better results (~700)

                   early_stopping_rounds=50,

                   verbose_eval=50, 

                   show_stdv=False

                  )

'''

#num_boost_rounds = len(cv_result)

#print(num_boost_rounds)

num_boost_rounds=800

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

preds=model.predict(dtest)



out = pd.DataFrame({'order_id': order_id, 'none_pred': preds})

out.to_csv('test_none_pred.csv', index=False)