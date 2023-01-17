# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

from collections import Counter



import os

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_colwidth',500)

pd.set_option('display.max_columns',5000)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading files

train = pd.read_csv('/kaggle/input/train.csv')

coupons = pd.read_csv('/kaggle/input/coupon_item_mapping.csv')

campaign = pd.read_csv('/kaggle/input/campaign_data.csv')

cust_tran = pd.read_csv('/kaggle/input/customer_transaction_data.csv')

cust_demo = pd.read_csv('/kaggle/input/customer_demographics.csv')

items = pd.read_csv('/kaggle/input/item_data.csv')



test = pd.read_csv('/kaggle/input/test_QyjYwdj.csv')

sample = pd.read_csv('/kaggle/input/sample_submission_Byiv0dS.csv')
train.head()
########################### Campaign #############################
campaign.head()
#todatetime

campaign['start_date'] = pd.to_datetime(campaign['start_date'], format = '%d/%m/%y')

campaign['end_date'] = pd.to_datetime(campaign['end_date'], format = '%d/%m/%y')
#adding campaign type to train and test

train['campaign_type'] = train.campaign_id.map(campaign.groupby('campaign_id').campaign_type.apply(lambda x: x.unique()[0]))

test['campaign_type'] = test.campaign_id.map(campaign.groupby('campaign_id').campaign_type.apply(lambda x: x.unique()[0]))
############################ Customer demographics ##############################
cust_demo.head()
#type of family size, no of children = int64

cust_demo['family_size'] = cust_demo.family_size.apply(lambda x: int(re.sub('\+','',x)))

cust_demo['no_of_children'] = cust_demo.no_of_children.apply(lambda x: float(re.sub('\+','',x)) if pd.notna(x) else x)
#Filling nans marital_status



#customers with family size =1 will be single

cust_demo.loc[pd.isnull(cust_demo.marital_status) & (cust_demo.family_size == 1),'marital_status'] = 'Single'



#customers whos fam size - no of childrens == 1, will also be single

cust_demo.loc[(cust_demo.family_size - cust_demo.no_of_children == 1) & pd.isnull(cust_demo.marital_status),'marital_status'] = 'Single'



#from the orignal data we have 142 of 152 customers with diff of 2 in their fam size and #childrens are Married

cust_demo.loc[(pd.isnull(cust_demo.marital_status)) & ((cust_demo.family_size - cust_demo.no_of_children) == 2)  & (pd.notnull(cust_demo.no_of_children)),'marital_status'] = 'Married'



#original data shows customers with fam size == 2, and nans in no of childrens are majorly Married

cust_demo.loc[pd.isnull(cust_demo.marital_status) & (pd.isnull(cust_demo.no_of_children)) & (cust_demo.family_size ==2),'marital_status'] = 'Married'
#Filling nans in no of children



#Married people with family_size ==2 will have 0 childrens

cust_demo.loc[pd.isnull(cust_demo.no_of_children) & (cust_demo.marital_status == 'Married') & (cust_demo.family_size == 2),'no_of_children'] = 0.0



#customers with family size 1 will have zero childrens

cust_demo.loc[pd.isnull(cust_demo.no_of_children) & (cust_demo.family_size == 1), 'no_of_children'] = 0.0



#singles with family size == 2, will probably have 1 child

cust_demo.loc[pd.isnull(cust_demo.no_of_children) & (cust_demo.family_size == 2),'no_of_children'] = 1.0
############################ Customer transactions ############################
cust_tran.head()
#to datetime

cust_tran['date'] = pd.to_datetime(cust_tran['date'])
############################# common ############################### 
#merging train and test with cust_demo on campaign_id

train = pd.merge(train,cust_demo, on='customer_id', how='left')

test = pd.merge(test,cust_demo, on='customer_id', how='left')
train.head()
# "bought_X_vailable" =  Intersection between Items bought by customer previously(from cust_tran) and all items available in coupon provided(from coupons)
#cust2items - dictionary mapping customer_ids to all items bought by them

cust_tran['str_item'] = cust_tran.item_id.apply(lambda x: str(x)) #did this to calculate d_cust2items, no need further

d_cust2items = cust_tran.groupby('customer_id').str_item.apply(lambda x: ' '.join(x)).to_dict()

cust_tran.drop('str_item',axis=1,inplace=True)
#coupon2items - dictionary mapping coupon_ids to all items under them

d_coupon2items = coupons.groupby('coupon_id').item_id.apply(lambda x: ' '.join(list(x.apply(lambda x: str(x))))).to_dict()
#intersect of cust2items and coupon2items (increased score by 0.14)

train['bought_X_vailable'] = train[['coupon_id','customer_id']].apply(lambda x : len(np.intersect1d(d_cust2items[x[1]].split() , d_coupon2items[x[0]].split())) , axis=1)

test['bought_X_vailable'] = test[['coupon_id','customer_id']].apply(lambda x : len(np.intersect1d(d_cust2items[x[1]].split() , d_coupon2items[x[0]].split())) , axis=1)
#############
#item2coupons - dictionary mapping item_ids to all coupons applicable to them

d_item2coupons = coupons.groupby('item_id').coupon_id.apply(lambda x: ' '.join(list(x.apply(lambda x: str(x))))).to_dict()
#adding col for whether coupon was applied on that item (i.e redeemed or not)

cust_tran['redeem'] = cust_tran.coupon_discount.apply(lambda x: 1 if x<0 else 0)
##############  1.) Calculating redeemed % per item from cust_tran

#               2.) Summing all those %'s for items in a coupon, take mean finally

#               3.) map it to coupons
#per_item_redeemed_history = dict mapping item_ids to redeemed %

d_per_item_redeemed_history = ((cust_tran.groupby('item_id').redeem.sum() / cust_tran.groupby('item_id').redeem.count()) *100).to_dict()
#some items corresponding to test coupons are not in d_per_item_redeemed_hist hence need for this func

def item_redeem_func(x):

    for item in d_coupon2items[x].split():

        per = []

        try:

            per.append(d_per_item_redeemed_history[int(item)])



        except:

            pass

    k = [np.mean(per) if pd.isna(np.mean(per)) == False else 0]

    return k[0]
#applying the above func to coupon_id

train['item_redeem'] = train.coupon_id.apply(item_redeem_func)

test['item_redeem'] = test.coupon_id.apply(item_redeem_func)
###############
##### 1.) Calculating redeemed % per customer from cust_tran

#     2.) map it to customer_ids in train and tests
#per_cust_redeem_history - dict mapping customer_id to redemmed %

d_per_cust_redeem_history = ((cust_tran.groupby('customer_id').redeem.sum() / cust_tran.groupby('customer_id').redeem.count())*100).to_dict()
#adding a col for cust redeem #increased score by 0.03

train['cust_redeem'] = train.customer_id.map(d_per_cust_redeem_history)

test['cust_redeem'] = test.customer_id.map(d_per_cust_redeem_history)
###############
#adding net price

#cust_tran['net_price'] = cust_tran['selling_price'] - cust_tran['other_discount'] - cust_tran['coupon_discount']



#dict for cust_id to income bracket

#d_cust2_incomebrac = cust_demo[['customer_id','income_bracket']].set_index('customer_id').to_dict()['income_bracket']



#adding income bracket col in customer trans

#cust_tran['income_bracket'] = cust_tran.customer_id.map(d_cust2_incomebrac)



#merging cust_trans with items on item_id

cust_tran = pd.merge(cust_tran, items, how='left', on='item_id')
cust_tran.head()
#################### 1.) Calculating redeemed % per category ---> per_cat_redeem_history

#                    2.) Calculating redeemed % per customer based on cat using (1) ---> per_cust_redeem_history_catwali

#                    3.) map (2) to customer_ids in train and test
#redeem history based on category

d_per_cat_redeem_history = (cust_tran.groupby('category').redeem.sum() / cust_tran.groupby('category').redeem.count()*1000).to_dict()
#(increased score by 0.0001)

d_per_cust_redeem_history_catwali = cust_tran.groupby('customer_id').category.apply(lambda x: np.mean([d_per_cat_redeem_history[k] for k in x.values]))



train['cat_cust_redeem'] = train.customer_id.map(d_per_cust_redeem_history_catwali)

test['cat_cust_redeem'] = test.customer_id.map(d_per_cust_redeem_history_catwali)
############
############ if for a customer, brands bought by him previously are available in the coupon given, high chance of redeem
#cust2brands - dict mapping customer_ids to all brands bought by them

d_cust2brands = cust_tran.groupby('customer_id').brand.apply(lambda x: ' '.join([str(k) for k in x.unique()])).to_dict()
#item2brand - dict mapping items to their respective brands

d_item2brand = cust_tran.groupby('item_id').brand.apply(lambda x: x.unique()[0]).to_dict()
#filling nans in brand of which we have no prior info

coupons['brand'] = coupons.item_id.map(d_item2brand).fillna('99999999999')
#coupon2brands - dict mapping coupons to all brands available in them to purchase

d_coupon2brands = coupons.groupby('coupon_id').brand.apply(lambda x: ' '.join([str(int(k)) for k in x.unique()])).to_dict()
#getting no of common brands in cust2brands and coupon2brands

train['brand_bot'] = train[['customer_id','coupon_id']].apply(lambda x: len(np.intersect1d(d_cust2brands[x[0]].split(), d_coupon2brands[x[1]].split())), axis=1)

test['brand_bot'] = test[['customer_id','coupon_id']].apply(lambda x: len(np.intersect1d(d_cust2brands[x[0]].split(), d_coupon2brands[x[1]].split())), axis=1)
#########
######### Filling some nans in rented, age_range
#filling nans in train.rented with 2

train.rented.fillna(2,inplace=True)

test.rented.fillna(2,inplace=True)
#imputing age_range based on campaign_id



def d_age(df):

    k = df.groupby('campaign_id').age_range.value_counts()

    k = k.reset_index(name='value').sort_values(['campaign_id','value'], ascending=[True,False])

    d_age = {}

    for i in list(df.campaign_id.unique()):

        df = k.loc[k.campaign_id == i,['age_range','value']]

        df = df.set_index('age_range')

        max_val_per_campaign = df.idxmax().value

        d_age[i] = max_val_per_campaign

        

    return d_age



    

#filling nans with d_age

train.loc[(pd.isnull(train.age_range)),'age_range'] = train.loc[(pd.isnull(train.age_range)),'campaign_id'].map(d_age(train))

test.loc[(pd.isnull(test.age_range)),'age_range'] = test.loc[(pd.isnull(test.age_range)),'campaign_id'].map(d_age(test))

###############
#adding brand (most frequent) per coupon_id

train['brand'] = train.coupon_id.map(coupons.groupby('coupon_id').brand.apply(lambda x: x.values[0]).to_dict())

test['brand'] = test.coupon_id.map(coupons.groupby('coupon_id').brand.apply(lambda x: x.values[0]).to_dict())
############### val set
#array's containing common customer_ids and coupon_ids in train,test ---> (in order to make val set)

commom_cust = np.intersect1d(train.customer_id.unique(),test.customer_id.unique())

commom_coup = np.intersect1d(train.coupon_id.unique(),test.coupon_id.unique())
#adding col to see whether cust, coup is in test or not 

train['test_cust'] = train.customer_id.apply(lambda x: 1 if x in commom_cust else 0)

train['test_coup'] = train.coupon_id.apply(lambda x: 1 if x in commom_coup else 0)
####Validation set



#(len(train[pd.isnull(train.family_size) & (train.redemption_status == 1)]) / len(train)) * 7837 #16

index1 = train[pd.isnull(train.family_size) & (train.redemption_status == 1) & (train.test_cust == 1) & (train.test_coup == 1)].sample(16, random_state=1996).index



#(len(train[pd.notnull(train.family_size) & (train.redemption_status == 1)]) / len(train) ) * 7837 #57

index2 = train[pd.notnull(train.family_size) & (train.redemption_status == 1) & (train.test_cust == 1) & (train.test_coup == 1)].sample(57, random_state=1996).index



#(len(train[pd.isnull(train.family_size) & (train.redemption_status == 0)]) / len(train)) * 7837 #3455

index3 = train[pd.isnull(train.family_size) & (train.redemption_status == 0) & (train.test_cust == 1) & (train.test_coup == 1)].sample(3366, random_state=1996).index



#(len(train[pd.notnull(train.family_size) & (train.redemption_status == 0)]) / len(train)) * 7837 #4309

index4 = train[pd.notnull(train.family_size) & (train.redemption_status == 0) & (train.test_cust == 1) & (train.test_coup == 1)].sample(4309, random_state=1996).index







#new train and val set

val_index = []

for i in [index1,index2, index3, index4]:

    val_index.extend(i)#main val_index

    

train_index = set(train.index)

train_index = train_index.symmetric_difference(val_index)#main train index



new_train = train.loc[train_index]

val = train.loc[val_index].sample(frac=1, random_state = 1996)

new_test = test
#final_train = new_train.dropna(axis=1).drop(['test_cust','test_coup'], axis=1)

#final_test = new_test.dropna(axis=1)#.drop(['coup_redeem'], axis=1)

#val = val.dropna(axis=1).drop(['test_cust','test_coup'], axis=1)



final_train = train.dropna(axis=1).drop(['test_cust','test_coup'], axis=1)

final_test = test.dropna(axis=1)



################# Label Encoding



#label encoding features

final_train['campaign_type'] = final_train.campaign_type.map({'X':0,'Y':1})

#val['campaign_type'] = val.campaign_type.map({'X':0,'Y':1})

final_test['campaign_type'] = final_test.campaign_type.map({'X':0,'Y':1})



final_train['age_range'] = final_train.age_range.map({'46-55':0,'36-45':1,'18-25':2,'26-35':3,'56-70':4,'70+':5})

#val['age_range'] = val.age_range.map({'46-55':0,'36-45':1,'18-25':2,'26-35':3,'56-70':4,'70+':5})

final_test['age_range'] = final_test.age_range.map({'46-55':0,'36-45':1,'18-25':2,'26-35':3,'56-70':4,'70+':5})



###############



############## train_test



#preparing data

X_train = final_train.drop(['redemption_status'],axis=1)

y_train = final_train.redemption_status



#val_x = val.drop(['redemption_status'],axis=1)

#val_y = val.redemption_status



X_test = final_test
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier



preds1= pd.DataFrame()

auc_roc1 = []

val_auc = []



for k in range(0,10):

    df_1 = final_train[final_train.redemption_status == 1]

    df_0 = final_train[final_train.redemption_status == 0].sample(1000, random_state =2*k*k*k)

    

    df = pd.concat([df_0,df_1],axis=0).sample(frac=1)



    X_train = df.drop('redemption_status',axis=1)

    y_train = df.redemption_status



    model1 = XGBClassifier(n_estimators=100, scale_pos_weight=2)

    model1.fit(X_train,y_train)



    #pred by model1

    auc_roc1.append(roc_auc_score(y_train, model1.predict_proba(X_train)[:,1].round(3)))

    preds1['k'+str(k)] = model1.predict_proba(X_test)[:,1].round(3)

    





    print(k, end='')
#pred = preds1.mean(axis=1)

#roc_auc_score(val_y, pred)
##############
pred_test = preds1.mean(axis=1)



sample['redemption_status'] = pred_test



name = 'final.csv'

sample.to_csv(name,index=False)