import math
import datetime
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import lightgbm as lgb
from hyperopt import hp
from sklearn.model_selection import train_test_split
from hyperopt import fmin,tpe,anneal,STATUS_OK,STATUS_FAIL,Trials
import random
import string
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.feature_extraction.text import CountVectorizer
import catboost as ctb
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_json('../input/prep-geo/prep_geo_train.json')
df_test = pd.read_json('../input/prep-geo/prep_geo_test.json')
id_train = df_train.shape[0]
listing_train = list(df_train.listing_id)
listing_test = list(df_test.listing_id)

target = df_train.interest_level
df = pd.concat([df_train, df_test])
df["price"] = df["price"].clip(upper=13000)
df["logprice"] = np.log(df["price"])
df['half_bathrooms'] = df["bathrooms"] - df["bathrooms"].apply(int)
df["price_t"] = df["price"]/df["bedrooms"]
df["room_sum"] = df["bedrooms"]+df["bathrooms"]
df['price_per_room'] = df['price']/df['room_sum']
df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))

df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day
df["created_hour"] = df["created"].dt.hour
df["created_weekday"] = df["created"].dt.weekday
df["created_week"] = df["created"].dt.week

df["pos"] = df.longitude.round(3).astype(str) + '_' + df.latitude.round(3).astype(str)
vals = df['pos'].value_counts()
dvals = vals.to_dict()
df["density"] = df['pos'].apply(lambda x: dvals.get(x, vals.min()))
# import reverse_geocoder as revgc

# df['area_name'] = df.apply(lambda x: revgc.search([x.latitude, x.longitude])[0]['name'], axis=1)
categorical = ["street_address", "display_address", "manager_id", "building_id",'area_name']
for f in categorical:
        if df[f].dtype=='object':
            lbl = LabelEncoder()
            df[f] = lbl.fit_transform(df[f])
def create_stat_feature(df,group_col,col,name):
    for i in ['min','max','std','mean']:
        df[name+'_'+i] = df.groupby(group_col)[col].transform(i)

create_stat_feature(df,'manager_id','price','price_manager')
create_stat_feature(df,'manager_id','building_id','building_manager')
create_stat_feature(df,'manager_id','street_address','street_manager')
def distance_to(df,cords,name):
    df[name] = df[['latitude', 'longitude']].apply(lambda x:math.sqrt((x[0]-cords[0])**2+(x[1]-cords[1])**2), axis=1)
    
cords_list = [(40.705628,-74.010278),(40.785091,-73.968285),(40.758896,-73.985130),
              (40.748817,-73.985428),(40.712742,-74.013382),(40.706086,-73.996864)]

names_list = ['distance_to_fi','distance_to_cp','distance_to_tq','distance_to_et',
              'distance_to_tf','distance_to_bb']

for i in range(len(cords_list)):
    distance_to(df,cords_list[i],names_list[i])
display = df["display_address"].value_counts()
manager_id = df["manager_id"].value_counts()
building_id = df["building_id"].value_counts()
street = df["street_address"].value_counts()
bedrooms = df["bedrooms"].value_counts()
bathrooms = df["bathrooms"].value_counts()

df["display_count"] = df["display_address"].apply(lambda x:display[x])
df["manager_count"] = df["manager_id"].apply(lambda x:manager_id[x])  
df["building_count"] = df["building_id"].apply(lambda x:building_id[x])
df["street_count"] = df["street_address"].apply(lambda x:street[x])
df["bedrooms_count"] = df["bedrooms"].apply(lambda x:bedrooms[x])
df["bathrooms_count"] = df["bathrooms"].apply(lambda x:bathrooms[x])
df['nums_of_desc'] = df['description']\
        .apply(lambda x:re.sub('['+string.punctuation+']', '', x).split())\
        .apply(lambda x: len([s for s in x if s.isdigit()]))
        
df['has_phone'] = df['description'].apply(lambda x:re.sub('['+string.punctuation+']', '', x).split())\
        .apply(lambda x: [s for s in x if s.isdigit()])\
        .apply(lambda x: len([s for s in x if len(str(s))==10]))\
        .apply(lambda x: 1 if x>0 else 0)

df['has_email'] = df['description'].apply(lambda x: 1 if '@renthop.com' in x else 0)
df['building_id_is_zero'] = df['building_id'].apply(lambda x:1 if x=='0' else 0)
df['num_of_html_tag'] = df.description.apply(lambda x:x.count('<'))
df['num_of_#'] = df.description.apply(lambda x:x.count('#'))
df['num_of_!'] = df.description.apply(lambda x:x.count('!'))
df['num_of_$'] = df.description.apply(lambda x:x.count('$'))
df['num_of_*'] = df.description.apply(lambda x:x.count('*'))
df['num_of_>'] = df.description.apply(lambda x:x.count('>'))
df["geo_area_50"] = \
    df[['latitude', 'longitude']]\
    .apply(lambda x:(int(x[0]*50)%50)*50+(int(-x[1]*50)%50),axis=1)                                         
                         
df["geo_area_100"] = \
    df[['latitude', 'longitude']]\
    .apply(lambda x:(int(x[0]*100)%100)*100+(int(-x[1]*100)%100),axis=1)                                         
  
df["geo_area_200"] = \
    df[['latitude', 'longitude']]\
    .apply(lambda x:(int(x[0]*200)%200)*200+(int(-x[1]*200)%200),axis=1)
df["price_manager"] = df['price'] * df['manager_level_medium'] * df['manager_level_low'] * df['manager_level_high']
df["rating_manager"] = -df['manager_level_medium'] - df['manager_level_low']*2 + df['manager_level_high']*10
def time_long(x,y):
    if x==4:
        return y
    if x==5:
        return 30+y
    if x==6:
        return 30+31+y
    
def merge_nunique(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].nunique()).reset_index()
    add.columns=columns+[cname]
    df = df.merge(add,on=columns,how="left")
    return df

def merge_median(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    add.columns=columns+[cname]
    df = df.merge(add,on=columns,how="left")
    return df

def merge_count(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    add.columns=columns+[cname]
    df = df.merge(add,on=columns,how="left")
    return df

def merge_nunique(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].nunique()).reset_index()
    add.columns=columns+[cname]
    df = df.merge(add,on=columns,how="left")
    return df

def merge_mean(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns=columns+[cname]
    df = df.merge(add,on=columns,how="left")
    return df

def merge_sum(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    add.columns=columns+[cname]
    df = df.merge(add,on=columns,how="left")
    return df
df["time"] = list(map(lambda x, y: time_long(x, y), df["created_month"], df["created_day"]))

df["price_bed"] = df["price"] / (df["bedrooms"] + 1)
df["price_bath"] = df["price"] / (df["bathrooms"] + 1)
df["price_bath_bed"] = df["price"] / (df["bathrooms"] + df["bedrooms"] + 1)
df["bed_bath_dif"] = df["bedrooms"] - df["bathrooms"]
df["bed_bath_per"] = df["bedrooms"] / df["bathrooms"]
df["bed_all_per"] = df["bedrooms"] / df["room_sum"]

df = merge_nunique(df, ["manager_id"], "time", "manager_active")
df = merge_nunique(df, ["manager_id"], "building_id", "manager_building")
df["manager_building_post_rt"] = df["manager_building"] / df["manager_count"]
df["build_day"] = df["manager_building"] / df["manager_active"]

df = merge_nunique(df, ["building_id"], "manager_id", "building_manager")
df = merge_median(df, ["building_id", "bedrooms", "bathrooms"], "price", "building_mean")

Min_lis_id = df["listing_id"].min()
Min_time = df["time"].min()
df["gradient"] = ((df["listing_id"]) - Min_lis_id) / (df["time"] - Min_time)
df["building_dif"] = df["price"] - df["building_mean"]
df["building_rt"] = df["price"] / df["building_mean"]

df["jwd_class"] = list(map(lambda x, y: (int(x * 100) % 100) * 100 + (int(-y * 100) % 100), df["latitude"], df["longitude"]))
df = merge_nunique(df, ["manager_id"], "building_rt", "manager_pay")
df = merge_nunique(df, ["jwd_class"], "manager_id", "manager_num_jwd")
df = merge_nunique(df, ["manager_id"], "jwd_class", "manager_jwd_class")

df = merge_nunique(df, ["jwd_class"], "building_id", "building_num_jwd")
df = merge_median(df, ["bathrooms", "bedrooms"], "price", "fangxing_mean")
df = merge_median(df, ["jwd_class", "bathrooms", "bedrooms"], "price", "type_jwd_price_mean")
df["type_jwd_price_mean_rt"] = df["price"] / df["type_jwd_price_mean"]

df["type_jwd_building_mean_rt"] = df["building_mean"] / df["type_jwd_price_mean"]
df["fangxing_mean_dif_jwd"] = df["fangxing_mean"] - df["type_jwd_price_mean"]
df["fangxing_mean_rt_jwd"] = df["fangxing_mean"] / df["type_jwd_price_mean"]
df = merge_mean(df, ["manager_id"], "type_jwd_price_mean_rt", "manager_pay_jwd")

df = merge_mean(df, ["building_id"], "type_jwd_building_mean_rt", "building_pay_jwd")
df = merge_mean(df, ["jwd_class"], "fangxing_mean_rt_jwd", "jwd_pay_all")
df = merge_mean(df, ["manager_id"], "building_pay_jwd", "manager_own_ud")
df = merge_mean(df, ["manager_id"], "jwd_pay_all", "manager_own_ud_all")
df["manager_building_all_rt"] = df["manager_own_ud"] / df["manager_own_ud_all"]

df["all_hours"] = df["time"] * 24 + df["created_hour"]
df = merge_nunique(df, ["manager_id"], "all_hours", "manager_hours")
df["manager_hours_rt"] = df["manager_hours"] / df["manager_active"]
df["manager_price_mean"] = 0
df = merge_sum(df, ["manager_id"], "price", "manager_price_sum")

df = merge_sum(df, ["manager_id"], "bedrooms", "manager_bedrooms_sum")
df = merge_sum(df, ["manager_id"], "building_dif", "earn_all")
df["manager_price_mean"] = df["manager_price_sum"] / df["manager_bedrooms_sum"]
df["earn_everyday"] = df["earn_all"] / df["manager_active"]

df["earn_all_rt"] = df["earn_all"] / df["manager_price_sum"]
df["manager_price_"] = df["manager_price_sum"] / df["manager_active"]
df = merge_mean(df, ["manager_id"], "created_hour", "manager_post_hour_mean")

df = merge_median(df, ["manager_id"], "longitude", "manager_longitude_median")
df = merge_median(df, ["manager_id"], "latitude", "manager_latitude_median")

df["same"] = list(map(lambda a, b, c, d, e: str(a) + str(b) + str(c) + str(d) + str(e), 
                      df["manager_id"], df["bedrooms"], df["bathrooms"], df["building_id"], 
                      df["features"]))
same_count = df["same"].value_counts()
df["same_count"] = list(map(lambda x: same_count[x], df["same"]))

df = merge_count(df, ["jwd_class"], "listing_id", "listing_num_jwd")
df["building_listing_num_jwd_rt"] = df["building_num_jwd"] / df["listing_num_jwd"]
df = merge_median(df, ["time"], "price", "price_today")
df = merge_median(df, ["created_month"], "price", "price_today_month")
df["price_rt_jwd"] = df["price"] / df["type_jwd_price_mean"]
image_date = pd.read_csv("../input/twosigma-magic-feature/listing_image_time.csv")
image_date.columns = ["listing_id", "image_time_stamp"]

df = pd.merge(df, image_date, on="listing_id", how="left")

df["time"] = df.image_time_stamp.apply(lambda x:datetime.datetime.fromtimestamp(x).strftime('%c'))
df["time"] = pd.to_datetime(df["time"])
df["time_month"] = df["time"].dt.month
df["time_day"] = df["time"].dt.day
df["time_hour"] = df["time"].dt.hour
df["time_weekday"] = df["time"].dt.weekday
df["time_week"] = df["time"].dt.week
df_train = df[df['listing_id'].isin(listing_train)]
df_test = df[df['listing_id'].isin(listing_test)]

df_train['features'] = df_train["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
df_test['features'] = df_test["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(df_train["features"])
te_sparse = tfidf.transform(df_test["features"])
# index=list(range(train_df.shape[0]))
# random.shuffle(index)
# a=[np.nan]*len(train_df)
# b=[np.nan]*len(train_df)
# c=[np.nan]*len(train_df)

# for i in range(5):
#     building_level={}
#     for j in train_df['manager_id'].values:
#         building_level[j]=[0,0,0]
    
#     test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
#     train_index=list(set(index).difference(test_index))
    
#     for j in train_index:
#         temp=train_df.iloc[j]
#         if temp['interest_level']=='low':
#             building_level[temp['manager_id']][0]+=1
#         if temp['interest_level']=='medium':
#             building_level[temp['manager_id']][1]+=1
#         if temp['interest_level']=='high':
#             building_level[temp['manager_id']][2]+=1
            
#     for j in test_index:
#         temp=train_df.iloc[j]
#         if sum(building_level[temp['manager_id']])!=0:
#             a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
#             b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
#             c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
            
# train_df['manager_level_low']=a
# train_df['manager_level_medium']=b
# train_df['manager_level_high']=c

# a=[]
# b=[]
# c=[]
# building_level={}
# for j in train_df['manager_id'].values:
#     building_level[j]=[0,0,0]

# for j in range(train_df.shape[0]):
#     temp=train_df.iloc[j]
#     if temp['interest_level']=='low':
#         building_level[temp['manager_id']][0]+=1
#     if temp['interest_level']=='medium':
#         building_level[temp['manager_id']][1]+=1
#     if temp['interest_level']=='high':
#         building_level[temp['manager_id']][2]+=1

# for i in test_df['manager_id'].values:
#     if i not in building_level.keys():
#         a.append(np.nan)
#         b.append(np.nan)
#         c.append(np.nan)
#     else:
#         a.append(building_level[i][0]*1.0/sum(building_level[i]))
#         b.append(building_level[i][1]*1.0/sum(building_level[i]))
#         c.append(building_level[i][2]*1.0/sum(building_level[i]))
# test_df['manager_level_low']=a
# test_df['manager_level_medium']=b
# test_df['manager_level_high']=c
df_train["interest_level"] = df_train["interest_level"].map({'high':0, 'medium':1, 'low':2})
train_y = df_train["interest_level"]

df_train = df_train.drop(['created','interest_level','building_mean','room_sum',
                         'type_jwd_price_mean_rt','building_manager','building_num_jwd',
                         'display_count','time','features','type_jwd_price_mean','photos',
                         'same','description','pos','fangxing_mean','area_admin2'],axis=1)

df_test = df_test.drop(['created','interest_level','building_mean','room_sum',
                         'type_jwd_price_mean_rt','building_manager','building_num_jwd',
                         'display_count','time','features','type_jwd_price_mean','photos',
                         'same','description','pos','fangxing_mean','area_admin2'],axis=1)

train_X = sparse.hstack([df_train, tr_sparse]).tocsr()
test_X = sparse.hstack([df_test, te_sparse]).tocsr()
from hyperopt import hp
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import randint

# XGB parameters
xgb_space = {
    "eta": hp.quniform("eta", 0.025, 1, 0.025),
    "max_depth": hp.choice("max_depth", np.arange(1, 20, dtype=int)),
    "min_child_weight": hp.choice("min_child_weight", np.arange(0, 100, dtype=int)),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.025, 1, 0.025),
    "subsample": hp.uniform("subsample", 0.8, 1),
    "n_estimators": hp.choice("n_estimators", np.arange(100, 2000, dtype=int)),
    "gamma": hp.choice("gamma", np.arange(0, 100, dtype=int)),
    #     'tree_method':      'gpu_hist',
    "objective": "multi:softprob",
}
xgb_fit_params = {
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 30,
    #'num_class': 3,
    "verbose": False,
}
xgb_params = dict()
xgb_params["reg_params"] = xgb_space
xgb_params["fit_params"] = xgb_fit_params

# LightGBM parameters
lgb_space = space = {
    "class_weight": hp.choice("class_weight", [None, "balanced"]),
    "boosting_type": hp.choice(
        "boosting_type",
        [
            {
                "boosting_type": "gbdt",
                "subsample": hp.uniform("gdbt_subsample", 0.5, 1),
            },
            {
                "boosting_type": "dart",
                "subsample": hp.uniform("dart_subsample", 0.5, 1),
            },
            {"boosting_type": "goss"},
        ],
    ),
    "n_estimators": hp.choice("n_estimators", np.arange(100, 2000, dtype=int)),
    "num_leaves": hp.choice("num_leaves", np.arange(10, 150, dtype=int)),
    "learning_rate": hp.quniform("learning_rate", 0.025, 1, 0.025),
    "subsample_for_bin": hp.choice(
        "subsample_for_bin", np.arange(2000, 300000, dtype=int)
    ),
    "min_child_samples": hp.choice("min_child_samples", np.arange(20, 500, dtype=int)),
    "reg_alpha": hp.quniform("reg_alpha", 0.025, 1, 0.025),
    "reg_lambda": hp.quniform("reg_lambda", 0.025, 1, 0.025),
    "max_depth": hp.choice("max_depth", np.arange(2, 20, dtype=int)),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.025, 1, 0.025),
}

lgb_fit_params = {
    "eval_metric": "multi_logloss",
    "early_stopping_rounds": 30,
    #    'num_class': 3,
    "verbose": False,
}
lgb_params = dict()
lgb_params["reg_params"] = lgb_space
lgb_params["fit_params"] = lgb_fit_params

# CatBoost parameters
ctb_space = {
    "learning_rate": hp.quniform("learning_rate", 0.025, 1, 0.025),
    "max_depth": hp.choice("max_depth", np.arange(2, 16, dtype=int)),
    #'colsample_bylevel': hp.quniform('colsample_bylevel', 0.025, 1, 0.025),
    "n_estimators": hp.choice("n_estimators", np.arange(100, 2000, dtype=int)),
    "l2_leaf_reg": hp.choice("l2_leaf_reg", np.arange(2, 100, dtype=int)),
    "border_count": hp.choice("border_count", np.arange(5, 200, dtype=int)),
    "eval_metric": "MultiClass",
}
ctb_fit_params = {
    "early_stopping_rounds": 30,
    "verbose": False,
    #     'task_type': 'GPU'
    #    'classes_count': 3
}
ctb_params = dict()
ctb_params["reg_params"] = ctb_space
ctb_params["fit_params"] = ctb_fit_params

# RandomForest parameters
rf_space = {
    "max_depth": hp.choice("max_depth", np.arange(1, 30, dtype=int)),
    "n_estimators": hp.choice("n_estimators", np.arange(100, 2000, dtype=int)),
    "max_features": hp.choice("max_features", np.arange(1, 150)),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "min_samples_split": hp.choice("min_samples_split", np.arange(2, 100, dtype=int)),
    "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(1, 20, dtype=int)),
}
rf_fit_params = {"verbose": None}

rf_params = dict()
rf_params["reg_params"] = rf_space
rf_params["fit_params"] = rf_fit_params
# XGB parameters
xgb_space_reg = {
    "eta": hp.quniform("eta", 0.025, 1, 0.025),
    "max_depth": hp.choice("max_depth", np.arange(1, 20, dtype=int)),
    "min_child_weight": hp.choice("min_child_weight", np.arange(0, 100, dtype=int)),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.025, 1, 0.025),
    "subsample": hp.uniform("subsample", 0.8, 1),
    "n_estimators": hp.choice("n_estimators", np.arange(100, 2000, dtype=int)),
    "gamma": hp.choice("gamma", np.arange(0, 100, dtype=int)),
    #     'tree_method':      'gpu_hist',
    "objective": "reg:squarederror",
}
xgb_fit_params_reg = {
    "eval_metric": "rmse",
    "early_stopping_rounds": 30,
    "verbose": False,
}
xgb_params_reg = dict()
xgb_params_reg["reg_params"] = xgb_space_reg
xgb_params_reg["fit_params"] = xgb_fit_params_reg

# LightGBM parameters
lgb_space_reg = space = {
    "n_estimators": hp.choice("n_estimators", np.arange(100, 2000, dtype=int)),
    "num_leaves": hp.choice("num_leaves", np.arange(10, 150, dtype=int)),
    "learning_rate": hp.quniform("learning_rate", 0.025, 1, 0.025),
    "subsample_for_bin": hp.choice(
        "subsample_for_bin", np.arange(2000, 300000, dtype=int)
    ),
    "min_child_samples": hp.choice("min_child_samples", np.arange(20, 500, dtype=int)),
    "reg_alpha": hp.quniform("reg_alpha", 0.025, 1, 0.025),
    "reg_lambda": hp.quniform("reg_lambda", 0.025, 1, 0.025),
    "max_depth": hp.choice("max_depth", np.arange(2, 20, dtype=int)),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.025, 1, 0.025),
}

lgb_fit_params_reg = {
    "eval_metric": "l2",
    "early_stopping_rounds": 30,
    "verbose": False,
}
lgb_params_reg = dict()
lgb_params_reg["reg_params"] = lgb_space_reg
lgb_params_reg["fit_params"] = lgb_fit_params_reg

# CatBoost parameters
ctb_space_reg = {
    "learning_rate": hp.quniform("learning_rate", 0.025, 1, 0.025),
    "max_depth": hp.choice("max_depth", np.arange(2, 16, dtype=int)),
    #'colsample_bylevel': hp.quniform('colsample_bylevel', 0.025, 1, 0.025),
    "n_estimators": hp.choice("n_estimators", np.arange(100, 2000, dtype=int)),
    "l2_leaf_reg": hp.choice("l2_leaf_reg", np.arange(2, 100, dtype=int)),
    "border_count": hp.choice("border_count", np.arange(5, 200, dtype=int)),
    "eval_metric": "RMSE",
}
ctb_fit_params_reg = {"early_stopping_rounds": 30, "verbose": False}
ctb_params_reg = dict()
ctb_params_reg["reg_params"] = ctb_space_reg
ctb_params_reg["fit_params"] = ctb_fit_params_reg

# RandomForest parameters
rf_space_reg = {
    "max_depth": hp.choice("max_depth", np.arange(1, 30, dtype=int)),
    "n_estimators": hp.choice("n_estimators", np.arange(100, 2000, dtype=int)),
    "max_features": hp.choice("max_features", np.arange(1, 150)),
    "min_samples_split": hp.choice("min_samples_split", np.arange(2, 100, dtype=int)),
    "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(1, 20, dtype=int)),
}
rf_fit_params_reg = {"verbose": None}

rf_params_reg = dict()
rf_params_reg["reg_params"] = rf_space_reg
rf_params_reg["fit_params"] = rf_fit_params_reg
class HPOpt(object):
    def __init__(self, x_train, x_test, y_train, y_test, x_valid=None, y_valid=None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.valid = None
        if (x_valid is not None) and (y_valid is not None):
            self.x_valid = x_valid
            self.y_valid = y_valid
            self.valid = True

    def process(self, fn_name, space, trials, algo, max_evals, random_state=42, cv=3):
        self.random_state = random_state
        self.cv = cv
        if self.valid == True:
            space["fit_params"]["eval_set"] = [
                (self.x_train, self.y_train),
                (self.x_valid, self.y_valid),
            ]
        else:
            space["fit_params"]["eval_set"] = [(self.x_train, self.y_train)]
        self.fit_params = space["fit_params"]
        fn = getattr(self, fn_name)
        result = fmin(
            fn=fn,
            space=space["reg_params"],
            algo=algo,
            max_evals=max_evals,
            trials=trials,
        )

        return result

    def xgb_reg(self, para):
        model = xgb.XGBRegressor(random_state=self.random_state, **para)
        score = -cross_val_score(
            model,
            self.x_train,
            self.y_train,
            cv=self.cv,
            scoring="neg_mean_squared_error",
            fit_params=self.fit_params,
        ).mean()
        return score

    def lgb_reg(self, para):
        model = lgb.LGBMRegressor(random_state=self.random_state, **para)

        score = -cross_val_score(
            model,
            self.x_train,
            self.y_train,
            cv=self.cv,
            scoring="neg_mean_squared_error",
            fit_params=self.fit_params,
        ).mean()
        return score

    def ctb_reg(self, para):
        model = ctb.CatBoostRegressor(random_state=self.random_state, **para)
        score = -cross_val_score(
            model,
            self.x_train,
            self.y_train,
            cv=self.cv,
            scoring="neg_mean_squared_error",
            fit_params=self.fit_params,
        ).mean()
        return score

    def rf_reg(self, para):
        model = RandomForestRegressor(random_state=self.random_state, **para)
        score = -cross_val_score(
            model,
            self.x_train,
            self.y_train,
            cv=self.cv,
            scoring="neg_mean_squared_error",
        ).mean()
        return score

    def et_reg(self, para):
        model = ExtraTreesRegressor(random_state=self.random_state, **para)
        score = -cross_val_score(
            model,
            self.x_train,
            self.y_train,
            cv=self.cv,
            scoring="neg_mean_squared_error",
        ).mean()
        return score

    def xgb_clf(self, para):
        model = xgb.XGBClassifier(random_state=self.random_state, **para)
        score = -cross_val_score(
            model,
            self.x_train,
            self.y_train,
            cv=self.cv,
            scoring="neg_log_loss",
            fit_params=self.fit_params,
        ).mean()
        return score

    def lgb_clf(self, para):
        model = lgb.LGBMClassifier(random_state=self.random_state, **para)
        score = -cross_val_score(
            model,
            self.x_train,
            self.y_train,
            cv=self.cv,
            scoring="neg_log_loss",
            fit_params=self.fit_params,
        ).mean()
        return score

    def ctb_clf(self, para):
        model = ctb.CatBoostClassifier(
            random_state=self.random_state, task_type="GPU", **para
        )
        score = -cross_val_score(
            model,
            self.x_train,
            self.y_train,
            cv=self.cv,
            scoring="neg_log_loss",
            fit_params=self.fit_params,
        ).mean()
        return score

    def rf_clf(self, para):
        model = RandomForestClassifier(random_state=self.random_state, **para)
        score = -cross_val_score(
            model, self.x_train, self.y_train, cv=self.cv, scoring="neg_log_loss",
        ).mean()
        return score

    def et_clf(self, para):
        model = ExtraTreesClassifier(random_state=self.random_state, **para)
        score = -cross_val_score(
            model, self.x_train, self.y_train, cv=self.cv, scoring="neg_log_loss",
        ).mean()
        return score
X_train, X_test, y_train, y_test = train_test_split(
     train_X, train_y, test_size=0.15, random_state=42)
# %%time
# obj = HPOpt(X_train, X_test, y_train, y_test)

# lgb_opt = obj.process(fn_name='lgb_clf', space=lgb_params, trials=Trials(), algo=tpe.suggest, max_evals=300, random_state=42, cv=5)
# xgb_opt = obj.process(fn_name='xgb_clf', space=xgb_params, trials=Trials(), algo=tpe.suggest, max_evals=350, random_state=42, cv=5)
# ctb_opt = obj.process(fn_name='ctb_clf', space=ctb_params, trials=Trials(), algo=tpe.suggest, max_evals=300, random_state=42, cv=5)
# lgb_opt_reg = obj.process(fn_name='lgb_reg', space=lgb_params_reg, trials=Trials(), algo=tpe.suggest, max_evals=200, random_state=42, cv=5)
# xgb_opt_reg = obj.process(fn_name='xgb_reg', space=xgb_params_reg, trials=Trials(), algo=tpe.suggest, max_evals=200, random_state=42, cv=5)
# ctb_opt_reg = obj.process(fn_name='ctb_reg', space=ctb_params_reg, trials=Trials(), algo=tpe.suggest, max_evals=100, random_state=42, cv=5)
# !git clone https://github.com/h2oai/pystacknet
# !cd pystacknet
# !python setup.py install
lgb_opt = {
    "colsample_bytree": 0.275,
    "learning_rate": 0.025,
    "max_depth": 9,
    "min_child_samples": 11,
    "n_estimators": 1367,
    "num_leaves": 17,
    "reg_alpha": 0.775,
    "reg_lambda": 0.1,
    "subsample_for_bin": 246417,
}
xgb_opt = {
    "colsample_bytree": 0.30000000000000004,
    "eta": 0.0325,
    "gamma": 4,
    "max_depth": 9,
    "min_child_weight": 10,
    "n_estimators": 1285,
    "subsample": 0.8981184721176737,
}
lgb_opt_reg = {
    "colsample_bytree": 0.15000000000000002,
    "learning_rate": 0.0325,
    "max_depth": 7,
    "min_child_samples": 234,
    "n_estimators": 1355,
    "num_leaves": 73,
    "reg_alpha": 0.525,
    "reg_lambda": 0.875,
    "subsample_for_bin": 81298,
}
xgb_opt_reg = {
    "colsample_bytree": 0.47500000000000003,
    "eta": 0.0325,
    "gamma": 4,
    "max_depth": 14,
    "min_child_weight": 32,
    "n_estimators": 326,
    "subsample": 0.9291710070461443,
}
lgb_opt_reg1 = {
    "colsample_bytree": 0.30000000000000004,
    "learning_rate": 0.025,
    "max_depth": 12,
    "min_child_samples": 433,
    "n_estimators": 1772,
    "num_leaves": 84,
    "reg_alpha": 0.525,
    "reg_lambda": 0.9750000000000001,
    "subsample_for_bin": 276155,
}
xgb_opt_reg1 = {
    "colsample_bytree": 1.0,
    "eta": 0.07500000000000001,
    "gamma": 7,
    "max_depth": 7,
    "min_child_weight": 88,
    "n_estimators": 336,
    "subsample": 0.9703796510413418,
}
ctb_opt = {'learning_rate': 0.17500000000000002, 'max_depth': 3, 'n_estimators': 903}

ctb_opt_reg = {'learning_rate': 0.625, 'max_depth': 9, 'n_estimators': 284}

xgb_opt1 = {
    "colsample_bytree": 0.7,
    "eta": 0.021,
    "gamma": 4,
    "max_depth": 6,
    "min_child_weight": 1,
    "n_estimators": 1285,
    "subsample": 0.7,
    "objective": 'multi:softprob',
    "eval_metric": 'mlogloss',
}

lgb_opt2 = {'class_weight': None, 'colsample_bytree': 0.325, 'learning_rate': 0.325, 'max_depth': 0, 'min_child_samples': 399, 'n_estimators': 644, 'num_leaves': 33, 'reg_alpha': 0.8, 'reg_lambda': 0.125, 'subsample_for_bin': 297903}
lgb_opt3 = {'class_weight': None, 'colsample_bytree': 0.65, 'learning_rate': 0.225, 'max_depth': 1, 'min_child_samples': 160, 'n_estimators': 294, 'num_leaves': 84, 'reg_alpha': 0.625, 'reg_lambda': 0.25, 'subsample_for_bin': 93757}
lgb_opt4 = {'class_weight': None, 'colsample_bytree': 0.5750000000000001, 'learning_rate': 0.225, 'max_depth': 0, 'min_child_samples': 360, 'n_estimators': 845, 'num_leaves': 12, 'reg_alpha': 1.0, 'reg_lambda': 0.07500000000000001, 'subsample_for_bin': 13177}
lgb_opt5 = {'class_weight': None, 'colsample_bytree': 0.9, 'learning_rate': 0.025, 'max_depth': 10, 'min_child_samples': 423, 'n_estimators': 718, 'num_leaves': 125, 'reg_alpha': 0.17500000000000002, 'reg_lambda': 0.675, 'subsample_for_bin': 183758}
lgb_opt6 = {'class_weight': None, 'colsample_bytree': 0.525, 'learning_rate': 0.1, 'max_depth': 0, 'min_child_samples': 421, 'n_estimators': 1654, 'num_leaves': 50, 'reg_alpha': 0.875, 'reg_lambda': 0.7250000000000001, 'subsample_for_bin': 7469}
lgb_opt7 = {'class_weight': None, 'colsample_bytree': 0.925, 'learning_rate': 0.15000000000000002, 'max_depth': 2, 'min_child_samples': 388, 'n_estimators': 451, 'num_leaves': 31, 'reg_alpha': 0.9750000000000001, 'reg_lambda': 0.7250000000000001, 'subsample_for_bin': 256587}
lgb_opt8 = {'class_weight': None, 'colsample_bytree': 0.17500000000000002, 'learning_rate': 0.05, 'max_depth': 11, 'min_child_samples': 282, 'n_estimators': 509, 'num_leaves': 122, 'reg_alpha': 0.42500000000000004, 'reg_lambda': 0.07500000000000001, 'subsample_for_bin': 190458}
lgb_opt9 = {'class_weight': None, 'colsample_bytree': 0.5750000000000001, 'learning_rate': 0.025, 'max_depth': 6, 'min_child_samples': 291, 'n_estimators': 739, 'num_leaves': 66, 'reg_alpha': 0.325, 'reg_lambda': 0.5, 'subsample_for_bin': 101043}
lgb_opt10 = {'class_weight': None, 'colsample_bytree': 0.15000000000000002, 'learning_rate': 0.07500000000000001, 'max_depth': 14, 'min_child_samples': 188, 'n_estimators': 207, 'num_leaves': 32, 'reg_alpha': 0.45, 'reg_lambda': 0.625, 'subsample_for_bin': 246295}
lgb_opt11 = {'class_weight': None, 'colsample_bytree': 0.07500000000000001, 'learning_rate': 0.025, 'max_depth': 10, 'min_child_samples': 316, 'n_estimators': 1494, 'num_leaves': 48, 'reg_alpha': 0.7000000000000001, 'reg_lambda': 0.17500000000000002, 'subsample_for_bin': 69709}
lgb_opt12 = {'class_weight': None, 'colsample_bytree': 0.7250000000000001, 'learning_rate': 0.25, 'max_depth': 0, 'min_child_samples': 111, 'n_estimators': 819, 'num_leaves': 17, 'reg_alpha': 0.525, 'reg_lambda': 0.5, 'subsample_for_bin': 110536}
lgb_opt13 = {'class_weight': None, 'colsample_bytree': 0.30000000000000004, 'learning_rate': 0.05, 'max_depth': 1, 'min_child_samples': 93, 'n_estimators': 1201, 'num_leaves': 32, 'reg_alpha': 1.0, 'reg_lambda': 0.225, 'subsample_for_bin': 32919}
lgb_opt14 = {'class_weight': None, 'colsample_bytree': 0.42500000000000004, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_samples': 131, 'n_estimators': 953, 'num_leaves': 103, 'reg_alpha': 0.07500000000000001, 'reg_lambda': 0.625, 'subsample_for_bin': 61887}
lgb_opt15 = {'class_weight': None, 'colsample_bytree': 0.675, 'learning_rate': 0.025, 'max_depth': 10, 'min_child_samples': 243, 'n_estimators': 408, 'num_leaves': 119, 'reg_alpha': 0.17500000000000002, 'reg_lambda': 0.775, 'subsample_for_bin': 17006}
lgb_opt16 = {'class_weight': None, 'colsample_bytree': 0.5, 'learning_rate': 0.025, 'max_depth': 3, 'min_child_samples': 287, 'n_estimators': 769, 'num_leaves': 50, 'reg_alpha': 0.55, 'reg_lambda': 0.42500000000000004, 'subsample_for_bin': 274633}
lgb_opt17 = {'class_weight': None, 'colsample_bytree': 0.30000000000000004, 'learning_rate': 0.025, 'max_depth': 2, 'min_child_samples': 111, 'n_estimators': 1877, 'num_leaves': 77, 'reg_alpha': 0.925, 'reg_lambda': 0.8500000000000001, 'subsample_for_bin': 172944}
lgb_opt18 = {'class_weight': None, 'colsample_bytree': 0.775, 'learning_rate': 0.07500000000000001, 'max_depth': 8, 'min_child_samples': 225, 'n_estimators': 85, 'num_leaves': 105, 'reg_alpha': 0.775, 'reg_lambda': 0.775, 'subsample_for_bin': 174495}
lgb_opt19 = {'class_weight': None, 'colsample_bytree': 0.25, 'learning_rate': 0.025, 'max_depth': 14, 'min_child_samples': 302, 'n_estimators': 1164, 'num_leaves': 23, 'reg_alpha': 0.42500000000000004, 'reg_lambda': 0.45, 'subsample_for_bin': 167143}
lgb_opt20 = {'class_weight': None, 'colsample_bytree': 0.9750000000000001, 'learning_rate': 0.07500000000000001, 'max_depth': 15, 'min_child_samples': 281, 'n_estimators': 137, 'num_leaves': 63, 'reg_alpha': 0.525, 'reg_lambda': 0.47500000000000003, 'subsample_for_bin': 9438}
lgb_opt21 = {'class_weight': None, 'colsample_bytree': 0.35000000000000003, 'learning_rate': 0.025, 'max_depth': 13, 'min_child_samples': 110, 'n_estimators': 394, 'num_leaves': 29, 'reg_alpha': 0.07500000000000001, 'reg_lambda': 0.2, 'subsample_for_bin': 99191}
lgb_opt22 = {'class_weight': None, 'colsample_bytree': 0.6000000000000001, 'learning_rate': 0.025, 'max_depth': 13, 'min_child_samples': 297, 'n_estimators': 1302, 'num_leaves': 16, 'reg_alpha': 0.225, 'reg_lambda': 0.65, 'subsample_for_bin': 162254}
lgb_opt23 = {'class_weight': None, 'colsample_bytree': 0.675, 'learning_rate': 0.07500000000000001, 'max_depth': 4, 'min_child_samples': 75, 'n_estimators': 129, 'num_leaves': 28, 'reg_alpha': 0.925, 'reg_lambda': 0.125, 'subsample_for_bin': 88871}
lgb_opt24 = {'class_weight': None, 'colsample_bytree': 0.25, 'learning_rate': 0.25, 'max_depth': 0, 'min_child_samples': 197, 'n_estimators': 357, 'num_leaves': 11, 'reg_alpha': 0.05, 'reg_lambda': 0.8, 'subsample_for_bin': 40493}
lgb_opt25 = {'class_weight': "balanced", 'colsample_bytree': 0.17500000000000002, 'learning_rate': 0.025, 'max_depth': 10, 'min_child_samples': 97, 'n_estimators': 1806, 'num_leaves': 99, 'reg_alpha': 0.5750000000000001, 'reg_lambda': 0.47500000000000003, 'subsample_for_bin': 14141}
lgb_opt26 = {'class_weight': None, 'colsample_bytree': 0.07500000000000001, 'learning_rate': 0.05, 'max_depth': 4, 'min_child_samples': 310, 'n_estimators': 414, 'num_leaves': 102, 'reg_alpha': 0.42500000000000004, 'reg_lambda': 0.6000000000000001, 'subsample_for_bin': 276631}
lgb_opt27 = {'class_weight': None, 'colsample_bytree': 0.675, 'learning_rate': 0.025, 'max_depth': 15, 'min_child_samples': 349, 'n_estimators': 285, 'num_leaves': 63, 'reg_alpha': 0.025, 'reg_lambda': 0.9500000000000001, 'subsample_for_bin': 24238}
lgb_opt28 = {'class_weight': None, 'colsample_bytree': 0.75, 'learning_rate': 0.025, 'max_depth': 9, 'min_child_samples': 179, 'n_estimators': 1166, 'num_leaves': 33, 'reg_alpha': 0.025, 'reg_lambda': 0.35000000000000003, 'subsample_for_bin': 103664}
lgb_opt29 = {'class_weight': None, 'colsample_bytree': 0.375, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_samples': 419, 'n_estimators': 943, 'num_leaves': 97, 'reg_alpha': 0.325, 'reg_lambda': 0.525, 'subsample_for_bin': 133484}
lgb_opt30 = {'class_weight': None, 'colsample_bytree': 0.05, 'learning_rate': 0.17500000000000002, 'max_depth': 0, 'min_child_samples': 20, 'n_estimators': 1374, 'num_leaves': 89, 'reg_alpha': 0.525, 'reg_lambda': 0.30000000000000004, 'subsample_for_bin': 295852}
lgb_opt31 = {'class_weight': None, 'colsample_bytree': 0.225, 'learning_rate': 0.05, 'max_depth': 14, 'min_child_samples': 359, 'n_estimators': 364, 'num_leaves': 120, 'reg_alpha': 0.75, 'reg_lambda': 0.75, 'subsample_for_bin': 182658}
lgb_opt32 = {'class_weight': None, 'colsample_bytree': 0.325, 'learning_rate': 0.025, 'max_depth': 14, 'min_child_samples': 17, 'n_estimators': 715, 'num_leaves': 61, 'reg_alpha': 0.2, 'reg_lambda': 0.55, 'subsample_for_bin': 266004}
models = [
    [
        ctb.CatBoostClassifier(iterations=2000, verbose=0, task_type="GPU"),
        ctb.CatBoostClassifier(verbose=0, task_type="GPU", **ctb_opt),
        lgb.LGBMClassifier(**lgb_opt),
        xgb.XGBClassifier(**xgb_opt),
        xgb.XGBClassifier(**xgb_opt1),
        lgb.LGBMClassifier(),
        xgb.XGBClassifier(),
        lgb.LGBMRegressor(**lgb_opt_reg),
        xgb.XGBRegressor(**xgb_opt_reg),
        lgb.LGBMRegressor(**lgb_opt_reg1),
        xgb.XGBRegressor(**xgb_opt_reg1),
        lgb.LGBMRegressor(),
        xgb.XGBRegressor(),
        ctb.CatBoostRegressor(iterations=2000, verbose=0, task_type="GPU"),
        ctb.CatBoostRegressor(verbose=0, task_type="GPU", **ctb_opt_reg),
        lgb.LGBMClassifier(**lgb_opt2),
        lgb.LGBMClassifier(**lgb_opt3),
        lgb.LGBMClassifier(**lgb_opt4),
        lgb.LGBMClassifier(**lgb_opt5),
        lgb.LGBMClassifier(**lgb_opt6),
        lgb.LGBMClassifier(**lgb_opt7),
        lgb.LGBMClassifier(**lgb_opt8),
        lgb.LGBMClassifier(**lgb_opt9),
        lgb.LGBMClassifier(**lgb_opt10),
        lgb.LGBMClassifier(**lgb_opt11),
        lgb.LGBMClassifier(**lgb_opt12),
        lgb.LGBMClassifier(**lgb_opt13),
        lgb.LGBMClassifier(**lgb_opt14),
        lgb.LGBMClassifier(**lgb_opt15),
        lgb.LGBMClassifier(**lgb_opt16),
        lgb.LGBMClassifier(**lgb_opt17),
        lgb.LGBMClassifier(**lgb_opt18),
        lgb.LGBMClassifier(**lgb_opt19),
        lgb.LGBMClassifier(**lgb_opt20),
        lgb.LGBMClassifier(**lgb_opt21),
        lgb.LGBMClassifier(**lgb_opt22),
        lgb.LGBMClassifier(**lgb_opt23),
        lgb.LGBMClassifier(**lgb_opt24),
        lgb.LGBMClassifier(**lgb_opt25),
        lgb.LGBMClassifier(**lgb_opt26),
        lgb.LGBMClassifier(**lgb_opt27),
        lgb.LGBMClassifier(**lgb_opt28),
        lgb.LGBMClassifier(**lgb_opt29),
        lgb.LGBMClassifier(**lgb_opt30),
        lgb.LGBMClassifier(**lgb_opt31),
        lgb.LGBMClassifier(**lgb_opt32),
    ],
    [lgb.LGBMClassifier()],
]
%%time
from pystacknet.pystacknet import StackNetClassifier

model = StackNetClassifier(
    models,
    metric="logloss",
    folds=5,
    restacking=True,
    use_retraining=True,
    use_proba=True,
    random_state=42,
    verbose=0,
)
  
model.fit(train_X, train_y)
preds = model.predict_proba(test_X)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = df_test.listing_id.values
out_df.to_csv("sub_stacking.csv", index=False)
from IPython.display import Image
Image("../input/prep-geo/sub.png")
