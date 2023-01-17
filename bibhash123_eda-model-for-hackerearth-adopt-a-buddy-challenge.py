import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import random
random.seed(101)
import datetime
train = pd.read_csv('../input/dataset/train.csv')

test = pd.read_csv('../input/dataset/test.csv')
train.head()
train.info()
train.corr()
train[train["condition"].isnull()]["breed_category"].value_counts()
train[train["condition"].isnull()]["pet_category"].value_counts()
train_data=train.drop("pet_id",axis=1)

test_data = test.drop("pet_id",axis=1)
train_data["condition"]=train_data["condition"].fillna(-1)

test_data["condition"]=test_data["condition"].fillna(-1)
train_data["issue_date"]  = list(map(pd.Timestamp.date,list(map(pd.Timestamp,train_data["issue_date"]))))

test_data["issue_date"]  = list(map(pd.Timestamp.date,list(map(pd.Timestamp,test_data["issue_date"]))))
train_data['list_date'] = list(map(pd.Timestamp.date,list(map(pd.Timestamp,train_data["listing_date"]))))

test_data['list_date'] = list(map(pd.Timestamp.date,list(map(pd.Timestamp,test_data["listing_date"]))))
train_data["gap_days"] = train_data["list_date"]-train_data["issue_date"]

test_data["gap_days"] = test_data["list_date"]-test_data["issue_date"]
train_data["gap_days"]=train_data["gap_days"].apply(lambda x: x.days)

test_data["gap_days"]=test_data["gap_days"].apply(lambda x: x.days)
def time_of_day(col):

    hour = pd.Timestamp.time(pd.Timestamp(col)).hour

    if hour in range(0,6,1):

        return "dawn"

    elif hour in range(6,12,1):

        return "morning"

    elif hour in range(12,18,1):

        return "afternoon"

    elif hour in range(18,24,1):

        return "night"
train_data["time_of_day"]=train_data["listing_date"].apply(time_of_day)

test_data["time_of_day"]=test_data["listing_date"].apply(time_of_day)
train_data["list_year"] = train_data["list_date"].apply(lambda x: x.year)



test_data["list_year"] = test_data["list_date"].apply(lambda x: x.year)
train_data["list_season"] = pd.cut(train_data["list_date"].apply(lambda x: x.month),3,labels=["spring","summer","winter"])



test_data["list_season"] = pd.cut(test_data["list_date"].apply(lambda x: x.month), 3, labels=["spring","summer","winter"])
train_data["list_time_month"] = pd.cut(train_data["list_date"].apply(lambda x: x.day),4,labels=["w1","w2","w3","w4"])

test_data["list_time_month"] = pd.cut(test_data["list_date"].apply(lambda x: x.day),4,labels=["w1","w2","w3","w4"])
train_data.drop(["issue_date","list_date","listing_date"],axis=1,inplace=True)

test_data.drop(["issue_date","list_date","listing_date"],axis=1,inplace=True)
train_data["rare"]=train_data["color_type"].apply(lambda x: 1 if x in train.color_type.value_counts().keys()[-8:] else 0)

test_data["rare"]=test_data["color_type"].apply(lambda x: 1 if x in train.color_type.value_counts().keys()[-8:] else 0)
train_data.corr()
color = pd.get_dummies(train_data["color_type"],drop_first=True)

t_color = pd.get_dummies(test_data["color_type"],drop_first=True)



train_data = pd.concat([train_data.drop("color_type",axis=1),color],axis=1)

test_data = pd.concat([test_data.drop("color_type",axis=1),t_color],axis=1)
test_data.columns
test_data.insert(16,value=np.zeros(8072),column="Black Tiger")
test_data.insert(29,value=np.zeros(8072),column="Brown Tiger")
time= pd.get_dummies(train_data["time_of_day"],drop_first=True)

t_time = pd.get_dummies(test_data["time_of_day"],drop_first=True)



train_data = pd.concat([train_data.drop("time_of_day",axis=1),time],axis=1)

test_data = pd.concat([test_data.drop("time_of_day",axis=1),t_time],axis=1)
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()

train_data["list_year"]=encode.fit_transform(train_data["list_year"])

test_data["list_year"]=encode.transform(test_data["list_year"])
season = pd.get_dummies(train_data["list_season"],drop_first=True)

season_t = pd.get_dummies(test_data["list_season"],drop_first=True)



train_data = pd.concat([train_data.drop("list_season",axis=1),season],axis=1)

test_data = pd.concat([test_data.drop("list_season",axis=1),season_t],axis=1)
month = pd.get_dummies(train_data["list_time_month"],drop_first=True)

month_t = pd.get_dummies(test_data["list_time_month"],drop_first=True)



train_data = pd.concat([train_data.drop("list_time_month",axis=1),month],axis=1)

test_data = pd.concat([test_data.drop("list_time_month",axis=1),month_t],axis=1)
train_data["pet_size"]=pd.cut(train_data[["length(m)","height(cm)"]].apply(lambda cols: cols[0]*cols[1] ,axis=1),4,labels=["small","little","medium","large"])

test_data["pet_size"]=pd.cut(test_data[["length(m)","height(cm)"]].apply(lambda cols: cols[0]*cols[1] ,axis=1),4,labels=["small","little","medium","large"])

size = pd.get_dummies(train_data["pet_size"],drop_first=True)

size_t = pd.get_dummies(test_data["pet_size"],drop_first=True)



train_data = pd.concat([train_data.drop("pet_size",axis=1),size],axis=1)

test_data = pd.concat([test_data.drop("pet_size",axis=1),size_t],axis=1)
train_data2 = train_data.copy()

test_data2 = test_data.copy()
train_data.drop(["length(m)","height(cm)"],axis=1,inplace=True)
test_data.drop(["length(m)","height(cm)"],axis=1,inplace=True)
train_data2.drop(["length(m)","height(cm)"],axis=1,inplace=True)
test_data2.drop(["length(m)","height(cm)"],axis=1,inplace=True)
cond = pd.get_dummies(train_data["condition"],drop_first=True)

cond_t = pd.get_dummies(test_data["condition"],drop_first=True)



train_data= pd.concat([train_data.drop("condition",axis=1),cond],axis=1)

test_data = pd.concat([test_data.drop("condition",axis=1),cond_t],axis=1)



cond2 = pd.get_dummies(train_data2["condition"],drop_first=True)

cond2_t = pd.get_dummies(test_data2["condition"],drop_first=True)



train_data2= pd.concat([train_data2.drop("condition",axis=1),cond2],axis=1)

test_data2 = pd.concat([test_data2.drop("condition",axis=1),cond2_t],axis=1)
x_train = train_data.drop(['breed_category','pet_category'],axis=1).values

x_test = test_data.values



x_train2 = train_data2.drop(['breed_category','pet_category'],axis=1).values

x_test2 = test_data2.values
y_train_breed = train_data["breed_category"]

y_train_pet = train_data2["pet_category"]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
sc2 = StandardScaler()
x_train2 = sc.fit_transform(x_train2)

x_test2 = sc.transform(x_test2)
train["pet_category"].value_counts()
train["breed_category"].value_counts()
from imblearn.over_sampling import SMOTE
sm_pet=SMOTE(sampling_strategy={0:941})



x_train2,y_train_pet = sm_pet.fit_resample(x_train2,y_train_pet)



y_train_pet.value_counts()
sm_breed=SMOTE(sampling_strategy={2:1500})



x_train,y_train_breed = sm_breed.fit_resample(x_train,y_train_breed)



y_train_breed.value_counts()
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
xgb = XGBClassifier(objective="multi:softmax")
parameters = {"learning_rate": [0.3, 0.01,0.1],

               "gamma" : [ 0.3, 0.5, 1, 1.5, 2,2.2,2.4],

               "max_depth": [6, 7, 8, 9],

               "colsample_bytree": [0.3, 0.6, 0.8, 1.0],

               "subsample": [0.2, 0.4, 0.5,0.6,0.8],

               "reg_alpha": [0, 0.5, 1],

               "reg_lambda": [0,1, 1.5, 2],

               "min_child_weight": [1, 3, 5, 7, 9]

              

               }
xgb_rscv = RandomizedSearchCV(xgb, param_distributions = parameters, scoring = "f1_micro",

                             cv = 5, verbose = 2, random_state = 1 )
#xgb_rscv.fit(x_train,y_train_breed)
#print(xgb_rscv.best_score_)

#print(xgb_rscv.best_params_)




xgb = XGBClassifier(objective="multi:softmax",max_depth= 8, gamma =2.2 ,learning_rate=0.01 ,colsample_bytree=1.0 ,subsample=0.8 , reg_alpha=0.5 ,reg_lambda= 2,min_child_weight= 3,random_state=101)



xgb.fit(x_train,y_train_breed)
pred_breed= xgb.predict(x_test)
xgb1 = XGBClassifier(objective="multi:softmax",max_depth=8,random_state=101)
xgb1.fit(x_train2,y_train_pet)
pred_pet = xgb1.predict(x_test2)