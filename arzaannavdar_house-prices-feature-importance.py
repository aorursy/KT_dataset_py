import pandas as pd

import sklearn.preprocessing as pre

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
#Read data into the training set

train_data = pd.read_csv("../input/train.csv")
#Trying to identify NaN values in the variables

total = train_data.isnull().sum().sort_values(ascending = False)

percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total,percent],axis=1,keys = ["Total","Percentage"])

print(missing_data)
#Delete all NaN values for now

train_data = train_data.drop(missing_data[missing_data["Total"]>0].index,1)
#Identify top features using a basic XGBoost

# I'd like to thank this from 

#https://www.kaggle.com/sudalairajkumar/sberbank-russian-housing-market/simple-exploration-notebook-sberbank

#It helped me to understand a simple way to build my feature importance

for f in train_data:

    if train_data[f].dtype == "object":

        lbl=pre.LabelEncoder()

        lbl.fit(list(train_data[f].values))

        train_data[f]=lbl.transform(list(train_data[f].values))



xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



y_train = train_data["price_doc"]

x_train = train_data.drop(["id","timestamp","price_doc"],axis = 1)

dtrain = xgb.DMatrix(x_train,y_train,feature_names = x_train.columns.values)

model = xgb.train(dict(xgb_params,silent=0),dtrain,num_boost_round=100)



fig,ax=plt.subplots(figsize = (12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show
#Change the Price values to log functions

cols = ["price_doc","full_sq","metro_min_avto","sub_area","kindergarten_km","green_zone_km","school_km","park_km","industrial_km"]

train_data["price_doc"]= np.log(train_data["price_doc"])
#Check the plots with each of the variables

corrmat = train_data.corr()

sns.pairplot(train_data[cols],size = 2.5)

plt.show()