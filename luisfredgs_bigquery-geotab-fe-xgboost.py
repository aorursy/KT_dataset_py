import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import folium

from folium.plugins import HeatMap

from catboost import CatBoostRegressor

import xgboost

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")
train_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/train.csv")

test_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/test.csv")
train_df.head(10)
ax = sns.countplot(x="City", data=train_df)

for item in ax.get_xticklabels():

    item.set_rotation(45)
#Creating Dummies for train Data

dfcity= pd.get_dummies(train_df["City"],prefix = 'city')

dfen = pd.get_dummies(train_df["EntryHeading"],prefix = 'en')

dfex = pd.get_dummies(train_df["ExitHeading"],prefix = 'ex')



train_df = pd.concat([train_df,dfcity],axis=1)

train_df = pd.concat([train_df,dfen],axis=1)

train_df = pd.concat([train_df,dfex],axis=1)



#Creating Dummies for test Data

dfcitytest= pd.get_dummies(test_df["City"],prefix = 'city')

dfent = pd.get_dummies(test_df["EntryHeading"],prefix = 'en')

dfext = pd.get_dummies(test_df["ExitHeading"],prefix = 'ex')



test_df = pd.concat([test_df,dfcitytest],axis=1)

test_df = pd.concat([test_df,dfent],axis=1)

test_df = pd.concat([test_df,dfext],axis=1)





## Thanks for: https://www.kaggle.com/danofer/baseline-feature-engineering-geotab-69-5-lb

        

directions = {

    'N': 0,

    'NE': 1/4,

    'E': 1/2,

    'SE': 3/4,

    'S': 1,

    'SW': 5/4,

    'W': 3/2,

    'NW': 7/4

}



train_df['EntryHeading'] = train_df['EntryHeading'].map(directions)

train_df['ExitHeading'] = train_df['ExitHeading'].map(directions)



test_df['EntryHeading'] = test_df['EntryHeading'].map(directions)

test_df['ExitHeading'] = test_df['ExitHeading'].map(directions)



# entering and exiting on same street

train_df["same_street_exact"] = (train_df["EntryStreetName"] ==  train_df["ExitStreetName"]).astype(int)

test_df["same_street_exact"] = (test_df["EntryStreetName"] ==  test_df["ExitStreetName"]).astype(int)
train_df.head(5)
X = train_df[["IntersectionId","Hour","Weekend","Month",'en_E', 'en_N', 'en_NE', 'en_NW', 'en_S', 'en_SE', 'en_SW', 'en_W', 'ex_E',

       'ex_N', 'ex_NE', 'ex_NW', 'ex_S', 'ex_SE', 'ex_SW', 'ex_W', 'city_Atlanta', 'city_Boston', 'city_Chicago', 'city_Philadelphia', 'same_street_exact', 'EntryHeading', 'ExitHeading']]

y1 = train_df["TotalTimeStopped_p20"]

y2 = train_df["TotalTimeStopped_p50"]

y3 = train_df["TotalTimeStopped_p80"]

y4 = train_df["DistanceToFirstStop_p20"]

y5 = train_df["DistanceToFirstStop_p50"]

y6 = train_df["DistanceToFirstStop_p80"]
testX = test_df[["IntersectionId","Hour","Weekend","Month",'en_E','en_N', 'en_NE', 'en_NW', 'en_S', 

                 'en_SE', 'en_SW', 'en_W', 'ex_E','ex_N', 'ex_NE', 'ex_NW', 'ex_S', 'ex_SE', 'ex_SW', 

                 'ex_W', 'city_Atlanta', 'city_Boston', 'city_Chicago', 'city_Philadelphia', 'same_street_exact', 'EntryHeading', 'ExitHeading']]
regressor = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.03,

                 max_depth=12,

                 min_child_weight=1.5,

                 n_estimators=500,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42) 
%%time

model_1 = regressor.fit(X, y1)

pred_1=model_1.predict(testX)



model_2 = regressor.fit(X, y2)

pred_2=model_2.predict(testX)





model_3 = regressor.fit(X, y3)

pred_3=model_3.predict(testX)



model_4 = regressor.fit(X, y4)

pred_4=model_1.predict(testX)



model_5 = regressor.fit(X, y5)

pred_5=model_5.predict(testX)



model_6 = regressor.fit(X, y6)

pred_6=model_6.predict(testX)



predictions = []

for i in range(len(pred_1)):

    for j in [pred_1,pred_2,pred_3,pred_4,pred_5,pred_6]:

        predictions.append(j[i])
submission = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

submission["Target"] = predictions

submission.to_csv("submission.csv",index = False)