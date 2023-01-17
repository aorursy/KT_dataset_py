# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/train.csv")

test_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/test.csv")
cities = train_df.iloc[:,-1].value_counts()

cities.plot(kind='bar')

plt.title("Count of Entries by City")

plt.ylabel("Number of Entries")

plt.xlabel("City")
hour_dist = train_df['Hour'].value_counts()

hour_dist.sort_index().plot(kind='bar')

plt.title("Count of Entries by Hour")

plt.xlabel("Hour")

plt.ylabel("Number of Entries")
weekend = []

weekdays = []

hours = [n for n in range(0,24)]

for hour in hours:

    weekend_congestion_time = sum(train_df[(train_df['Hour'] == hour) & (train_df['Weekend'] == 1)]['TotalTimeStopped_p20'])

    weekdays_congestion_time = sum(train_df[(train_df['Hour'] == hour) & (train_df['Weekend'] == 0)]['TotalTimeStopped_p20'])

    total_congestion_time = sum(train_df[(train_df['Hour'] == hour)]['TotalTimeStopped_p80'])

    weekend.append(weekend_congestion_time*100/total_congestion_time)

    weekdays.append(weekdays_congestion_time*100/total_congestion_time)

print(len(weekend))



f,ax = plt.subplots(figsize=(12,5))

ax.plot(hours, weekend, label = 'Weekend', alpha=0.8, color='r')

ax.plot(hours, weekdays, label = 'Weekdays', alpha=0.4, color='b')

leg = ax.legend()

plt.ylabel("Percentage")

plt.xlabel("Hour")

plt.title("Congestion Percentage (p20) By Hour (Weekdays vs. Weekend)")
weekend = []

weekdays = []

hours = [n for n in range(0,24)]

for hour in hours:

    weekend_congestion_time = sum(train_df[(train_df['Hour'] == hour) & (train_df['Weekend'] == 1)]['TotalTimeStopped_p60'])

    weekdays_congestion_time = sum(train_df[(train_df['Hour'] == hour) & (train_df['Weekend'] == 0)]['TotalTimeStopped_p60'])

    total_congestion_time = sum(train_df[(train_df['Hour'] == hour)]['TotalTimeStopped_p80'])

    weekend.append(weekend_congestion_time*100/total_congestion_time)

    weekdays.append(weekdays_congestion_time*100/total_congestion_time)

print(len(weekend))



f,ax = plt.subplots(figsize=(12,5))

ax.plot(hours, weekend, label = 'Weekend', alpha=0.8, color='r')

ax.plot(hours, weekdays, label = 'Weekdays', alpha=0.4, color='b')

leg = ax.legend()

plt.ylabel("Percentage")

plt.xlabel("Hour")

plt.title("Congestion Percentage (p60) By Hour (Weekdays vs. Weekend)")
weekend = []

weekdays = []

hours = [n for n in range(0,24)]

for hour in hours:

    weekend_congestion_time = sum(train_df[(train_df['Hour'] == hour) & (train_df['Weekend'] == 1)]['TotalTimeStopped_p80'])

    weekdays_congestion_time = sum(train_df[(train_df['Hour'] == hour) & (train_df['Weekend'] == 0)]['TotalTimeStopped_p80'])

    total_congestion_time = sum(train_df[(train_df['Hour'] == hour)]['TotalTimeStopped_p80'])

    weekend.append(weekend_congestion_time*100/total_congestion_time)

    weekdays.append(weekdays_congestion_time*100/total_congestion_time)

print(len(weekend))



f,ax = plt.subplots(figsize=(12,5))

ax.plot(hours, weekend, label = 'Weekend', alpha=0.8, color='r')

ax.plot(hours, weekdays, label = 'Weekdays', alpha=0.4, color='b')

leg = ax.legend()

plt.ylabel("Percentage")

plt.xlabel("Hour")

plt.title("Congestion Percentage (p80) By Hour (Weekdays vs. Weekend)")
intersections = train_df['IntersectionId'].value_counts()

plt.figure(figsize=(15,6))

intersections[:30].plot(kind='bar')

plt.title("Top 30 Intersection ID")
intersections = train_df['EntryStreetName'].value_counts()

plt.figure(figsize=(15,6))

intersections[:30].plot(kind='bar')

plt.title("Top 30 EntryStreetName")
intersections = train_df['ExitStreetName'].value_counts()

plt.figure(figsize=(15,6))

intersections[:30].plot(kind='bar')

plt.title("Top 30 ExitStreetName")
intersections = train_df['EntryHeading'].value_counts()

plt.figure(figsize=(15,6))

intersections.plot(kind='bar')

plt.title("EntryHeading")

plt.ylabel("Number of Entries")
print('Number of entries in train:',train_df.shape[0]) 

print('Number of entries in test:',test_df.shape[0])
train_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/train.csv")

test_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/test.csv")
#Dummies for train data

dfcity= pd.get_dummies(train_df["City"],prefix = 'city')

dfen = pd.get_dummies(train_df["EntryHeading"],prefix = 'en')

dfex = pd.get_dummies(train_df["ExitHeading"],prefix = 'ex')



train_df = pd.concat([train_df,dfcity],axis=1)

train_df = pd.concat([train_df,dfen],axis=1)

train_df = pd.concat([train_df,dfex],axis=1)



#Dummies for test Data

dfcitytest= pd.get_dummies(test_df["City"],prefix = 'city')

dfent = pd.get_dummies(test_df["EntryHeading"],prefix = 'en')

dfext = pd.get_dummies(test_df["ExitHeading"],prefix = 'ex')



test_df = pd.concat([test_df,dfcitytest],axis=1)

test_df = pd.concat([test_df,dfent],axis=1)

test_df = pd.concat([test_df,dfext],axis=1)



print(test_df.head())
## Label Encoding: directions

directions = {

    "N": 0,

    "NE": 1/4,

    "E": 1/2,

    "SE": 3/4,

    "S": 1,

    "SW": 5/4,

    "W": 3/2,

    "NW": 7/4

}



train_df['EntryHeading'] = train_df['EntryHeading'].map(directions)

train_df['ExitHeading'] = train_df['ExitHeading'].map(directions)



test_df['EntryHeading'] = test_df['EntryHeading'].map(directions)

test_df['ExitHeading'] = test_df['ExitHeading'].map(directions)



train_df['EntryExitSameSt'] = (train_df['EntryStreetName'] == train_df['ExitStreetName']).astype(int)

test_df['EntryExitSameSt'] = (test_df['EntryStreetName'] == test_df['ExitStreetName']).astype(int)

print(test_df.head())
## Label encoding: Street

road_encoding = {

    'Road': 1,

    'Rd': 1,

    'Street': 2,

    'St': 2,

    'Ave': 3,

    'Av': 3,

    'Avenue': 3,

    'Drive': 4,

    'Dr': 4,

    'Boulevard': 5,

    'Blvd': 5

}

def encode(x):

    if pd.isna(x):

        return 0

    for road in road_encoding.keys():

        if road in x:

            return road_encoding[road]

    return 0



train_df['EntryType'] = train_df['EntryStreetName'].apply(encode)

train_df['ExitType'] = train_df['ExitStreetName'].apply(encode)

test_df['EntryType'] = test_df['EntryStreetName'].apply(encode)

test_df['ExitType'] = test_df['ExitStreetName'].apply(encode)





## Label encoding: time of day

def time_of_day(x):

    if x < 8:

         return "midnight"

    elif x < 12:

         return "morning"

    elif x < 16:

         return "afternoon"

    elif x < 19:

         return "evening"

    else:

         return "midnight"



print(train_df['Hour'])

train_df['TimeCategory'] = train_df['Hour'].apply(time_of_day) 

test_df['TimeCategory'] = test_df['Hour'].apply(time_of_day)
df_train_time = pd.get_dummies(train_df["TimeCategory"],prefix = 'time')

df_test_time = pd.get_dummies(test_df["TimeCategory"],prefix = 'time')

train_df = pd.concat([train_df,df_train_time],axis=1)

test_df = pd.concat([test_df,df_test_time],axis=1)

print(train_df.columns)
#x = train_df[["IntersectionId", "Hour", "Weekend",

#        "Month","city_Atlanta", "city_Boston", "city_Chicago", "city_Philadelphia",

#        "en_E", "en_N", "en_NE", "en_NW", "en_S", "en_SE", "en_SW", "en_W",

#        "ex_E", "ex_N", "ex_NE", "ex_NW", "ex_S", "ex_SE", "ex_SW", "ex_W", "EntryExitSameSt"]]

x = train_df[["Weekend", "time_afternoon", "time_midnight", "time_evening", "Hour", "en_SE", "EntryType", "ex_SE", "ex_W", "EntryExitSameSt"]]

y1 = train_df[['TotalTimeStopped_p20']]

y2 = train_df[['TotalTimeStopped_p50']]

y3 = train_df[['TotalTimeStopped_p80']]

y7 = train_df[['DistanceToFirstStop_p20']]

y8 = train_df[['DistanceToFirstStop_p50']]

y9 = train_df[['DistanceToFirstStop_p80']]
testX = test_df[["Weekend", "time_afternoon", "time_midnight", "time_evening", "Hour", "en_SE", "EntryType", "ex_SE", "ex_W", "EntryExitSameSt"]]
import xgboost

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
model_1 = regressor.fit(x, y1)

pred_1=model_1.predict(testX)



model_2 = regressor.fit(x, y2)

pred_2=model_2.predict(testX)



model_3 = regressor.fit(x, y3)

pred_3=model_3.predict(testX)



model_7 = regressor.fit(x, y7)

pred_7=model_7.predict(testX)



model_8 = regressor.fit(x, y8)

pred_8=model_8.predict(testX)



model_9 = regressor.fit(x, y9)

pred_9=model_9.predict(testX)
## collect the results into predictions list

predictions = []

for i in range(len(pred_1)):

    for j in [pred_1,pred_2,pred_3,pred_7,pred_8,pred_9]:

        predictions.append(j[i])
print(len(predictions))

print(len(submission))
## write results to csv file as output

submission = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

submission["Target"] = predictions

submission.to_csv("submission.csv",index = False)
import xgboost

parameters = {

              'learning_rate': .03, #so called `eta` value

              'max_depth': 12,

              'min_child_weight': 1.5,

              'subsample': 0.6,

              'colsample_bytree': 0.4,

              'n_estimators': 500,

              'seed': 42,

              'reg_alpha': 0.75,

              'reg_lambda': 0.45,

              'gamma': 0}

x = train_df[["Weekend", "time_afternoon", "time_midnight", "time_evening", "Hour", "en_SE", "EntryType", "ex_SE", "ex_W", "EntryExitSameSt"]]

y1 = train_df[['TotalTimeStopped_p20']]

y2 = train_df[['TotalTimeStopped_p50']]

y3 = train_df[['TotalTimeStopped_p80']]

y7 = train_df[['DistanceToFirstStop_p20']]

y8 = train_df[['DistanceToFirstStop_p50']]

y9 = train_df[['DistanceToFirstStop_p80']]



## Change the label to y1...y9 to see corresponding graph

data_dmatrix = xgboost.DMatrix(data=x,label=y1)

xg_reg = xgboost.train(params=parameters, dtrain=data_dmatrix, num_boost_round=10)

xgboost.plot_importance(xg_reg)

plt.rcParams['figure.figsize'] = [5, 5]

plt.show()