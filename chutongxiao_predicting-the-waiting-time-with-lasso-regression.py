# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/train.csv")

test_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/test.csv")
# Data preprocessing

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

#road size

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

    'Broad': 4,

    'Boulevard': 6,

    'Blvd': 6

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



def turn_direction(x):

    if x < -1.6: 

        return "straight"

    elif x < -0.4:

        return "right"

    elif x < 0.4:

        return "straight"

    elif x < 1.6:

        return "left"

    else:

        return "straight"



train_df['EntryHeading'] = train_df['EntryHeading'].map(directions)

train_df['ExitHeading'] = train_df['ExitHeading'].map(directions)

train_df['diffHeading'] = train_df['EntryHeading']-train_df['ExitHeading']  

train_df['TurnDirection'] = train_df['diffHeading'].apply(turn_direction)  

test_df['EntryHeading'] = test_df['EntryHeading'].map(directions)

test_df['ExitHeading'] = test_df['ExitHeading'].map(directions)

test_df['diffHeading'] = test_df['EntryHeading']-train_df['ExitHeading']  

test_df['TurnDirection'] = test_df['diffHeading'].apply(turn_direction)  



df_train_turn = pd.get_dummies(train_df["TurnDirection"],prefix = 'turn')

df_test_turn = pd.get_dummies(test_df["TurnDirection"],prefix = 'turn')

train_df = pd.concat([train_df,df_train_turn],axis=1)

test_df = pd.concat([test_df,df_test_turn],axis=1)

print(df_train_turn)



#time of day

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



#is vacation

def is_vacation(x):

    if x == 1:

         return 1

    elif x < 6:

         return 0

    elif x < 9:

         return 1

    elif x < 12:

         return 0

    else:

         return 1



train_df['IsVacation'] = train_df['Month'].apply(is_vacation)  

test_df['IsVacation'] = test_df['Month'].apply(is_vacation) 



#create input for regressor tree
stop_20 = train_df['TotalTimeStopped_p20']

stop_40 = train_df['TotalTimeStopped_p40']

stop_50 = train_df['TotalTimeStopped_p50']

stop_60 = train_df['TotalTimeStopped_p60']

stop_80 = train_df['TotalTimeStopped_p80']

dist_20 = train_df['DistanceToFirstStop_p20']

dist_40 = train_df['DistanceToFirstStop_p40']

dist_50 = train_df['DistanceToFirstStop_p50']

dist_60 = train_df['DistanceToFirstStop_p60']

dist_80 = train_df['DistanceToFirstStop_p80']



features_list = ["IntersectionId", "Hour", "Weekend",

       "Month","city_Atlanta", "city_Boston", "city_Chicago", "city_Philadelphia",

       "en_E", "en_N", "en_NE", "en_NW", "en_S", "en_SE", "en_SW", "en_W",

       "ex_E", "ex_N", "ex_NE", "ex_NW", "ex_S", "ex_SE", "ex_SW", "ex_W", 

       "EntryExitSameSt",'EntryType', 'ExitType', 'time_morning', 'time_afternoon', 

                 'time_evening', 'time_midnight', 'IsVacation']



train_data_df = train_df[features_list]

test_data_df = test_df[features_list]
print(train_data_df.columns.values)
# Univariate feature selection 

# preprocess the estimator

Kb = SelectKBest(f_classif, k=10)

Kb.fit(train_data_df, stop_20)

Zip = zip(Kb.get_support(), features_list)

set_zip = set(Zip)

train_np = np.zeros((train_data_df.shape[0], 10))

test_np = np.zeros((test_data_df.shape[0], 10))

index = 0

for pair in set_zip:

    if(pair[0] == True):

        print(pair[1])

        train_np[:,index] = train_data_df[pair[1]]

        test_np[:,index] = test_data_df[pair[1]]

        index+=1

print(set_zip)

#ex_S, city_Atlanta, ex_SE, city_Philadelphia, Weekend, en_SE. ex_W, EntryExitSameSt, city_Boston, Hour
# Lasso Regression

esNet = ElasticNet()

lasso = Lasso()

parameters=[{'alpha':np.logspace(-10, 2, 20)}]

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=10)

lasso_regressor.fit(train_np,stop_20)



print(lasso_regressor.best_params_)

print(lasso_regressor.best_score_)



# Edit negative MSE scores

allscores = lasso_regressor.cv_results_['mean_test_score']*(-1)

alpha = np.logspace(-10, 2, 20)
plt.plot(alpha, -allscores)

plt.xlabel('lambda')

plt.ylabel('mean_test_score')

plt.show()
predict_stop_20 = lasso_regressor.predict(test_np)

plt.hist(predict_stop_20, bins = range(stop_20.min(),stop_20.max(),3), range=(stop_20.min(),stop_20.max()))

plt.title("Prediction on Total_time_stopped_20th")

plt.xlabel("time_stopped")

plt.ylabel("count")

plt.show()
# Lasso Regression

parameters=[{'alpha':np.logspace(-10, 2, 20)}]

esNet_regressor = GridSearchCV(esNet, parameters, scoring='neg_mean_squared_error', cv=10)

esNet_regressor.fit(train_np,stop_50)



print(esNet_regressor.best_params_)

print(esNet_regressor.best_score_)



# Edit negative MSE scores

allscores=esNet_regressor.cv_results_['mean_test_score']*(-1)

alpha=np.logspace(-10, 2, 20)

plt.plot(alpha, allscores)

plt.xlabel('lambda')

plt.ylabel('mean_test_score')

plt.show()
predict_stop_50 = esNet_regressor.predict(test_np)

plt.hist(predict_stop_50, bins = range(stop_50.min(),stop_50.max(),3), range=(stop_50.min(),stop_50.max()))

plt.title("Prediction on Total_time_stopped_50th")

plt.xlabel("time_stopped")

plt.ylabel("count")

plt.show()
parameters=[{'alpha':[0.0000001,0.000001,.00001, .0001, .001, .01, .1, .5]}]

esNet_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=10)

esNet_regressor.fit(train_np,stop_80)



print(esNet_regressor.best_params_)

print(esNet_regressor.best_score_)



# Edit negative MSE scores

allscores=esNet_regressor.cv_results_['mean_test_score']*(-1)

alpha=[0.0000001,0.000001,.00001, .0001, .001, .01, .1, .5]

plt.plot(alpha, allscores)

plt.xlabel('lambda')

plt.ylabel('mean_test_score')

plt.show()
predict_stop_80 = esNet_regressor.predict(test_np)

plt.hist(predict_stop_80, bins = range(stop_80.min(),stop_80.max(),3), range=(stop_80.min(),stop_80.max()))

plt.title("Prediction on Total_time_stopped_80th")

plt.xlabel("time_stopped")

plt.ylabel("count")

plt.show()
# Lasso Regression

parameters=[{'alpha':np.logspace(-10, 2, 20)}]

esNet_regressor = GridSearchCV(esNet, parameters, scoring='neg_mean_squared_error', cv=10)

esNet_regressor.fit(train_np,dist_20)



print(esNet_regressor.best_params_)

print(esNet_regressor.best_score_)



# Edit negative MSE scores

allscores=esNet_regressor.cv_results_['mean_test_score']*(-1)

alpha=np.logspace(-10, 2, 20)

plt.plot(alpha, allscores)

plt.xlabel('lambda')

plt.ylabel('mean_test_score')

plt.show()
predict_dist_20 = esNet_regressor.predict(test_np)

plt.hist(predict_dist_20, bins = range(int(dist_20.min()),int(dist_20.max()),3), range=(int(dist_20.min()),int(dist_20.max())))

plt.title("Prediction on Distance_to_first_stop_p20")

plt.xlabel("distance")

plt.ylabel("count")

plt.show()
# Lasso Regression

esNet = ElasticNet()

parameters=[{'alpha': np.logspace(-10, 2, 20)}]

esNet_regressor = GridSearchCV(esNet, parameters, scoring='neg_mean_squared_error', cv=10)

esNet_regressor.fit(train_np,dist_50)



print(esNet_regressor.best_params_)

print(esNet_regressor.best_score_)



# Edit negative MSE scores

allscores=esNet_regressor.cv_results_['mean_test_score']*(-1)

alpha= np.logspace(-10, 2, 20)

plt.plot(alpha, -allscores)

plt.xlabel('lambda')

plt.ylabel('mean_test_score')

plt.show()
predict_dist_50 = esNet_regressor.predict(test_np)

plt.hist(predict_dist_50, bins = range(int(dist_50.min()),int(dist_50.max()),3), range=(int(dist_50.min()),int(dist_50.max())))

plt.title("Prediction on Distance_to_first_stop_p50")

plt.xlabel("distance")

plt.ylabel("count")

plt.show()
# Lasso Regression

parameters=[{'alpha':np.logspace(-10, 2, 20)}]

esNet_regressor = GridSearchCV(esNet, parameters, scoring='neg_mean_squared_error', cv=10)

esNet_regressor.fit(train_np,dist_80)



print(esNet_regressor.best_params_)

print(esNet_regressor.best_score_)



# Edit negative MSE scores

allscores=esNet_regressor.cv_results_['mean_test_score']*(-1)

alpha=np.logspace(-10, 2, 20)

plt.plot(alpha, allscores)

plt.xlabel('lambda')

plt.ylabel('mean_test_score')

plt.show()
predict_dist_80 = esNet_regressor.predict(test_np)

plt.hist(predict_dist_80, bins = range(int(dist_80.min()),int(dist_80.max()),3), range=(int(dist_80.min()),int(dist_80.max())))

plt.title("Prediction on Distance_to_first_stop_p80")

plt.xlabel("distance")

plt.ylabel("count")

plt.show()
predictions = []

for i in range(len(predict_stop_20)):

    for j in [predict_stop_20,predict_stop_50,predict_stop_80,predict_dist_20,predict_dist_50,predict_dist_80]:

        predictions.append(j[i])

submission = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

submission["Target"] = predictions

submission.to_csv("submission.csv",index = False)