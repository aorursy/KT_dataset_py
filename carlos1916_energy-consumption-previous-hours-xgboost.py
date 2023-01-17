# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import seaborn as sns

import matplotlib.pyplot as plt

from pprint import pprint

from sklearn import metrics

import shap

from xgboost import XGBRegressor

import time



# Any results you write to the current directory are saved as output.
def add_date_data(df,col):

    df["Weekday"] = df[col].dt.dayofweek  

    df["Week_of_year"] = df[col].dt.week      

    df["Month"] = df[col].dt.month            

    df["Hour"] = df[col].dt.hour              

    df["Year"] = df[col].dt.year              

    df["Day"] = df[col].dt.day   

    df['dayofyear'] = df[col].dt.dayofyear

    

    return df
def add_consumption(df, num):

    last_hours = []



    for row in df["DAYTON_MW"]:

        last_hours.append(row)

    

    for i in range(num):

    

        last_hours.insert(i, 0)

        del last_hours[-1]

        df["Last_" +str(i+1)+ "_hour(s)" ] = last_hours



    return df
def former_weeks(df, num):

    last_hour = df.Lastweek.tolist()

    

    for i in range(num):

    

        last_hour.insert(i, 0)

        del last_hour[-1]

        df["LastWeek_Minus_" +str(i+1)+ "_hours" ] = last_hour



    return df
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
consumption = pd.read_csv("../input/DAYTON_hourly.csv", parse_dates=[0])

consumption.head()

add_date_data(consumption,"Datetime")

consumption.head()
days = 3

consumption = add_consumption(consumption, days)

consumption.head()
Lastweek = []

consumption_list = consumption["DAYTON_MW"].tolist()

for i in range(168):

    Lastweek.append(consumption_list[i])

for i in range(168,len(consumption_list)):

    Lastweek.append(consumption_list[i-168])

    
consumption["Lastweek"] = Lastweek

consumption.head()
hours = 3

consumption = former_weeks(consumption, hours)

consumption.head()
consumption["Delta1m2"] = consumption["Last_1_hour(s)"] - consumption["Last_2_hour(s)"]

consumption.Delta1m2.iloc[1]= 0

consumption["Delta1W_m1"] = consumption.Lastweek - consumption.LastWeek_Minus_1_hours

consumption.Delta1W_m1.iloc[1]= 0

consumption.Delta1W_m1.iloc[0]= 0

consumption["Delta1W_1m2"] = consumption.LastWeek_Minus_1_hours - consumption.LastWeek_Minus_2_hours

consumption.Delta1W_1m2.iloc[1]= 0

consumption["Delta1W_2m3"] = consumption.LastWeek_Minus_2_hours - consumption.LastWeek_Minus_3_hours

consumption.Delta1W_2m3.iloc[1]= 0

consumption.Delta1W_2m3.iloc[2]= 0
consumption.head()
consumption.DAYTON_MW.plot()

plt.title("Historic Consumption")

plt.xlabel("Time")

plt.ylabel("Consumption in Mwh")

plt.show

sns.distplot(consumption.DAYTON_MW)
Month = consumption.groupby(consumption["Month"]).mean()

sns.barplot(Month.index, Month.DAYTON_MW)
weekday = consumption.groupby(consumption["Weekday"]).mean()

sns.barplot(weekday.index, weekday.DAYTON_MW)
weekofyear = consumption.groupby(consumption["Week_of_year"]).mean()

sns.barplot(weekofyear.index, weekofyear.DAYTON_MW)
years = consumption.groupby(consumption["Year"]).mean()

sns.barplot(years.index, years.DAYTON_MW)
consumption = consumption.set_index("Datetime")
training_variables = consumption.columns.tolist()  

objective_variable = training_variables[0]

del training_variables[0]

X = consumption[training_variables]

y = consumption[objective_variable]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, shuffle=False)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size =0.5, shuffle=False)



print(X_train.shape, y_train.shape)

print(X_val.shape, y_val.shape)

print(X_test.shape, y_test.shape)
MAPE_scoreDict = {}

MAE_scoreDict={}

r2_scoreDict = {}

median_r2_scoreDict = {}

mse_scoreDict = {}

evs_scoreDict = {}



learning_rate = [1,.5, 0.3,.01]

max_depth = [1,3,5,6,8,10]

best_score = 100000000000000

best_params = {'Max_D': None, 'Learning_rate': None}



for m in max_depth:

    MAPE_scoreDict[m] = {}

    MAE_scoreDict[m] = {}

    r2_scoreDict[m] = {}

    mse_scoreDict[m] = {}

    median_r2_scoreDict[m] = {}

    evs_scoreDict[m] = {}

    for l in learning_rate:

        t0=time.time()

        clf = XGBRegressor(max_depth=m, n_jobs=-1, learning_rate=l, n_estimators=12000)

        clf.fit(X_train, y_train, verbose=False, early_stopping_rounds=50, eval_set=[(X_train,y_train),(X_val,y_val)])

        

        tf=time.time()

        total_time=tf-t0

        score = mean_absolute_percentage_error(y_val, clf.predict(X_val))

        trees = clf.best_ntree_limit

        

        print("%d, %.03f, MAPE: %.05f, Number of trees: %d, Training Time: %.05f  " % (m, l, score, trees, total_time ))

        

        

        

        

        

        MAPE_scoreDict[m][l] = score

        MAE_scoreDict[m][l] = metrics.mean_absolute_error(y_val, clf.predict(X_val))

        r2_scoreDict[m][l] = metrics.r2_score(y_val, clf.predict(X_val))

        median_r2_scoreDict[m][l] = metrics.median_absolute_error(y_val, clf.predict(X_val))

        mse_scoreDict[m][l] = metrics.mean_squared_error(y_val, clf.predict(X_val))

        evs_scoreDict[m][l] = metrics.explained_variance_score(y_val, clf.predict(X_val))

        

        if score < best_score:

            best_score = score

            best_params['Max_D'] = m

            best_params['Learning_rate'] = l



        evs_scoreDict[m][l] = metrics.explained_variance_score(y_val, clf.predict(X_val))
Mape_score =pd.DataFrame(MAPE_scoreDict)

mse_score2 = pd.DataFrame(mse_scoreDict)

mae_score2 = pd.DataFrame(MAE_scoreDict)

r2_score2 = pd.DataFrame(r2_scoreDict)

median2 = pd.DataFrame(median_r2_scoreDict) 

evs = pd.DataFrame(evs_scoreDict)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))



plt.subplot(231)

for l in learning_rate:

    plt.plot(max_depth, median2.loc[l], label=l)

plt.legend()

plt.xlabel("Max Depth")

plt.ylabel("MED")

plt.title("XGB Regressor MED")



plt.subplot(232)

for l in learning_rate:

    plt.plot(max_depth, mse_score2.loc[l], label=l)

plt.legend()

plt.xlabel("Max Depth")

plt.ylabel("MSE")

plt.title("XGB Regressor MSE")



plt.subplot(233)

for l in learning_rate:

    plt.plot(max_depth, mae_score2.loc[l], label=l)

plt.legend()

plt.xlabel("Max Depth")

plt.ylabel("MAE")

plt.title("XGB Regressor MAE")



plt.subplot(234)

for l in learning_rate:

    plt.plot(max_depth, r2_score2.loc[l], label=l)

plt.legend()

plt.xlabel("Max Depth")

plt.ylabel("R2")

plt.title("XGB Regressor R2")



plt.subplot(235)

for l in learning_rate:

    plt.plot(max_depth, evs.loc[l], label=l)

plt.legend()

plt.xlabel("Max Depth")

plt.ylabel("EVS")

plt.title("XGB Regressor EVS")



plt.subplot(236)

for l in learning_rate:

    plt.plot(max_depth, Mape_score.loc[l], label=l)

plt.legend()

plt.xlabel("Max Depth")

plt.ylabel("MAPE")

plt.title("XGB Regressor MAPE")
t0=time.time()

new_model = XGBRegressor(max_depth=best_params['Max_D'], n_jobs=-1, learning_rate=best_params['Learning_rate'], n_estimators=12000)

new_model.fit(X_train, y_train, verbose=1000, early_stopping_rounds=50, eval_set=[(X_train,y_train),(X_val,y_val)])



tf=time.time()

total_time=tf-t0

MAE_scoreDict1 = metrics.mean_absolute_error(y_val, new_model.predict(X_val))

score = mean_absolute_percentage_error(y_val, new_model.predict(X_val))

median_r2_score1 = metrics.median_absolute_error(y_val, new_model.predict(X_val))

mse_score1 = metrics.mean_squared_error(y_val, new_model.predict(X_val))

evs_score1 = metrics.explained_variance_score(y_val, new_model.predict(X_val))



trees = new_model.best_ntree_limit

print("Number of trees: {} ; MAPE: {}, training time: {}," .format(trees, score, total_time))
metrics.mean_absolute_error(y_test, new_model.predict(X_test))
mean_absolute_percentage_error(y_test,new_model.predict(X_test))
explainer = shap.TreeExplainer(new_model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)
Dates = pd.DataFrame(y_test) 

Dates["pred"] = new_model.predict(X_test)

Dates["MAPE"] = (abs(Dates.DAYTON_MW - Dates.pred)/Dates.DAYTON_MW)*100
Dates.sort_values("MAPE", ascending=False).head(20)
plt.figure(figsize=[12.4, 8.8])

sns.lineplot(Dates.index, Dates.DAYTON_MW, data=Dates, label="Real Consumption")

sns.lineplot(Dates.index, Dates.pred, data=Dates, label="Predicted")

plt.legend()

plt.title("Prediction vs Real consumption")

plt.show()
sns.distplot(Dates.MAPE)