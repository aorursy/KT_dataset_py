import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

import datetime



#from bokeh.plotting import figure, show, output_notebook

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



plt.style.use("ggplot") # Set plot style

%matplotlib inline
df = pd.read_csv(r"../input/hourly-energy-consumption/PJME_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])



#Date features

df["Hour"] = df["Datetime"].dt.hour

df["Day"] = df["Datetime"].dt.dayofweek

df["Month"] = df["Datetime"].dt.month

df["Year"] = df["Datetime"].dt.year

df["Q"] = df["Datetime"].dt.quarter

df["Dayofyear"] = df["Datetime"].dt.dayofyear

df["Dayofmonth"] = df["Datetime"].dt.day

df["Weekofyear"] = df["Datetime"].dt.weekofyear



df["Drop_me"] = df["Datetime"].dt.strftime("%m-%d")



df.index = df["Datetime"]

df = df.drop(["Datetime"],axis=1)
df.head()
ax, fig = plt.subplots(figsize=(10,5))



plt.hist(df["PJME_MW"])



plt.yticks(alpha=0.75, weight="bold")



plt.xlabel("",alpha=0.75, weight="bold")

plt.ylabel("",alpha=0.75, weight="bold")



plt.title("Consumption distribution", alpha=0.60, weight="bold", fontsize=15, loc="left", pad=10)
#Data prep

Q1 = df[df["Q"]==1]

Q2 = df[df["Q"]==2]

Q3 = df[df["Q"]==3]

Q4 = df[df["Q"]==4]



#Plot

fig,axes = plt.subplots(2,2,figsize=(17,7),sharex=True,sharey=True)



sns.distplot(Q1["PJME_MW"],color="skyblue", ax=axes[0,0]).set_title("Q1 - Consumption")

sns.distplot(Q2["PJME_MW"],color="red", ax=axes[0,1]).set_title("Q2 - Consumption")

sns.distplot(Q3["PJME_MW"],color="green", ax=axes[1,0]).set_title("Q3 - Consumption")

sns.distplot(Q4["PJME_MW"],color="gray", ax=axes[1,1]).set_title("Q4 - Consumption")



del Q1, Q2, Q3, Q4
#Data prep

mean_per_day = df.groupby("Day")["PJME_MW"].agg(["mean"])



#Plot

fig, ax = plt.subplots(figsize=(10,5))





plt.plot(mean_per_day.index,mean_per_day["mean"])



plt.xticks(mean_per_day.index, ["Monday","Tuesday", "Wednesday", "Thursday", "Friday", "Saturday","Sunday"], alpha=0.75, weight="bold")

plt.yticks(alpha=0.75, weight="bold")



plt.xlabel("Day of the week",alpha=0.75, weight="bold")

plt.ylabel("Mean consumption",alpha=0.75, weight="bold")



plt.title("Mean daily consumption", alpha=0.60, weight="bold", fontsize=15, loc="left", pad=10)



#del mean_per_day
#Data

mean_per_hour = df.groupby("Hour")["PJME_MW"].agg(["mean"])



#Plot

fig, ax = plt.subplots(figsize=(10,5))



plt.bar(mean_per_hour.index, mean_per_hour["mean"])



plt.xticks(range(24),alpha=0.75, weight="bold")

plt.yticks(alpha=0.75, weight="bold")



plt.xlabel("Hour",alpha=0.75, weight="bold")

plt.ylabel("Mean consumption",alpha=0.75, weight="bold")



plt.title("Mean hourly consumption", alpha=0.60, weight="bold", fontsize=15, loc="left", pad=10)



del mean_per_hour
fig,ax = plt.subplots(figsize=(15,5))



plt.plot(df.index,df["PJME_MW"])



plt.xlabel("Date", alpha=0.75, weight="bold")

plt.ylabel("Consumption", alpha=0.75, weight="bold")



plt.xticks(alpha=0.75,weight="bold", fontsize=11)

plt.yticks(alpha=0.75,weight="bold", fontsize=11)



plt.title("Consumption over time", alpha=0.75, weight="bold", fontsize=15, pad=10, loc="left")
#Be ready for some bruteforce if functions

def feature_holidays(row):



    if row["Drop_me"] == "01-01":

        return "New Year"

    if row["Drop_me"] == "04-07":

        return "Ind Day"

    if row["Drop_me"] == "11-28":

        return "Thanksgiving"

    if row["Drop_me"] == "12-25":

        return "Christmas"

    return 'Other'

def feature_worktime(row):

    if row["Hour"] > 7 & row["Hour"] <= 17:

        return "Worktime"

    return "NonWorkTime"

def feature_peak(row):

    if row["Hour"] > 7 & row["Hour"] <= 22:

        return "Peak"

    return "NonPeak"

def feature_weekend(row):

    if row["Day"] == 5 or row["Day"] == 6:

        return "Weekend"

    return "NonWeekend"
df["Holiday"] = df.apply (lambda row : feature_holidays(row), axis=1)

df["Work"] = df.apply(lambda row: feature_worktime(row), axis=1)

df["Peak"] = df.apply(lambda row: feature_peak(row), axis=1)

df["Weekend"] = df.apply(lambda row: feature_weekend(row), axis=1)
df = df.drop(["Drop_me"],axis=1)
dummies = pd.get_dummies(df[["Holiday","Peak","Work","Weekend"]],prefix="Dummy")

df = df.join(dummies,lsuffix="_left")

df = df.drop(df[["Holiday","Peak","Work","Weekend"]], axis=1)
train_test_date = "01-01-2015"

consum_test = df[df.index > train_test_date].copy()

consum_train = df[df.index <= train_test_date].copy()
def lag_features(lag_dataset,period_list):



    temp_data = lag_dataset["PJME_MW"]



    for period in period_list:



        lag_dataset["lag_consumption_{}".format(period)] = temp_data.shift(period)

        lag_dataset["mean_rolling_{}".format(period)] = temp_data.rolling(period).mean()

        lag_dataset["max_rolling_{}".format(period)] = temp_data.rolling(period).max()

        lag_dataset["min_rolling_{}".format(period)] = temp_data.rolling(period).min()



    for column in lag_dataset.columns[20:]:



        lag_dataset[column] = lag_dataset[column].fillna(lag_dataset.groupby("Hour")["PJME_MW"].transform("mean"))

    

    return lag_dataset
consum_train = lag_features(consum_train,[7,14,30])



consum_test = lag_features(consum_test,[7,14,30])
fig,ax = plt.subplots(figsize=(15,5))



coloring = 70000



plt.plot(consum_train.index,consum_train["PJME_MW"],color="blue", alpha=0.75)

plt.fill_between(consum_train.index,coloring, facecolor="blue", alpha=0.2)



plt.plot(consum_test.index,consum_test["PJME_MW"], color = "red", alpha=0.60)

plt.fill_between(consum_test.index,coloring, facecolor="red", alpha=0.2)



plt.xlabel("Date", alpha=0.75, weight="bold")

plt.ylabel("Consumption", alpha=0.75, weight="bold")



plt.xticks(alpha=0.75,weight="bold", fontsize=11)

plt.yticks(alpha=0.75,weight="bold", fontsize=11)



plt.title("Train - Test Split", alpha=0.75, weight="bold", fontsize=15, pad=10, loc="left")
#Train - Test

X_train = consum_train.drop("PJME_MW", axis=1) 

y_train = consum_train["PJME_MW"]

X_test = consum_test.drop("PJME_MW", axis=1)

y_test = consum_test["PJME_MW"]
#XGBoost



xgd_reg = xgb.XGBRegressor(n_estimators=1000)



xgd_reg.fit(X_train,y_train,

           eval_set=[(X_train,y_train),(X_test,y_test)],

           early_stopping_rounds=50,

           verbose=False)
plot_importance(xgd_reg)
consum_test["Prediction"] = xgd_reg.predict(X_test)
fig,ax = plt.subplots(figsize=(15,5))



plt.plot(consum_train.index,consum_train["PJME_MW"],alpha=.3)

plt.plot(consum_test.index,consum_test["Prediction"])

plt.plot(consum_test.index,consum_test["PJME_MW"],alpha=.3)



plt.xlabel("Date", alpha=0.75, weight="bold")

plt.ylabel("Consumption", alpha=0.75, weight="bold")



plt.xticks(alpha=0.75,weight="bold", fontsize=11)

plt.yticks(alpha=0.75,weight="bold", fontsize=11)



plt.title("Predicted consumption", alpha=0.75, weight="bold", fontsize=15, pad=10, loc="left")
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(15,10),sharey=True)



ax1.plot(consum_test.index,consum_test["PJME_MW"],alpha=0.75)

ax2.plot(consum_test.index,consum_test["Prediction"],color="blue",alpha=0.50)



ax1.set_xlabel("Date", alpha=0.75, weight="bold")

ax1.set_ylabel("Consumption", alpha=0.75, weight="bold")

ax2.set_xlabel("Date", alpha=0.75, weight="bold")

ax2.set_ylabel("Consumption", alpha=0.75, weight="bold")



ax1.set_title("Actual consumption", alpha=0.75, weight="bold", fontsize=15, pad=10, loc="left")

ax2.set_title("Predicted consumption", alpha=0.75, weight="bold", fontsize=15, pad=10, loc="left")
mean_sq = mean_squared_error(y_test,xgd_reg.predict(X_test))

rmse = np.sqrt(mean_sq)



mean_abs_sq = mean_absolute_error(y_test,xgd_reg.predict(X_test))



print("Root Mean Squared Error : {}".format(rmse))

print("Mean Absolute Error : {}".format(mean_abs_sq))
# Worst Hour Prediction

consum_test["Difference"] = np.abs(consum_test["PJME_MW"] - consum_test["Prediction"])

consum_test["Difference"].sort_values(ascending=False)[:10]
fig,ax= plt.subplots(figsize=(15,5))





ax.plot(consum_test.index,consum_test["PJME_MW"],label="Actual")

ax.plot(consum_test.index,consum_test["Prediction"],alpha=.5,zorder=10,label="Predicted")



consum = consum_test["PJME_MW"]

pred = consum_test["Prediction"]



plt.fill_between(consum_test.index, consum,pred, facecolor="green", alpha=.2,label="Difference")



ax.set_ylim(25000, 45000)

ax.set_xbound(lower="2017-03-12 00:00:00", upper="2017-03-12 23:30:00")



plt.xlabel("Date", alpha=0.75, weight="bold")

plt.ylabel("Consumption", alpha=0.75, weight="bold")



plt.xticks(alpha=0.75,weight="bold", fontsize=11)

plt.yticks(alpha=0.75,weight="bold", fontsize=11)



plt.title("Period with the worst hourly prediction", alpha=0.75, weight="bold", fontsize=15, pad=10, loc="left")

plt.legend()
#Worst Days

worst_days = consum_test.groupby(['Year','Month','Dayofmonth']).mean()[['PJME_MW','Prediction','Difference']]

worst_days.sort_values(by="Difference",ascending=False)[:10]
fig,ax= plt.subplots(figsize=(15,5))





ax.plot(consum_test.index,consum_test["PJME_MW"],label="Actual")

ax.plot(consum_test.index,consum_test["Prediction"],color="red",alpha=.5,zorder=10,label="Predicted")



consum = consum_test["PJME_MW"]

pred = consum_test["Prediction"]



plt.fill_between(consum_test.index, consum,pred, facecolor="green", alpha=.2,label="Difference")



ax.set_ylim(20000, 40000)

ax.set_xbound(lower="2016-02-20", upper="2016-02-25")



plt.xlabel("Date", alpha=0.75, weight="bold")

plt.ylabel("Consumption", alpha=0.75, weight="bold")



plt.xticks(alpha=0.75,weight="bold", fontsize=11)

plt.yticks(alpha=0.75,weight="bold", fontsize=11)



plt.title("Period with the worst daily prediction", alpha=0.75, weight="bold", fontsize=15, pad=10, loc="left")



plt.legend()
#Best Days

worst_days.sort_values(by="Difference",ascending=True)[:10]
fig,ax= plt.subplots(figsize=(15,5))



ax.plot(consum_test.index,consum_test["PJME_MW"], label="Actual")

ax.plot(consum_test.index,consum_test["Prediction"], label="Predicted")

consum = consum_test["PJME_MW"]

pred = consum_test["Prediction"]



plt.fill_between(consum_test.index, consum,pred, facecolor="green", alpha=.2,label="Difference")



ax.set_ylim(20000, 35000)

ax.set_xbound(lower="2015-10-28", upper="2015-10-30")



plt.xlabel("Date", alpha=0.75, weight="bold")

plt.ylabel("Consumption", alpha=0.75, weight="bold")



plt.xticks(alpha=0.75,weight="bold", fontsize=11)

plt.yticks(alpha=0.75,weight="bold", fontsize=11)



plt.title("Period with the best daily prediction", alpha=0.75, weight="bold", fontsize=15, pad=10, loc="left")





plt.legend()