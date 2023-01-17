import os

import pandas as pd

import numpy as np

import json

import matplotlib.pyplot as plt

import datetime as datetime

from datetime import timedelta, date

import seaborn as sns

import matplotlib.cm as CM

import collections

import lightgbm as lgb

from sklearn import preprocessing

from sklearn.tree import DecisionTreeRegressor , plot_tree

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV, train_test_split

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
path = "../input/ga-customer-revenue-prediction/train.csv"

train_data = pd.read_csv(path)

train_data.head()
print (train_data.shape)

print("\n")

print(list(train_data.columns.values))
tmp = train_data.channelGrouping.value_counts()

labels = tmp.index

sizes = tmp.values



fig1, ax1 = plt.subplots(figsize=(8,8))

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops=dict(color="w"))

ax1.legend(labels, loc="center right",bbox_to_anchor=(0.8, 0, 0.5, 1))



plt.show()
snstemp = pd.DataFrame(train_data.channelGrouping.value_counts())

sns.barplot(y = snstemp.index, x = snstemp.iloc[:,0])
print("date :{}, visitStartTime:{}".format(train_data.head(1).date[0],train_data.head(1).visitStartTime[0]))
train_data["date"] = pd.to_datetime(train_data["date"],format="%Y%m%d")

train_data["visitStartTime"] = pd.to_datetime(train_data["visitStartTime"],unit='s')
train_data.head()[["date","visitStartTime"]]
tmp_device_df = pd.DataFrame(train_data.device.apply(json.loads).tolist())
# replace the JSON's "not available in demo dataset" and "(not set)" to NULL

tmp_device_df = tmp_device_df.replace(["not available in demo dataset","(not set)"], pd.NaT)

tmp_device_df.head()
tmp_device_df.nunique()
tmp_device_df = pd.DataFrame(train_data.device.apply(json.loads).tolist())[["browser","operatingSystem","deviceCategory","isMobile"]]

tmp_device_df = tmp_device_df.replace(["not available in demo dataset","(not set)"], pd.NaT)
fig, axes = plt.subplots(2,2,figsize=(16,16))

tmp_device_df["isMobile"].value_counts().plot(kind="barh",ax=axes[0][0],legend="isMobile",color='tan').invert_yaxis()

tmp_device_df["browser"].value_counts().head(10).plot(kind="barh",ax=axes[0][1],legend="browser",color='teal').invert_yaxis()

tmp_device_df["deviceCategory"].value_counts().head(10).plot(kind="barh",ax=axes[1][0],legend="deviceCategory",color='green').invert_yaxis()

tmp_device_df["operatingSystem"].value_counts().head(10).plot(kind="barh",ax=axes[1][1],legend="operatingSystem",color='c').invert_yaxis()
tmp_geo_df = pd.DataFrame(train_data.geoNetwork.apply(json.loads).tolist())
# replace the JSON's "not available in demo dataset" and "(not set)" to NULL

tmp_geo_df = tmp_geo_df.replace(["not available in demo dataset","(not set)"], pd.NaT)

tmp_geo_df.head()
tmp_geo_df.nunique()
tmp_geo_df.isna().sum()
tmp_geo_df = pd.DataFrame(train_data.geoNetwork.apply(json.loads).tolist())[["continent", "subContinent", "country", "city", "metro"]]

tmp_geo_df = tmp_geo_df.replace(["not available in demo dataset","(not set)"], pd.NaT)  

tmp_geo_df.head()
fig, axes = plt.subplots(3,2, figsize=(15,15))

tmp_geo_df["continent"].value_counts().plot(kind="bar",ax=axes[0][0],title="Global Distributions",rot=0,color="c")

tmp_geo_df[tmp_geo_df["continent"] == "Americas"]["subContinent"].value_counts().plot(kind="bar",ax=axes[1][0], title="America Distributions",rot=0,color="tan")

tmp_geo_df[tmp_geo_df["continent"] == "Asia"]["subContinent"].value_counts().plot(kind="bar",ax=axes[0][1], title="Asia Distributions",rot=0,color="r")

tmp_geo_df[tmp_geo_df["continent"] == "Europe"]["subContinent"].value_counts().plot(kind="bar",ax=axes[1][1],  title="Europe Distributions",rot=0,color="green")

tmp_geo_df[tmp_geo_df["continent"] == "Oceania"]["subContinent"].value_counts().plot(kind="bar",ax = axes[2][0], title="Oceania Distributions",rot=0,color="teal")

tmp_geo_df[tmp_geo_df["continent"] == "Africa"]["subContinent"].value_counts().plot(kind="bar" , ax=axes[2][1], title="Africa Distributions",rot=0,color="silver")
train_data["socialEngagementType"].describe()
tmp_totals_df = pd.DataFrame(train_data.totals.apply(json.loads).tolist())
# replace the JSON's "not available in demo dataset" and "(not set)" to NULL

tmp_totals_df = tmp_totals_df.replace(["not available in demo dataset","(not set)"], pd.NaT)
tmp_totals_df.describe()
tmp_totals_df = tmp_totals_df.fillna(0)

tmp_totals_df.describe()
tmp_totals_df = tmp_totals_df.drop("visits", axis = 1)
# Printing some statistics of our data

# Convert the data type from string to float 

tmp_totals_df = tmp_totals_df.astype(float)



print("Transaction Revenue Min Value: ", 

      tmp_totals_df[tmp_totals_df['transactionRevenue'] > 0]["transactionRevenue"].min()) # printing the min value

print("Transaction Revenue Mean Value: ", 

      tmp_totals_df[tmp_totals_df['transactionRevenue'] > 0]["transactionRevenue"].mean()) # mean value

print("Transaction Revenue Median Value: ", 

      tmp_totals_df[tmp_totals_df['transactionRevenue'] > 0]["transactionRevenue"].median()) # median value

print("Transaction Revenue Max Value: ", 

      tmp_totals_df[tmp_totals_df['transactionRevenue'] > 0]["transactionRevenue"].max()) # the max value
plt.figure(figsize=(14,5))



plt.subplot(1,2,1)

tmp = [ len(tmp_totals_df[tmp_totals_df['transactionRevenue'] == 0]),len(tmp_totals_df[tmp_totals_df['transactionRevenue'] > 0])]

lebels = ["Sessions w/o revenue","Sessions w/ revenue"]



plt.pie(tmp, autopct='%1.1f%%', startangle=90, textprops=dict(color="w",fontsize= 14))

plt.legend(lebels,loc="up right",bbox_to_anchor=(0.8, 0, 0.5, 1))



# seting the distribuition of our data and normalizing using np.log on values highest than 0 and + 

# also, we will set the number of bins and if we want or not kde on our histogram

plt.subplot(1,2,2)

ax = sns.distplot(np.log(tmp_totals_df[tmp_totals_df['transactionRevenue'] > 0]["transactionRevenue"]), bins=60, kde=True)

ax.set_xlabel('Transaction RevenueLog', fontsize=15) #seting the xlabel and size of font

ax.set_ylabel('Distribuition', fontsize=15) #seting the ylabel and size of font

ax.set_title("Distribuition of Revenue Log", fontsize=20) #seting the title and size of font



plt.show()
traffic_source_df = pd.DataFrame(train_data.trafficSource.apply(json.loads).tolist())[["keyword","medium" , "source"]]

traffic_source_df.head()
fig,axes = plt.subplots(1,2,figsize=(15,10))

traffic_source_df["medium"].value_counts().plot(kind="barh",ax = axes[0],title="Medium",color="tan").invert_yaxis()

traffic_source_df["source"].value_counts().head(10).plot(kind="barh",ax=axes[1],title="source",color="teal").invert_yaxis()
traffic_source_df.loc[traffic_source_df["source"].str.contains("google") ,"source"] = "google"

fig,axes = plt.subplots(1,1,figsize=(8,8))

traffic_source_df["source"].value_counts().head(15).plot(kind="barh",ax=axes,title="source",color="teal").invert_yaxis()
fig,axes = plt.subplots(1,2,figsize=(16,8))

traffic_source_df["keyword"].value_counts().head(5).plot(kind="barh",ax=axes[0], title="keywords (total)",color="orange").invert_yaxis()

traffic_source_df[traffic_source_df["keyword"] != "(not provided)"]["keyword"].value_counts().head(10).plot(kind="barh",ax=axes[1],title="keywords (dropping NA)",color="c").invert_yaxis()
train_data.visitNumber.describe()
print("90 percent of sessions have visitNumber lower than {} times.".format(np.percentile(list(train_data.visitNumber),90)))
train_data.fullVisitorId.nunique()
# Merge all the useful features back together after the cleaning procedures above.

train_data_cleaned = train_data[["fullVisitorId", "channelGrouping", "date", "visitStartTime","visitNumber"]]

train_data_cleaned = train_data_cleaned.merge(tmp_device_df,left_index=True, right_index=True)

train_data_cleaned = train_data_cleaned.merge(tmp_geo_df,left_index=True, right_index=True)

train_data_cleaned = train_data_cleaned.merge(tmp_totals_df,left_index=True, right_index=True)

train_data_cleaned = train_data_cleaned.merge(traffic_source_df,left_index=True, right_index=True)

train_data_cleaned.head(1)
train_data_rev = train_data_cleaned[train_data_cleaned['transactionRevenue'] > 0]

train_data_rev.head(1)
def plotCategoryRateBar(a, b, colName, topN=np.nan):

    if topN == topN:

        vals = b[colName].value_counts()[:topN]

        subA = a.loc[a[colName].isin(vals.index.values), colName]

        df = pd.DataFrame({'% in Rows with Revenue':subA.value_counts() / len(a) *100, '% in Overall Dataset':vals / len(b)*100})

    else:

        df = pd.DataFrame({'% in Rows with Revenue':a[colName].value_counts() / len(a)*100, '% in Overall Dataset':b[colName].value_counts() / len(b)*100})

    #return the barh plot

    df.sort_values('% in Rows with Revenue').plot.barh(colormap='bwr', title = colName +" and Revenue relationship")
def plotCategoryAvgBar(a, colName,topN=np.nan):

    df = pd.DataFrame()

    

    for item in a[colName].unique():

        mean = np.mean(a[a[colName] == item].transactionRevenue)/100000

        median = np.median(a[a[colName] == item].transactionRevenue)/100000

        total = np.sum(a[a[colName] == item].transactionRevenue)/100000

        temp_df = pd.DataFrame([[mean,median,total]],index = [item],columns=["mean","median","total"])

        df = pd.concat([df,temp_df])

    

    if topN == topN:

        fig,axes=plt.subplots(1,2,figsize=(14,5))

        df.loc[:,"total"].sort_values().tail(topN).plot.barh(ax=axes[0],color="b",title = colName +" and total revenue relationship")

        df.loc[:,["mean","median"]].sort_values(by = "median").tail(topN).plot.barh(ax=axes[1],colormap='PiYG', title = colName +" and Revenue mean/median relationship")

    else:

        fig,axes=plt.subplots(1,2,figsize=(14,5))

        df.loc[:,"total"].sort_values().plot.barh(ax=axes[0],color="b",title = colName +" and total revenue relationship")

        df.loc[:,["mean","median"]].sort_values(by = "median").plot.barh(ax=axes[1],colormap='PiYG', title = colName +" and Revenue mean/median relationship")
plotCategoryRateBar(train_data_rev, train_data_cleaned, "channelGrouping")
plotCategoryAvgBar(train_data_rev, "channelGrouping",5)
plotCategoryRateBar(train_data_rev, train_data_cleaned, "browser",5)
plotCategoryRateBar(train_data_rev, train_data_cleaned, "operatingSystem",5)
plotCategoryAvgBar(train_data_rev, "operatingSystem",5)
plotCategoryRateBar(train_data_rev, train_data_cleaned, "deviceCategory",5)
plotCategoryAvgBar(train_data_rev, "deviceCategory",5)
plotCategoryRateBar(train_data_rev, train_data_cleaned, "continent",5)
plotCategoryRateBar(train_data_rev, train_data_cleaned, "metro",5)
plotCategoryRateBar(train_data_rev, train_data_cleaned, "city",5)
plotCategoryRateBar(train_data_rev, train_data_cleaned, "medium",5)
plotCategoryRateBar(train_data_rev, train_data_cleaned, "source",5)
# list how many days of observations in our dataset

date_list = np.sort(list(set(list(train_data["date"]))))

"first_day:'{}' and last_day:'{}' and toal number of data we have is: '{}' days.".format(date_list[0], date_list[-1],len(set(list(train_data["date"]))))
tmp_churn_df = pd.DataFrame()

tmp_churn_df["date"] = train_data["date"]

tmp_churn_df["year"] = pd.DatetimeIndex(tmp_churn_df["date"]).year

tmp_churn_df["month"] =pd.DatetimeIndex(tmp_churn_df["date"]).month

tmp_churn_df["fullVisitoId"] = train_data["fullVisitorId"]

tmp_churn_df.head()
"distinct users who visited the website on 2016-08 are:'{}'persons".format(len(set(tmp_churn_df[(tmp_churn_df.year == 2016) & (tmp_churn_df.month == 8) ]["fullVisitoId"])))
# so that we could use the same method to save the interval visitors in different months

target_intervals_list = [(2016,8),(2016,9),(2016,10),(2016,11),(2016,12),(2017,1),(2017,2),(2017,3),(2017,4),(2017,5),(2017,6),(2017,7)]

intervals_visitors = []

for tmp_tuple in target_intervals_list: 

    intervals_visitors.append(tmp_churn_df[(tmp_churn_df.year == tmp_tuple[0]) & (tmp_churn_df.month == tmp_tuple[1]) ]["fullVisitoId"])
tmp_matrix = np.zeros((11,11))



for i in range(0,11):

    k = False

    tmp_set = []

    for j in range(i,11): 

        if k:

            tmp_set = tmp_set & set(intervals_visitors[j])

        else:

            tmp_set = set(intervals_visitors[i]) & set(intervals_visitors[j])

        tmp_matrix[i][j] = len(list(tmp_set))

        k = True
xticklabels = ["interval 1","interval 2","interval 3","interval 4","interval 5","interval 6","interval 7","interval 8",

              "interval 9","interval 10","interval 11"]

yticklabels = [(2016,8),(2016,9),(2016,10),(2016,11),(2016,12),(2017,1),(2017,2),(2017,3),(2017,4),(2017,5),(2017,6),(2017,7)]

fig, ax = plt.subplots(figsize=(11,11))

ax = sns.heatmap(np.array(tmp_matrix,dtype=int), annot=True, cmap="coolwarm",xticklabels=xticklabels,fmt="d",yticklabels=yticklabels)

ax.set_title("Churn-rate heatmap")

ax.set_xlabel("intervals")

ax.set_ylabel("months")
train_data_cleaned = train_data_cleaned.drop(['fullVisitorId','date','visitStartTime'],axis=1)
train_data_binaryprediction= train_data_cleaned

train_data_binaryprediction["ifrevenue"] = 0

for i in range(len(train_data_binaryprediction.transactionRevenue)):

    if train_data_binaryprediction.transactionRevenue[i] != 0: train_data_binaryprediction["ifrevenue"][i] = 1



train_data_binaryprediction = train_data_binaryprediction.drop("transactionRevenue", axis = 1)

train_data_binaryprediction.head()
df_train_model1, df_test_model1 = train_test_split(train_data_binaryprediction, test_size=0.2, random_state=42)
categorical_features = ['channelGrouping', 'browser', 'operatingSystem', 'deviceCategory', 'isMobile',

                        'continent', 'subContinent', 'country', 'city','metro', 'keyword', 'medium', 'source']



numerical_features = ['visitNumber', 'newVisits', 'bounces', 'pageviews', 'hits']



for column_iter in categorical_features:

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(df_train_model1[column_iter].values.astype('str')) + list(df_test_model1[column_iter].values.astype('str')))

    df_train_model1[column_iter] = lbl.transform(list(df_train_model1[column_iter].values.astype('str')))

    df_test_model1[column_iter] = lbl.transform(list(df_test_model1[column_iter].values.astype('str')))



for column_iter in numerical_features:

    df_train_model1[column_iter] = df_train_model1[column_iter].astype(np.float)

    df_test_model1[column_iter] = df_test_model1[column_iter].astype(np.float)
params_model1 = {

    "objective": "binary",

    "metric": "binary_logloss",

    "num_leaves": 30,

    "min_child_samples": 100,

    "learning_rate": 0.1,

    "bagging_fraction": 0.7,

    "feature_fraction": 0.5,

    "bagging_frequency": 5,

    "bagging_seed": 2018,

    "verbosity": -1

}

lgb_train_model1 = lgb.Dataset(df_train_model1.loc[:,df_train_model1.columns != "ifrevenue"], np.log1p(df_train_model1.loc[:,"ifrevenue"]))

lgb_eval_model1 = lgb.Dataset(df_test_model1.loc[:,df_test_model1.columns != "ifrevenue"], np.log1p(df_test_model1.loc[:,"ifrevenue"]), reference=lgb_train_model1)

gbm_model1 = lgb.train(params_model1, lgb_train_model1, num_boost_round=2000, valid_sets=[lgb_eval_model1], early_stopping_rounds=100,verbose_eval=100)
lgb.plot_importance(gbm_model1,grid=False,height=0.6)
df_train = train_data_cleaned

df_train = df_train.drop("ifrevenue",axis = 1)
df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42)



df_train["transactionRevenue"] = df_train["transactionRevenue"].astype(np.float)

df_test["transactionRevenue"] = df_test["transactionRevenue"].astype(np.float)

print("We have these columns for our regression problems:\n{}".format(df_train.columns))
df_train.head(1)
categorical_features = ['channelGrouping', 'browser', 'operatingSystem', 'deviceCategory', 'isMobile',

                        'continent', 'subContinent', 'country', 'city','metro', 'keyword', 'medium', 'source']



numerical_features = ['visitNumber', 'newVisits', 'bounces', 'pageviews', 'hits']



for column_iter in categorical_features:

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(df_train[column_iter].values.astype('str')) + list(df_test[column_iter].values.astype('str')))

    df_train[column_iter] = lbl.transform(list(df_train[column_iter].values.astype('str')))

    df_test[column_iter] = lbl.transform(list(df_test[column_iter].values.astype('str')))



for column_iter in numerical_features:

    df_train[column_iter] = df_train[column_iter].astype(np.float)

    df_test[column_iter] = df_test[column_iter].astype(np.float)
params = {

    "objective": "regression",

    "metric": "rmse",

    "num_leaves": 30,

    "min_child_samples": 100,

    "learning_rate": 0.1,

    "bagging_fraction": 0.7,

    "feature_fraction": 0.5,

    "bagging_frequency": 5,

    "bagging_seed": 2018,

    "verbosity": -1

}

lgb_train = lgb.Dataset(df_train.loc[:,df_train.columns != "transactionRevenue"], np.log1p(df_train.loc[:,"transactionRevenue"]))

lgb_eval = lgb.Dataset(df_test.loc[:,df_test.columns != "transactionRevenue"], np.log1p(df_test.loc[:,"transactionRevenue"]), reference=lgb_train)

gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=[lgb_eval], early_stopping_rounds=100,verbose_eval=100)
lgb.plot_importance(gbm,grid=False,height=0.6)