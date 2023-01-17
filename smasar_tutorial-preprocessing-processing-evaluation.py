import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import timedelta, date
import seaborn as sns
import matplotlib.cm as CM
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
%matplotlib inline
train_data = pd.read_csv("../input/train_v2.csv",nrows=700000)
train_data.head()
train_data.describe()
list(train_data.columns.values)
train_data.channelGrouping.value_counts().plot(kind="bar",title="channelGrouping distro",figsize=(8,8),rot=25,colormap='Paired')
"date :{}, visitStartTime:{}".format(train_data.head(1).date[0],train_data.head(1).visitStartTime[0])
train_data["date"] = pd.to_datetime(train_data["date"],format="%Y%m%d")
train_data["visitStartTime"] = pd.to_datetime(train_data["visitStartTime"],unit='s')
train_data.head(1)[["date","visitStartTime"]]
list_of_devices = train_data.device.apply(json.loads).tolist()
keys = []
for devices_iter in list_of_devices:
    for list_element in list(devices_iter.keys()):
        if list_element not in keys:
            keys.append(list_element)
"keys existed in device attribute are:{}".format(keys)
tmp_device_df = pd.DataFrame(train_data.device.apply(json.loads).tolist())[["browser","operatingSystem","deviceCategory","isMobile"]]
tmp_device_df.head()
tmp_device_df.describe()
fig, axes = plt.subplots(2,2,figsize=(15,15))
tmp_device_df["isMobile"].value_counts().plot(kind="bar",ax=axes[0][0],rot=25,legend="isMobile",color='tan')
tmp_device_df["browser"].value_counts().head(10).plot(kind="bar",ax=axes[0][1],rot=40,legend="browser",color='teal')
tmp_device_df["deviceCategory"].value_counts().head(10).plot(kind="bar",ax=axes[1][0],rot=25,legend="deviceCategory",color='lime')
tmp_device_df["operatingSystem"].value_counts().head(10).plot(kind="bar",ax=axes[1][1],rot=80,legend="operatingSystem",color='c')
tmp_geo_df = pd.DataFrame(train_data.geoNetwork.apply(json.loads).tolist())[["continent","subContinent","country","city"]]
tmp_geo_df.head()
tmp_geo_df.describe()
fig, axes = plt.subplots(3,2, figsize=(15,15))
tmp_geo_df["continent"].value_counts().plot(kind="bar",ax=axes[0][0],title="Global Distributions",rot=0,color="c")
tmp_geo_df[tmp_geo_df["continent"] == "Americas"]["subContinent"].value_counts().plot(kind="bar",ax=axes[1][0], title="America Distro",rot=0,color="tan")
tmp_geo_df[tmp_geo_df["continent"] == "Asia"]["subContinent"].value_counts().plot(kind="bar",ax=axes[0][1], title="Asia Distro",rot=0,color="r")
tmp_geo_df[tmp_geo_df["continent"] == "Europe"]["subContinent"].value_counts().plot(kind="bar",ax=axes[1][1],  title="Europe Distro",rot=0,color="lime")
tmp_geo_df[tmp_geo_df["continent"] == "Oceania"]["subContinent"].value_counts().plot(kind="bar",ax = axes[2][0], title="Oceania Distro",rot=0,color="teal")
tmp_geo_df[tmp_geo_df["continent"] == "Africa"]["subContinent"].value_counts().plot(kind="bar" , ax=axes[2][1], title="Africa Distro",rot=0,color="silver")
train_data["socialEngagementType"].describe()
train_data.head()
train_data["revenue"] = pd.DataFrame(train_data.totals.apply(json.loads).tolist())[["transactionRevenue"]]

revenue_datetime_df = train_data[["revenue" , "date"]].dropna()
revenue_datetime_df["revenue"] = revenue_datetime_df.revenue.astype(np.int64)
revenue_datetime_df.head()
daily_revenue_df = revenue_datetime_df.groupby(by=["date"],axis = 0 ).sum()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(figsize=(20,10))
axes.set_title("Daily Revenue")
axes.set_ylabel("Revenue")
axes.set_xlabel("date")
axes.plot(daily_revenue_df["revenue"])

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
axes.set_title("Daily revenue Violin")
axes.set_ylabel("revenue")
axes.violinplot(list(daily_revenue_df["revenue"].values),showmeans=False,showmedians=True)
visit_datetime_df = train_data[["date","visitNumber"]]
visit_datetime_df["visitNumber"] = visit_datetime_df.visitNumber.astype(np.int64)
daily_visit_df = visit_datetime_df.groupby(by=["date"], axis = 0).sum()

fig, axes = plt.subplots(1,1,figsize=(20,10))
axes.set_ylabel("# of visits")
axes.set_xlabel("date")
axes.set_title("Daily Visits")
axes.plot(daily_visit_df["visitNumber"])
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
axes.set_title("Daily visits Violin")
axes.set_ylabel("# of visitors")
axes.violinplot(list(daily_visit_df["visitNumber"].values),showmeans=False,showmedians=True)
train_data.visitNumber.describe()
"90 percent of sessions have visitNumber lower than {} times.".format(np.percentile(list(train_data.visitNumber),90))
import collections

tmp_least_10_visitNumbers_list = collections.Counter(list(train_data.visitNumber)).most_common()[:-10-1:-1]
tmp_most_10_visitNumbers_list = collections.Counter(list(train_data.visitNumber)).most_common(10)
least_visitNumbers = []
most_visitNumbers = []
for i in tmp_least_10_visitNumbers_list:
    least_visitNumbers.append(i[0])
for i in tmp_most_10_visitNumbers_list:
    most_visitNumbers.append(i[0])
"10 most_common visitNumbers are {} times and 10 least_common visitNumbers are {} times".format(most_visitNumbers,least_visitNumbers)
fig,ax = plt.subplots(1,1,figsize=(9,5))
ax.set_title("Histogram of log(visitNumbers) \n don't forget it is per session")
ax.set_ylabel("Repetition")
ax.set_xlabel("Log(visitNumber)")
ax.grid(color='b', linestyle='-', linewidth=0.1)
ax.hist(np.log(train_data.visitNumber))
traffic_source_df = pd.DataFrame(train_data.trafficSource.apply(json.loads).tolist())[["keyword","medium" , "source"]]
fig,axes = plt.subplots(1,2,figsize=(15,10))
traffic_source_df["medium"].value_counts().plot(kind="bar",ax = axes[0],title="Medium",rot=0,color="tan")
traffic_source_df["source"].value_counts().head(10).plot(kind="bar",ax=axes[1],title="source",rot=75,color="teal")
traffic_source_df.loc[traffic_source_df["source"].str.contains("google") ,"source"] = "google"
fig,axes = plt.subplots(1,1,figsize=(8,8))
traffic_source_df["source"].value_counts().head(15).plot(kind="bar",ax=axes,title="source",rot=75,color="teal")
fig,axes = plt.subplots(1,2,figsize=(15,10))
traffic_source_df["keyword"].value_counts().head(10).plot(kind="bar",ax=axes[0], title="keywords (total)",color="orange")
traffic_source_df[traffic_source_df["keyword"] != "(not provided)"]["keyword"].value_counts().head(15).plot(kind="bar",ax=axes[1],title="keywords (dropping NA)",color="c")
repetitive_users = list(np.sort(list(collections.Counter(list(train_data["fullVisitorId"])).values())))
"25% percentile: {}, 50% percentile: {}, 75% percentile: {}, 88% percentile: {}, 88% percentile: {}".format(
np.percentile(repetitive_users,q=25),np.percentile(repetitive_users,q=50),
np.percentile(repetitive_users,q=75),np.percentile(repetitive_users,q=88), np.percentile(repetitive_users,q=89))
date_list = np.sort(list(set(list(train_data["date"]))))
"first_day:'{}' and last_day:'{}' and toal number of data we have is: '{}' days.".format(date_list[0], date_list[-1],len(set(list(train_data["date"]))))
month = 8
start_date = datetime.date(2016, month, 1)
end_date = datetime.date(2017, month, 1)
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
dates_month = []
for single_date in daterange(start_date, end_date):
    dates_month.append(single_date.strftime("%Y-%m"))
dates_month = list(set(dates_month))
dates_month
tmp_churn_df = pd.DataFrame()
tmp_churn_df["date"] = train_data["date"]
tmp_churn_df["yaer"] = pd.DatetimeIndex(tmp_churn_df["date"]).year
tmp_churn_df["month"] =pd.DatetimeIndex(tmp_churn_df["date"]).month
tmp_churn_df["fullVisitoId"] = train_data["fullVisitorId"]
tmp_churn_df.head()
"distinct users who visited the website on 2016-08 are:'{}'persons".format(len(set(tmp_churn_df[(tmp_churn_df.yaer == 2016) & (tmp_churn_df.month == 8) ]["fullVisitoId"])))
target_intervals_list = [(2016,8),(2016,9),(2016,10),(2016,11),(2016,12),(2017,1),(2017,2),(2017,3),(2017,4),(2017,5),(2017,6),(2017,7)]
intervals_visitors = []
for tmp_tuple in target_intervals_list:
    intervals_visitors.append(tmp_churn_df[(tmp_churn_df.yaer == tmp_tuple[0]) & (tmp_churn_df.month == tmp_tuple[1]) ]["fullVisitoId"])
"Size of intervals_visitors:{} ".format(len(intervals_visitors))
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
ax = sns.heatmap(np.array(tmp_matrix,dtype=int), annot=True, cmap="RdBu_r",xticklabels=xticklabels,fmt="d",yticklabels=yticklabels)
ax.set_title("Churn-rate heatmap")
ax.set_xlabel("intervals")
ax.set_ylabel("months")

A = tmp_matrix
mask =  np.tri(A.shape[0], k=-1)
A = np.ma.array(A, mask=mask) # mask out the lower triangle
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111)
ax.set_xlabel("interval")
ax.set_ylabel("period")
cmap = CM.get_cmap('RdBu_r', 50000) 
cmap.set_bad('w') # default value is 'k'
ax.imshow(A, interpolation="nearest", cmap=cmap)
revenue_datetime_df = train_data[["revenue" , "date"]].dropna()
revenue_datetime_df["revenue"] = revenue_datetime_df.revenue.astype(np.int64)
revenue_datetime_df.head()
total_revenue_daily_df = revenue_datetime_df.groupby(by=["date"],axis=0).sum()
total_revenue_daily_df.head()
total_visitNumber_daily_df = train_data[["date","visitNumber"]].groupby(by=["date"],axis=0).sum()
total_visitNumber_daily_df.head()
datetime_revenue_visits_df = pd.concat([total_revenue_daily_df,total_visitNumber_daily_df],axis=1)

fig, ax1 = plt.subplots(figsize=(20,10))
t = datetime_revenue_visits_df.index
s1 = datetime_revenue_visits_df["visitNumber"]
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('day')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('visitNumber', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
s2 = datetime_revenue_visits_df["revenue"]
ax2.plot(t, s2, 'r--')
ax2.set_ylabel('revenue', color='r')
ax2.tick_params('y', colors='r')
fig.tight_layout()
revenue_df = train_data.dropna(subset=["revenue"])
revenue_os_df = pd.DataFrame(revenue_df.device.apply(json.loads).tolist())[["browser","operatingSystem","deviceCategory","isMobile"]]

buys_is_mobile_dict = dict(collections.Counter(list(revenue_os_df.isMobile)))
percent_buys_is_mobile_dict = {k: v / total for total in (sum(buys_is_mobile_dict.values()),) for k, v in buys_is_mobile_dict.items()}
sizes = list(percent_buys_is_mobile_dict.values())
explode=(0,0.1)
labels = 'isNotMobile', 'isMobile'
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_title("buys mobile distro")
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
mobiles_browsers = dict(collections.Counter(revenue_os_df[revenue_os_df["isMobile"] == True]["browser"]))
not_mobiles_browsers = dict(collections.Counter(revenue_os_df[revenue_os_df["isMobile"] == False]["browser"]))
print("for mobile users:")
for i,v in mobiles_browsers.items():
    print("{}:{}".format(i,v))
print("\nfor not mobile users:")
for i,v in not_mobiles_browsers.items():
    print("{}:{}".format(i,v))
vals = np.array([[552.,6.,2.,12.,16.,431.], [9801.,189.,58.,93.,5.,349.]])

fig, ax = plt.subplots(subplot_kw=dict(polar=True),figsize=(9,9))
size = 0.3
valsnorm = vals / np.sum(vals) * 2 * np.pi

# obtain the ordinates of the bar edges
valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(vals.shape)

cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3) * 4)
inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

ax.bar(x=valsleft[:, 0],
       width=valsnorm.sum(axis=1), bottom=1 - size, height=size,
       color=outer_colors, edgecolor='w', linewidth=1, align="edge")

ax.bar(x=valsleft.flatten(),
       width=valsnorm.flatten(), bottom=1 - 2 * size, height=size,
       color=inner_colors, edgecolor='w', linewidth=1, align="edge")
# ax.set_axis_off()

ax.set(title="Nested pi-plot for buyers devices.")
df_train = train_data.drop(["date", "socialEngagementType", "visitStartTime", "visitId", "fullVisitorId" , "revenue","customDimensions"], axis=1)

devices_df = pd.DataFrame(df_train.device.apply(json.loads).tolist())[["browser", "operatingSystem", "deviceCategory", "isMobile"]]
geo_df = pd.DataFrame(df_train.geoNetwork.apply(json.loads).tolist())[["continent", "subContinent", "country", "city"]]
traffic_source_df = pd.DataFrame(df_train.trafficSource.apply(json.loads).tolist())[["keyword", "medium", "source"]]
totals_df = pd.DataFrame(df_train.totals.apply(json.loads).tolist())[["transactionRevenue", "newVisits", "bounces", "pageviews", "hits"]]


df_train = pd.concat([df_train.drop(["hits"],axis=1), devices_df, geo_df, traffic_source_df, totals_df], axis=1)
df_train = df_train.drop(["device", "geoNetwork", "trafficSource", "totals"], axis=1)

df_train.head(1)
df_train["transactionRevenue"] = df_train["transactionRevenue"].fillna(0)
df_train["bounces"] = df_train["bounces"].fillna(0)
df_train["pageviews"] = df_train["pageviews"].fillna(0)
df_train["hits"] = df_train["hits"].fillna(0)
df_train["newVisits"] = df_train["newVisits"].fillna(0)
df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42)

df_train["transactionRevenue"] = df_train["transactionRevenue"].astype(np.float)
df_test["transactionRevenue"] = df_test["transactionRevenue"].astype(np.float)
"Finaly, we have these columns for our regression problems: {}".format(df_train.columns)
df_train.head(1)
categorical_features = ['channelGrouping', 'browser', 'operatingSystem', 'deviceCategory', 'isMobile',
                        'continent', 'subContinent', 'country', 'city', 'keyword', 'medium', 'source']

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
predicted_revenue = gbm.predict(df_test.loc[:,df_test.columns != "transactionRevenue"], num_iteration=gbm.best_iteration)
predicted_revenue[predicted_revenue < 0] = 0 
df_test["predicted"] = np.expm1(predicted_revenue)
df_test[["transactionRevenue","predicted"]].head(10)
import gc; gc.collect()
import time; time.sleep(5)

df_train = pd.read_csv(filepath_or_buffer="../input/train_v2.csv",nrows=50000)
df_actual_test = pd.read_csv(filepath_or_buffer="../input/test_v2.csv",nrows=25000)

# drop useless features => date, fullVisitorId, sessionId, socialEngagement, visitStartTime
df_train = df_train.drop(["date", "socialEngagementType", "visitStartTime", "visitId", "fullVisitorId"], axis=1)
df_actual_test = df_actual_test.drop(["date", "socialEngagementType", "visitStartTime", "visitId"], axis=1)


#preprocessing for trains
devices_df = pd.DataFrame(df_train.device.apply(json.loads).tolist())[["browser", "operatingSystem", "deviceCategory", "isMobile"]]
geo_df = pd.DataFrame(df_train.geoNetwork.apply(json.loads).tolist())[["continent", "subContinent", "country", "city"]]
traffic_source_df = pd.DataFrame(df_train.trafficSource.apply(json.loads).tolist())[["keyword", "medium", "source"]]
totals_df = pd.DataFrame(df_train.totals.apply(json.loads).tolist())[["transactionRevenue", "newVisits", "bounces", "pageviews", "hits"]]
df_train = pd.concat([df_train.drop(["hits"],axis=1), devices_df, geo_df, traffic_source_df, totals_df], axis=1)
df_train = df_train.drop(["device", "geoNetwork", "trafficSource", "totals"], axis=1)
df_train["transactionRevenue"] = df_train["transactionRevenue"].fillna(0)
df_train["bounces"] = df_train["bounces"].fillna(0)
df_train["pageviews"] = df_train["pageviews"].fillna(0)
df_train["hits"] = df_train["hits"].fillna(0)
df_train["newVisits"] = df_train["newVisits"].fillna(0)

#preprocessing for tests
devices_df = pd.DataFrame(df_actual_test.device.apply(json.loads).tolist())[["browser", "operatingSystem", "deviceCategory", "isMobile"]]
geo_df = pd.DataFrame(df_actual_test.geoNetwork.apply(json.loads).tolist())[["continent", "subContinent", "country", "city"]]
traffic_source_df = pd.DataFrame(df_actual_test.trafficSource.apply(json.loads).tolist())[["keyword", "medium", "source"]]
totals_df = pd.DataFrame(df_actual_test.totals.apply(json.loads).tolist())[["newVisits", "bounces", "pageviews", "hits"]]
df_actual_test = pd.concat([df_actual_test.drop(["hits"],axis=1), devices_df, geo_df, traffic_source_df, totals_df], axis=1)
df_actual_test = df_actual_test.drop(["device", "geoNetwork", "trafficSource", "totals"], axis=1)
# df_actual_test["transactionRevenue"] = df_train["transactionRevenue"].fillna(0)
df_actual_test["bounces"] = df_train["bounces"].fillna(0)
df_actual_test["pageviews"] = df_train["pageviews"].fillna(0)
df_actual_test["hits"] = df_train["hits"].fillna(0)
df_actual_test["newVisits"] = df_train["newVisits"].fillna(0)

#garbage collector ';-)'
del devices_df,geo_df,traffic_source_df,totals_df


#evaluation 
df_train, df_eval = train_test_split(df_train, test_size=0.2, random_state=42)

# lgb_train = lgb.Dataset(df_train.loc[:, df_train.columns != "revenue"], df_train["revenue"])
# lgb_eval = lgb.Dataset(df_test.loc[:, df_test.columns != "revenue"], df_test["revenue"], reference=lgb_train)

df_train["transactionRevenue"] = df_train["transactionRevenue"].astype(np.float)
df_eval["transactionRevenue"] = df_eval["transactionRevenue"].astype(np.float)

print(df_train.columns)
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

print('Start training...')
df_actual_test = df_actual_test.drop(["customDimensions"],axis=1)
df_train = df_train.drop(["customDimensions"],axis=1)
df_eval = df_eval.drop(["customDimensions"],axis=1)
categorical_features = ['channelGrouping', 'browser', 'operatingSystem', 'deviceCategory', 'isMobile',
                        'continent', 'subContinent', 'country', 'city', 'keyword', 'medium', 'source']

numerical_features = ['visitNumber', 'newVisits', 'bounces', 'pageviews', 'hits']

for column_iter in categorical_features:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train[column_iter].values.astype('str')) + list(df_eval[column_iter].values.astype('str')) + list(df_actual_test[column_iter].values.astype('str')))
    
    df_train[column_iter] = lbl.transform(list(df_train[column_iter].values.astype('str')))
    df_eval[column_iter] = lbl.transform(list(df_eval[column_iter].values.astype('str')))
    df_actual_test[column_iter] = lbl.transform(list(df_actual_test[column_iter].values.astype('str')))

for column_iter in numerical_features:
    df_train[column_iter] = df_train[column_iter].astype(np.float)
    df_eval[column_iter] = df_eval[column_iter].astype(np.float)
    df_actual_test[column_iter] = df_actual_test[column_iter].astype(np.float)
lgb_train = lgb.Dataset(df_train.loc[:,df_train.columns != "transactionRevenue"], np.log1p(df_train.loc[:,"transactionRevenue"]))
lgb_eval = lgb.Dataset(df_eval.loc[:,df_eval.columns != "transactionRevenue"], np.log1p(df_eval.loc[:,"transactionRevenue"]), reference=lgb_train)
gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=[lgb_eval], early_stopping_rounds=100,verbose_eval=100)
eval_predicted_revenue = gbm.predict(df_eval.loc[:,df_eval.columns != "transactionRevenue"], num_iteration=gbm.best_iteration)
eval_predicted_revenue[eval_predicted_revenue < 0] = 0 
df_eval["predicted"] = np.expm1(eval_predicted_revenue)
df_eval[["transactionRevenue","predicted"]].head()
actual_predicted_revenue = gbm.predict(df_actual_test.loc[:,df_actual_test.columns != "fullVisitorId"], num_iteration=gbm.best_iteration)
actual_predicted_revenue[actual_predicted_revenue < 0] = 0 
# df_actual_test["predicted"] = np.expm1(actual_predicted_revenue)
df_actual_test["predicted"] = actual_predicted_revenue
df_actual_test.head()

df_actual_test = df_actual_test[["fullVisitorId" , "predicted"]]
df_actual_test["fullVisitorId"] = df_actual_test.fullVisitorId.astype('str')
df_actual_test["predicted"] = df_actual_test.predicted.astype(np.float)
df_actual_test.index = df_actual_test.fullVisitorId
df_actual_test = df_actual_test.drop("fullVisitorId",axis=1)
df_actual_test.head()
df_submission_test = pd.read_csv(filepath_or_buffer="../input/sample_submission_v2.csv",index_col="fullVisitorId")
df_submission_test.shape
"test shape is :{} and submission shape is : {}".format(df_actual_test.shape , df_submission_test.shape)
final_df = df_actual_test.loc[df_submission_test.index,:]
final_df = final_df[~final_df.index.duplicated(keep='first')]
final_df = final_df.rename(index=str, columns={"predicted": "PredictedLogRevenue"})
final_df.PredictedLogRevenue.fillna(0).head()
# final_df.head()
fig, ax = plt.subplots(figsize=(10,16))
lgb.plot_importance(gbm, max_num_features=30, height=0.8, ax=ax)
plt.title("Feature Importance", fontsize=15)
plt.show()