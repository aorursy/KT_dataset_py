# Big Query
from google.cloud import bigquery
from bq_helper import BigQueryHelper

# I/O and Computation
import numpy as np
import pandas as pd

# Viz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
%matplotlib inline
# Glance
bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
bq_assistant.head("pm25_frm_daily_summary", num_rows=3)
# Helpers
def uniques_counts(cols, data):
    """
    Just want to peak at the categorical classes and their data occurence
    """
    for x in data.columns:
        print("{} Column has {} Unique Values".format(x, len(set(data[x]))))
    print("Shape:\n{} rows, {} columns".format(*data.shape))
    print("Column Names:\n{}\n".format(data.columns))
    for x in cols:
        for y in data[x].unique():
            print("{} from {} has {} values".format(y,x,data[x][data[x]==y].shape))
QUERY = """
    SELECT
        date_local,
        aqi as pm25,
        city_name as City,
        county_name as County,
        sample_duration,
        poc
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
      state_name = "California"
      AND EXTRACT(YEAR FROM date_local) = 2015
        """
bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
pm25 = bq_assistant.query_to_pandas(QUERY)
uniques_counts(cols=["poc","sample_duration"], data=pm25)
pm25.head()
pm25 = pm25[pm25["sample_duration"]!= "1 HOUR"]
pm25.drop(["sample_duration","poc"], axis=1, inplace= True)
pm25 = pm25.groupby(["date_local","County","City"]).mean().reset_index()
pm25.head()
# PM 10
QUERY = """
    SELECT
        date_local,
        aqi as pm10,
        city_name as City,
        county_name as County,
        sample_duration,
        poc
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm10_daily_summary`
    WHERE
      state_name = "California"
      AND EXTRACT(YEAR FROM date_local) = 2015
        """
bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
pm10 = bq_assistant.query_to_pandas(QUERY)
uniques_counts(cols =["sample_duration","poc"], data= pm10)
pm10.drop(["sample_duration","poc"], axis=1, inplace= True)
pm10 = pm10.groupby(["date_local","County","City"]).mean().reset_index()
pm10.head()
df = pd.merge(pm10, pm25, on=["date_local","County","City"], how='outer')
print("Shape: {}".format(df.shape))
print("Missing:\n",df.isnull().sum())
df.head()
f, ax = plt.subplots(1,1,figsize=(15,5))

state_avg = (df[["pm10","pm25","County"]]
 .groupby(["County"])
 .agg({"pm10": "mean",
      "pm25": "mean"})
 .reset_index()
 .melt("County"))

sns.barplot(x="County", y="value", hue="variable", data=state_avg, ax=ax)
plt.xticks(rotation=45)
plt.title("2015 Annual Averge Pollution by State")
plt.show()
def cat_mean_plot(target, var,data):
    data[[target,var]].groupby([var]).mean().sort_values(by=target, ascending=False).plot.bar()
# cat_mean_plot(target="pm10", var="County",data=df)
# plt.show()
df[["pm25","date_local","City","County"]].groupby(["date_local"]).mean().plot(rot=90)
plt.ylabel("AQI")
plt.xlabel("2015")
plt.title("Total Average for PM 2.5")
plt.show()
df[["pm10","date_local","City","County"]].groupby(["date_local"]).mean().plot(rot=90)
plt.ylabel("AQI")
plt.xlabel("2015")
plt.title("Total Average for PM 10")
plt.show()
# New Time Variables
df['date_local'] = pd.to_datetime(df['date_local'],format='%Y-%m-%d')
df["Day of Year"] = df['date_local'].dt.dayofyear
def cat_plot(var, data, timevar, log=False):
    for x in data[var].unique():
        plt.plot(data[data[var] == x].drop(var,axis=1).set_index(timevar), label=x)
        
plt.figure(figsize=(10,5))
cat_plot(var= "County", data=df[["Day of Year","County","pm10"]], timevar="Day of Year", log=True)
plt.ylim([0,200])
plt.title("2015 PM 10 Time Series by State")
plt.ylabel("AQI")
plt.xlabel("Day of Year")
plt.show()
def rolling_cat_plot(var, data, timevar, rolling_wind):
    for x in data[var].unique():
        temp = data[data[var] == x].drop(var,axis=1).set_index(timevar).rolling(window = rolling_wind).mean()
        plt.plot(temp, label=x)
        
plt.figure(figsize=(10,5))
rolling_cat_plot(var= "County", data=df[["Day of Year","County","pm10"]],
                 timevar="Day of Year", rolling_wind=30)
plt.ylim([0,75])
plt.title("2015 PM 10 Time Series by County with Rolling Average")
plt.ylabel("AQI")
plt.xlabel("Day of Year")
plt.show()
plt.figure(figsize=(10,5))
rolling_cat_plot(var= "County", data=df[["Day of Year","County","pm25"]],
                 timevar="Day of Year", rolling_wind=30)
plt.ylim([0,100])
plt.title("2015 PM 2.5 Time Series by County with Rolling Average")
plt.ylabel("AQI")
plt.xlabel("Day of Year")
plt.show()
def top_loc_plots(data, target, cat, timevar, rolling_wind=30,size= (9,6)):
    f, axarr = plt.subplots(2,1,sharex=True, squeeze=True,figsize=size) 
    sliced = data.groupby([cat,timevar]).mean().groupby(level=cat)
    for index,x in enumerate(target):
        temp = sliced[x].mean().nlargest(10).index
        for i in temp:
            lineplot= sliced[x].get_group(i).groupby(pd.Grouper(level=timevar))\
            .mean().rolling(window = rolling_wind).mean()
            axarr[index].plot(lineplot)
        axarr[index].legend(temp,fontsize='small', loc='center left',
                            bbox_to_anchor=(1, 0.5))
        axarr[index].set_ylabel("{}".format(x))
top_loc_plots(data=df[["pm10","pm25","Day of Year","County"]],
              target = ["pm10","pm25"], cat="County", timevar="Day of Year",rolling_wind=20)
plt.tight_layout()
plt.suptitle("PM 2.5 and PM 10 for Top 10 Counties")
plt.subplots_adjust(top=0.94)
plt.xlabel("Day of Year")
plt.show()
top_loc_plots(data=df[["pm10","pm25","Day of Year","City"]],
              target = ["pm10","pm25"], cat="City", timevar="Day of Year",rolling_wind=20)
plt.tight_layout()
plt.suptitle("PM 2.5 and PM 10 for Top 10 Cities")
plt.subplots_adjust(top=0.94)
plt.xlabel("Day of Year")
plt.show()
df.to_csv("pm25_pm10_df.csv")
df.head()