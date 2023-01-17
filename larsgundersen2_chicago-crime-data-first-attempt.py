import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chicago_crime")
bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")
bq_assistant.list_tables()
bq_assistant.head("crime", num_rows=3)
bq_assistant.table_schema("crime")
query3=  """SELECT year,popular_crimetype from (
select year, max(primary_type) as popular_crimetype
  FROM 
    `bigquery-public-data.chicago_crime.crime`
where primary_type != ""
group by year)
ORDER BY
  year DESC
        """
all_data = chicago_crime.query_to_pandas_safe(query3)
all_data.head(10)

query4 =  """SELECT date,year,primary_type 
from 
    `bigquery-public-data.chicago_crime.crime`
where year < 2018 and year > 2014
ORDER BY
  year DESC
        """
all_data = chicago_crime.query_to_pandas_safe(query4)
all_data.head(10)


import pandas as pd
import datetime as dt
all_data["date"].head()
all_data['date'] = pd.to_datetime(all_data['date'])
all_data['day'] = all_data['date'].dt.date
all_data['day'] = all_data['date'].dt.strftime('%Y-%m-%d')
all_data.head()
import pandas as pd
df = pd.read_csv('../input/austin-weathercsv/austin_weather.csv')
df.head()
df["day"] = df["Date"]

rain_data = df[["day","Events"]]
print(rain_data.head())
print(all_data.head())

lst = rain_data.iloc[0][1]
print(lst == "Rain")



merged_df = pd.merge(all_data,rain_data, on='day')
merged_df["Events"].unique()


import matplotlib.pyplot as plt
import seaborn as sns
bad_weather = ['Rain' , 'Thunderstorm', 'Rain', 'Fog , Rain', 'Thunderstorm',
       'Fog', 'Fog , Rain , Thunderstorm', 'Fog , Thunderstorm']

merged_df['primary_type'].unique()
#Try to make a binary variable for Rain

merged_df["bad_weather"] = 0
merged_df["rain"] = 0
#is_rain = (merged_df["Events"]) == "Rain" 
merged_df.loc[merged_df['Events'].isin(bad_weather), 'bad_weather'] = 1
merged_df.loc[is_rain,"rain"] = 1

 
merged_df["thunderstorm"] = 0
is_thunder = (merged_df["Events"]) == "Thunderstorm"
merged_df.loc[is_thunder,"thunderstorm"] = 1

rain_crime_percentage = merged_df[["primary_type", "rain"]].groupby("primary_type").mean()

sns.countplot(x=merged_df["primary_type"],order=pd.value_counts(merged_df['primary_type']).iloc[:5].index)



sns.countplot(x=merged_df["rain"],order=pd.value_counts(merged_df['rain']).iloc[:5].index)

sns.countplot(x=merged_df["bad_weather"])
sns.barplot(y="primary_type", x="bad_weather", orient = 'h',data=merged_df, ci=None,order=pd.value_counts(merged_df['primary_type']).iloc[:20].index)
sns.barplot(y="primary_type", x="thunderstorm",orient = 'h', data=merged_df, ci=None,order=pd.value_counts(merged_df['primary_type']).iloc[:20].index)