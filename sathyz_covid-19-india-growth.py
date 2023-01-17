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



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

df1 = df.drop(["Province/State", "Lat", "Long"], axis=1).groupby("Country/Region").sum().T

df1.index = pd.to_datetime(df1.index)



grouped = df1.unstack().reset_index().groupby("Country/Region")

by_country = grouped.sum().sort_values(0, ascending=False)



by_country.head(10)
def get_data(grouped, country, min_confirmed=1):

    gdf = grouped.get_group(country)



    day1 = gdf[ gdf[0] >= min_confirmed].level_1.min()

    # print(country, day1)

    index = (gdf.level_1 - day1).dt.days

    index.name = "day"



    confirmed = gdf[0]

    #reported = gdf[0] - gdf[0].shift(1)



    mdf = pd.Series(confirmed.values, index=index, name=country)

    return mdf[mdf.index >= 0]
countries = ["India", "Italy", "Korea, South", "Iran", "US", "China", "France" ]



data = {  country: get_data(grouped, country, min_confirmed=50) for country in countries } 



pdf = pd.DataFrame(data)

ax = pdf.plot(logy=True)

ax.set_ylabel("Confirmed");

ax.legend(loc="upper left");
ax = pdf.plot()

ax.set_ylim(0, 5000)

ax.set_ylabel("confirmed");
india = grouped.get_group('India').rename(columns={"level_1": "date", 0: "confirmed"}).set_index("date")

( 100.0 * ( india - india.shift() ) / india)["2020-03-05":].plot();
( india - india.shift() ).plot(style="x-");
df2 = df1.unstack().reset_index()



n1 = df2.loc[df2[0] > 0].groupby("Country/Region").first().level_1

n1.name = "n1"



n50 = df2.loc[df2[0] > 50].groupby("Country/Region").first().level_1

n50.name = "n50"



n500 = df2.loc[df2[0] > 500].groupby("Country/Region").first().level_1

n500.name = "n500"



n5000 = df2.loc[df2[0] > 5000].groupby("Country/Region").first().level_1

n5000.name = "n5000"



n50000 = df2.loc[df2[0] > 50000].groupby("Country/Region").first().level_1

n50000.name = "n50000"



ndf = pd.DataFrame([n1, n50, n500, n5000, n50000]).T

ax = (ndf.n500 - ndf.n50).dt.days.loc[countries].sort_values().plot(kind="barh")

ax.set_xlabel("days")

ax.set_title("Growth from 50 to 500");
y_median = (ndf.n5000 - ndf.n500).median()

print(y_median)

ax = (ndf.n5000 - ndf.n500).dt.days.dropna().sort_values().plot(kind="barh")

ax.set_xlabel("days")

ax.set_title("Growth from 500 to 5000");
pd.Timestamp.now() - ndf.loc['India'].n500
df2.groupby("Country/Region").max().loc['India']
(ndf.n50000 - ndf.n5000).dt.days.dropna().sort_values().plot(kind="barh");
today = df.columns[-2]

yesterday = df.columns[-3]



g = df.groupby('Country/Region').sum()

new_cases = g[today] - g[yesterday]



pd.concat( [new_cases.sort_values(ascending=False).head(10), new_cases.loc[['India']]]).plot(kind="barh", logx=True, title="New Confirmed Today");
recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv").drop(["Lat", "Long", "Province/State"], axis=1).groupby("Country/Region").sum().sum(axis=1)

deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv").drop(["Lat", "Long", "Province/State"], axis=1).groupby("Country/Region").sum().sum(axis=1)
ax = ( 100.0 * deaths/(recovered + deaths)).loc[countries].sort_values().plot(kind="barh")

ax.set_title("Deaths")

ax.set_xlabel("deaths (%)");