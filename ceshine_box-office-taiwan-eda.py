# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from datetime import datetime, timedelta



import plotly.graph_objects as go

import plotly.express as px
df = pd.read_csv("/kaggle/input/taiwanweeklyboxoffice/box_office.csv", parse_dates=["week", "release_date"])

df.tail()
latest_week = df.week.max()

df_latest = df[df.week == latest_week].copy().sort_values("revenue", ascending=False)

print(f"{df_latest.shape[0]} movies this week")

df_latest.head()
fig = px.scatter(df_latest.iloc[:30], x="revenue", y="theaters", color="country", hover_data=['name'])

fig.update_layout(

    title_text=f"{latest_week.strftime('%Y%m%d')} Box Office (Top 30)", 

    xaxis_title_text='Revenue ($/TWD in log scale)', # xaxis label

    yaxis_title_text='Theaters', # yaxis label

    width=800,

    height=400,

    xaxis_type="log",

    margin=dict(l=20, r=20, t=50, b=50),

    template="plotly_white"

)

fig.show()
df[df.name == "屍速列車：感染半島"]
df[df.name == "我的A級秘密"]
df[df.name == "終極追殺令"]
print(df[["name"]].drop_duplicates().shape[0])

print(df[["name", "agent"]].drop_duplicates().shape[0])

print(df[["name", "publisher"]].drop_duplicates().shape[0])

print(df[["name", "agent", "publisher"]].drop_duplicates().shape[0])
df_tmp = df[df.release_date >= "2017-11-01"].sort_values("week").groupby(["name", "publisher"], as_index=False).first()

# No movie was released on Monday

print(df_tmp.shape[0], df_tmp[df_tmp.week == df_tmp.release_date].shape[0])
df_tmp.head(3)
tmp = []

for _, row in df_tmp.iterrows():

    row = row.copy()

    row["total_revenue"] = row["total_revenue"] - row["revenue"]

    row["total_tickets"] = row["total_tickets"] - row["tickets"]

    row["revenue"] = row["total_revenue"]

    row["tickets"] = row["total_tickets"]

    assert (row["revenue"] >= 0) and (row["tickets"] >= 0)

    row["week"] = row["week"] - timedelta(days=7)

    tmp.append(row)

df_recovered = pd.DataFrame(tmp)

df_recovered.head(3)
df = pd.concat([df, df_recovered], axis=0, ignore_index=True).sort_values(["week", "revenue"], ascending=[True, False]).reset_index(drop=True)

df.head(3)
print("Problematic entries:")

cnt = df.groupby(["name", "publisher", "week"])["week"].count()

print(cnt[cnt > 1])

# throw away a random row for now

print(df.shape[0])

df = df.drop_duplicates(["name", "publisher", "week"])

print(df.shape[0])
df[df.name == "我的A級秘密"]
df[df.name == "屍速列車：感染半島"]
df["quarter"] = df["week"].apply(lambda x: datetime(x.year, (x.month - 1) // 3 * 3 + 1, 1))

df.quarter.unique()
df_quarterly = df.groupby("quarter", as_index=False)[["tickets", "revenue"]].sum()
fig = go.Figure(data=[

    go.Bar(

        x=df_quarterly.quarter.dt.date,

        y=df_quarterly.revenue.values

    )

])

fig.update_layout(

    title_text="Quarterly Box Office Revenue in Taiwan", 

    xaxis_title_text='Quarter', # xaxis label

    yaxis_title_text='Revenue ($/TWD)', # yaxis label

    width=800,

    height=400,

    bargap=0.2, 

    barmode="stack",

    margin=dict(l=20, r=20, t=50, b=50),

    template="plotly_white"

)

fig.show()
df_2019 = df[(df.week >= "2019-01-01") & (df.week < "2020-01-01")]

df_2019_sums = df_2019.groupby(["name", "country"], as_index=False)["tickets", "revenue"].sum().sort_values("revenue", ascending=False)

df_2019_sums
fig = go.Figure(data=[

    go.Bar(

        x=df_2019_sums.iloc[:20].name.values,

        y=df_2019_sums.iloc[:20].revenue.values

    )

])

fig.update_layout(

    title_text="Top 20 Movies in 2019", 

    xaxis_title_text='Quarter', # xaxis label

    yaxis_title_text='Revenue ($/TWD)', # yaxis label

    width=800,

    height=400,

    bargap=0.2, 

    barmode="stack",

    margin=dict(l=20, r=20, t=50, b=50),

    template="plotly_white"

)

fig.show()
df_tmp = df_2019_sums.iloc[:20].sort_values("tickets", ascending=False)

fig = go.Figure(data=[

    go.Bar(

        x=df_tmp .name.values,

        y=df_tmp .tickets.values

    )

])

fig.update_layout(

    title_text="Top 20 Movies in 2019", 

    xaxis_title_text='Quarter', # xaxis label

    yaxis_title_text='Tickets Sold', # yaxis label

    width=800,

    height=400,

    bargap=0.2, 

    barmode="stack",

    margin=dict(l=20, r=20, t=50, b=50),

    template="plotly_white"

)

fig.show()
pd.DataFrame({"Ticket Ranking": df_tmp.name.values, "Revenue Ranking": df_2019_sums.iloc[:20].name.values})
df_2019[df_2019.name == "驚奇隊長"]