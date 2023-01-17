import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



df=pd.read_csv(r"../input/911.csv")
df.head()
df.describe()
# Drop rows where no zip-code is provided so there is only full data-items (~8% of data)

df.dropna(inplace=True)

df.describe()
df.drop("e",axis=1).head(1)
df.plot.scatter(y="lat",x="lng")
fig=plt.figure(figsize=(10,10))

ax=fig.add_axes([0,0,1,1])

ax.set_xlim(-74.98, -75.75)

ax.set_ylim(39.90, 40.67)

ax.invert_xaxis()

ax.scatter(y=df["lat"],x=df["lng"],s=0.001)
df[(df["zip"]<30000)&(df["zip"]>18500)]["zip"].hist(bins=40)
print(df.zip.corr(df.lat))

print(df.zip.corr(df.lng))
from datetime import datetime as dt

df["Day"]=df["timeStamp"].apply(lambda x: dt.strptime(str(x)[0:10], '%Y-%m-%d'))

df["Month"]=df["timeStamp"].apply(lambda x: dt.strptime(str(x)[5:7], '%m'))

df["Year"]=df["timeStamp"].apply(lambda x: dt.strptime(str(x)[0:4], '%Y'))

df["Time"]=df["timeStamp"].apply(lambda x: dt.strptime(str(x)[11:19], '%X'))
df.groupby("Year").size()
by_year=df.groupby("Year").size()

by_year.plot(kind="bar")
by_month=df.groupby("Month").size()

by_month.plot(kind="bar")
df2=df[df["Year"]!="2015-01-01"]

by_month=df2.groupby("Month").size()

by_month.plot()
by_day=df.groupby("Day").size()

by_day.plot()
# Thanks stackoverflow: https://stackoverflow.com/questions/30222533/create-a-day-of-week-column-in-a-pandas-dataframe-using-python

df["Weekday"]=df["Day"].dt.weekday_name
df.groupby("Weekday").size().plot(kind="bar")

#weekday.plot()
by_time=df.groupby("Time").size()

by_time.size
by_time.plot()
two_calls=df.groupby("addr")

two_calls.size().head(5)
incidents=df.groupby("title")

incidents.size().head(5)
incidents=df.groupby("title").sum()

top50=incidents.sort_values("e",ascending=False).head(50).drop(["lat","lng","zip"],axis=1)

top50
def fire(title):

        if title[0:4]=="Fire":

            return 1

        else:

            return 0

def Traffic(title):

        if title[0:7]=="Traffic":

            return 1

        else:

            return 0



def EMS(title):

        if title[0:3]=="EMS":

            return 1

        else:

            return 0



df["Fire"]=df["title"].apply(lambda x: fire(x))

df["Traffic"]=df["title"].apply(lambda x: Traffic(x))

df["EMS"]=df["title"].apply(lambda x: EMS(x))



df.head(10)
df.Fire.sum()
df.Traffic.sum()
df.EMS.sum()
def categorization(title):

    if title[0:4]=="Fire":

            return "Fire"

    elif title[0:7]=="Traffic":

            return "Traffic"

    elif title[0:3]=="EMS":

            return "EMS"

        

df["Category"]=df["title"].apply(lambda x: categorization(x))

df.head(5)
sns.countplot(df.Category)