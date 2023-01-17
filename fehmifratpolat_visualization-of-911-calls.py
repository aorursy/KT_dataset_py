import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("../input/montcoalert/911.csv")
df.head()
df.columns
df["title"].iloc[0].split(":")[0]
df["Reason_1"] = df["title"].apply(lambda title: title.split(":")[0])
df["Reason_1"].head(5)
df["Reason_1"].value_counts()
df["Reason_2"] = df["title"].apply(lambda title: title.split(":")[1])
df["Reason_2"].value_counts().head(10)
df["timeStamp"] = pd.to_datetime(df["timeStamp"])
time = df["timeStamp"].iloc[0]
print(time.hour)
print(time.dayofweek)
df["Hour"] = df["timeStamp"].apply(lambda time: time.hour)
df["Month"] = df["timeStamp"].apply(lambda time: time.month)
df["Year"] = df["timeStamp"].apply(lambda time: time.year)
df["Day_of_Week"] = df["timeStamp"].apply(lambda time: time.dayofweek)
dmap = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
df["Day_of_Week"] = df["Day_of_Week"].map(dmap)
df.head()
dmap1 = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
sns.countplot(x="Day_of_Week", data=df, order = dmap1)
sns.countplot(x="Day_of_Week", data=df, hue="Reason_1", order = dmap1)
group_month = df.groupby("Month").count()
group_month.head(12)
sns.countplot(x="Month", data=df, hue="Reason_1")
sns.countplot(x= "Year", data= df, palette="Paired", hue = "Reason_1")
plt.title(" Calls Reason Yearly having the hue of reasons")
plt.show()