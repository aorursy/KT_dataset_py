import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv("../input/cab-trip/Cab_data_details.csv")

df.head()
len(df["Request id"].unique())
df.shape
df.isnull().sum()
df.isnull().sum()/df.shape[0]*100 # df.shape[0] gives the number of rows
df.info()
df.describe(include="all")
df["Request timestamp"].value_counts()
df["Request timestamp"]=df["Request timestamp"].astype(str)
df["Request timestamp"]=df["Request timestamp"].replace("/","-")
df["Request timestamp"]=pd.to_datetime(df["Request timestamp"],dayfirst=True)
df.info()
df["Drop timestamp"]=pd.to_datetime(df["Drop timestamp"],dayfirst=True)
df.info()
df["Drop timestamp"]
req_hour=df["Request timestamp"].dt.hour # Fetching the hour number from the request timestamp
df["req_hour"]=req_hour # Adding a new column "req_hour" from the hour number fetched from above.
req_day=df["Request timestamp"].dt.day # Fetching the day number from the request timestamp
df["req_day"]=req_day # Adding a new column "req_day" from the day number fetched from above
sns.countplot(x="req_hour",data=df,hue="Status")

plt.show()
sns.factorplot(x="req_hour",data=df,row="req_day",hue="Status",kind="count")

plt.show()
sns.factorplot(x="req_hour",data=df,row="req_day",hue="Pickup point",kind="count")

plt.show()
sns.factorplot(x="req_hour",data=df,hue="Pickup point",kind="count")

plt.show()
df
df["Time_Slot"]=0
df
j=0

for i in df["req_hour"]:

    if df.iloc[j,6]<5:

        df.iloc[j,8]="Pre_Morning"

    elif 5<=df.iloc[j,6]<10:

        df.iloc[j,8]="Morning_Rush"

        

    elif 10<=df.iloc[j,6]<17:

        df.iloc[j,8]="Day_Time"

        

    elif 17<=df.iloc[j,6]<22:

        df.iloc[j,8]="Evening_Rush"

    else:

        df.iloc[j,8]="Late_Night"

    j=j+1
df
df["Time_Slot"].value_counts()
plt.figure(figsize=(10,6))

sns.countplot(x="Time_Slot",hue="Status",data=df)

plt.show()
df_morning_rush=df[df['Time_Slot']=='Morning_Rush']
sns.countplot(x="Pickup point",hue="Status",data=df_morning_rush)
df_airport_cancelled=df_morning_rush.loc[(df_morning_rush["Pickup point"]=="Airport") & (df_morning_rush["Status"]=="Cancelled")]
df_airport_cancelled.shape[0]
df_city_cancelled=df_morning_rush.loc[(df_morning_rush["Pickup point"]=="City") & (df_morning_rush["Status"]=="Cancelled")]
df_city_cancelled.shape[0]
df_morning_rush
df_morning_rush.loc[(df_morning_rush["Pickup point"]=="City")].shape[0]
df_morning_rush.loc[(df_morning_rush["Pickup point"]=="City") & (df_morning_rush["Status"]=="Cancelled")].shape[0]
df_morning_rush.loc[(df_morning_rush["Pickup point"]=="City") & (df_morning_rush["Status"]=="Trip Completed")].shape[0]
df_morning_rush.loc[(df_morning_rush["Pickup point"]=="City") & (df_morning_rush["Status"]=="No Cars Available")].shape[0]
df_morning_rush.loc[(df_morning_rush["Pickup point"]=="Airport")].shape[0]
df_morning_rush.loc[(df_morning_rush["Pickup point"]=="Airport") & (df_morning_rush["Status"]=="Cancelled")].shape[0]
df_morning_rush.loc[(df_morning_rush["Pickup point"]=="Airport") & (df_morning_rush["Status"]=="Trip Completed")].shape[0]
df_morning_rush.loc[(df_morning_rush["Pickup point"]=="Airport") & (df_morning_rush["Status"]=="No Cars Available")].shape[0]
df_evening_rush=df[df['Time_Slot']=='Evening_Rush']
df_city_cancelled=df_evening_rush.loc[(df_evening_rush["Pickup point"]=="City") & (df_evening_rush["Status"]=="Cancelled")]
sns.countplot(x="Pickup point",hue="Status",data=df_evening_rush)
df_city_cancelled.shape[0]
df_evening_rush["Status"].value_counts()
df_evening_rush.loc[(df_evening_rush["Pickup point"]=="City")].shape[0]
df_evening_rush.loc[(df_evening_rush["Pickup point"]=="City") & (df_evening_rush["Status"]=="Cancelled")].shape[0]
df_evening_rush.loc[(df_evening_rush["Pickup point"]=="City") & (df_evening_rush["Status"]=="Trip Completed")].shape[0]
df_evening_rush.loc[(df_evening_rush["Pickup point"]=="City") & (df_evening_rush["Status"]=="No Cars Available")].shape[0]
df_evening_rush.loc[(df_evening_rush["Pickup point"]=="Airport")].shape[0]
df_evening_rush.loc[(df_evening_rush["Pickup point"]=="Airport") & (df_evening_rush["Status"]=="Cancelled")].shape[0]
df_evening_rush.loc[(df_evening_rush["Pickup point"]=="Airport") & (df_evening_rush["Status"]=="Trip Completed")].shape[0]
df_evening_rush.loc[(df_evening_rush["Pickup point"]=="Airport") & (df_evening_rush["Status"]=="No Cars Available")].shape[0]
df_morning_city=df.loc[(df["Pickup point"]=="City")&(df["Time_Slot"]=="Morning_Rush")]
df_morning_city_count=pd.DataFrame(df_morning_city["Status"].value_counts())
df_morning_city_count
df_morning_city_count["Status"].values
df_morning_city_count["Status"].index
fig,ax=plt.subplots()

ax.pie(df_morning_city_count["Status"].values,labels=df_morning_city_count["Status"].index,

      autopct="%.2f%%",startangle=90)

plt.show()
df_evening_airport=df.loc[(df["Pickup point"]=="Airport")&(df["Time_Slot"]=="Evening_Rush")]
df_evening_airport_count=pd.DataFrame(df_evening_airport["Status"].value_counts())
df_evening_airport_count
df_evening_airport_count["Status"].values
df_evening_airport_count["Status"].index
fig,ax=plt.subplots()

ax.pie(df_evening_airport_count["Status"].values,labels=df_evening_airport_count["Status"].index,

      autopct="%.2f%%",startangle=90)

plt.show()