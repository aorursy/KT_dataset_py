import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#df = pd.read_csv("log_temp.log")

df_csv = pd.read_csv("../input/dht11-temperature-and-humidity-sensor-1-day/log_temp.csv")

df_csv.head()
#df = pd.read_csv("log_temp.log")

df = pd.read_csv("../input/dht11-temperature-and-humidity-sensor-1-day/log_temp.log")

df.head()
df = pd.read_csv("../input/dht11-temperature-and-humidity-sensor-1-day/log_temp.log", sep=" ", header=None)

df.head()
df.columns = ["date", "hour", "temp", "humi"]

df.head()
df.info()
df.describe(include="all")
df["date"].value_counts()
df["temp"].value_counts()
df["humi"].value_counts()
df = df[df.date=="2019-03-15"]
df["date"].value_counts()
df = df.replace("error",np.NaN)
df.info()
df = df.fillna("0000.0")
df.info()
df["hour"] = df["hour"].str.slice(stop=2)
df["temp"] = df["temp"].str.slice(start=2,stop=6)
df["humi"] = df["humi"].str.slice(start=2,stop=6)
df.head()
df=df.drop("date",1)

df.head()
df.reset_index(drop=True,inplace=True)

df.head()
df.dtypes
df.hour = df.hour.astype(int)

df.temp= df.temp.astype(float)

df.humi = df.humi.astype(float)
df.dtypes
df.groupby("hour")["temp"].mean().plot(kind="line",color="blue")

df.groupby("hour")["humi"].mean().plot(kind="line",color="orange")
#Two columns to review

columns = ["temp","humi"]

#Identify the 00.0 as the value to replace

flag = 00.0



#For each two columns, get and save the mean value of the column as a temp value in the case that the first 

#value will be 00.0, otherwise save the first value of the column as a temp value to replace in case of 00.0

for each in columns:

    if df[each].iloc[0] == flag:

        temp_t = df[each].mean()

    else:

        temp_t = df[each].iloc[0]

#In case of 00.0 replace with the temp value, otherwise update the temp value with the current value of the column

    for index, row in df.iterrows():

        if row[each] == flag:

            df.loc[index, each] = temp_t

        else:

            temp_t = df[each].iloc[index]  
df.describe()
df.groupby("hour")["temp"].mean().plot(kind="line",color="blue")

df.groupby("hour")["humi"].mean().plot(kind="line",color="orange")



xint2 = np.arange(df["hour"].min(), df["hour"].max()+1, 2)

plt.xticks(xint2)

yint2 = np.arange(df["temp"].min()-2, df["temp"].max()+2, 2)

plt.yticks(yint2)



plt.grid()

plt.title("2019-03-15")

plt.legend(("Temperature","Humidity"))

plt.xlabel("Hour")

plt.ylabel("°C / RH")