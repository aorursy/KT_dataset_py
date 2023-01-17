# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
mydateparser = lambda x: pd.datetime.strptime(x, "%d-%m-%Y %H:%M")

gen_data = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv",index_col = "DATE_TIME",parse_dates = ["DATE_TIME"] , date_parser = mydateparser)
gen_data.head()
gen_data["Date"] = pd.to_datetime(gen_data.index.map(lambda x : x.date()))

gen_data["Time"] = gen_data.index.map(lambda x : x.time())
gen_data.loc[(gen_data["DC_POWER"] == 0) & (gen_data["AC_POWER"] != 0)]
gen_data.loc[(gen_data["DC_POWER"] != 0) & (gen_data["AC_POWER"] == 0)]
[any(pd.isnull(gen_data[column])) for column in gen_data.columns]
numeric_columns = [column for column in gen_data.columns if gen_data[column].dtype in ['int64','float64']]

other_columns = [column for column in gen_data.columns if column not in numeric_columns]
[any(np.isnan(gen_data[column])) for column in numeric_columns]
gen_data["Date"].head()
gen_data.columns
len(gen_data["SOURCE_KEY"].unique())
inverters = gen_data["SOURCE_KEY"].unique()
fig = plt.figure(figsize = (25,16))

for i,inverter in enumerate(inverters,1):

    plt.subplot(6,4,i)

    plt.yscale("log")

    gen_data.loc[(gen_data["Date"] == "2020-05-15") &  (gen_data["SOURCE_KEY"] == inverter),"DC_POWER"].plot(label = inverter + " DC")

    gen_data.loc[(gen_data["Date"] == "2020-05-15") & (gen_data["SOURCE_KEY"] == inverter),"AC_POWER"].plot(label = inverter + " AC")

    plt.legend()
gen_data.groupby("SOURCE_KEY").count()
34 * 24 * 4 #Number of data points required
gen_data["Date"].unique()
gen_data.groupby("SOURCE_KEY").sum()["DC_POWER"]
#split the dataframes by inverter IDs first

split_by_inverters = {}

for inverter in inverters:

    split_by_inverters[inverter] = gen_data.loc[gen_data["SOURCE_KEY"] == inverter]
unique_dates = gen_data.index.map(lambda x : x.date()).unique()
temp = split_by_inverters['1BY6WEcLGh8j5v7']
fig = plt.figure(figsize = (30,25))

inverter_daily_power = {}

for i,(inverter,data) in enumerate(split_by_inverters.items(),1):

    plt.subplot(6,4,i)

    inverter_daily_power[inverter] = data.groupby("Date").sum()["DC_POWER"]

    inverter_daily_power[inverter].plot(label = inverter)

    plt.legend()
weather_data = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv",index_col = "DATE_TIME",parse_dates = True)
weather_data["Date"] = pd.to_datetime(weather_data.index.map(lambda x : x.date()))

weather_data["Time"] = weather_data.index.map(lambda x : x.time())
weather_data.head()
fig = plt.figure(figsize = (18,4))

plt.subplot(131)

weather_data.loc[(weather_data["Date"] == "2020-05-15"), "IRRADIATION"].plot(legend = True)

#plt.legend()

plt.subplot(132)

weather_data.loc[(weather_data["Date"] == "2020-05-15"),"AMBIENT_TEMPERATURE"].plot(legend = True)

plt.subplot(133)

weather_data.loc[(weather_data["Date"] == "2020-05-15 00:00:00"),"MODULE_TEMPERATURE"].plot(legend = True)
sns.scatterplot(x = weather_data.loc[(weather_data["Date"] == "2020-05-15"),"AMBIENT_TEMPERATURE"], y =weather_data.loc[(weather_data["Date"] == "2020-05-15"),"MODULE_TEMPERATURE"])
#Timestamp of maximum irradiation on the 15th of May

weather_data.loc[(weather_data["Date"] == "2020-05-15"),"IRRADIATION"].idxmax()
#Timestamp of maximum ambient temperature on the 15th of May

weather_data.loc[(weather_data["Date"] == "2020-05-15"),"AMBIENT_TEMPERATURE"].idxmax()
weather_data.loc[(weather_data["Date"] == "2020-05-15"),"MODULE_TEMPERATURE"].idxmax()
inverter_daily_power.keys()
weather_data["date"] = weather_data.index.map(lambda x : x.date())

daily_irradiation = weather_data.groupby("date").sum()["IRRADIATION"]

sns.scatterplot(x = daily_irradiation, y = inverter_daily_power["1BY6WEcLGh8j5v7"])
max_temps = weather_data.groupby("date").max()["AMBIENT_TEMPERATURE"]

min_temps = weather_data.groupby("date").min()["AMBIENT_TEMPERATURE"]
plt.figure(figsize = (12,6))

max_temps.plot(label = "Maximum Temperature")

min_temps.plot(label = "Minimum Temperature")

plt.legend()
max_temps = weather_data.groupby("date").max()["AMBIENT_TEMPERATURE"]

min_temps = weather_data.groupby("date").min()["AMBIENT_TEMPERATURE"]

diff_temps = max_temps - min_temps

daily_irradiation = weather_data.groupby("date").sum()["IRRADIATION"]
sns.scatterplot(daily_irradiation,diff_temps)
temp_before_sunrise = weather_data.loc[(weather_data["Time"] < pd.to_datetime("07:00").time()) & (weather_data["IRRADIATION"] > 0)].groupby("date")["AMBIENT_TEMPERATURE"].min()
diff_temps = max_temps - temp_before_sunrise
sns.scatterplot(daily_irradiation,diff_temps)
average_power = gen_data.reset_index().groupby("DATE_TIME").mean()[["DC_POWER","AC_POWER"]]
total_power = average_power * gen_data["PLANT_ID"].nunique()
total_power["Date"] = total_power.index.map(lambda x : x.date())

fig = plt.figure()

total_power.groupby("Date").sum().plot()

plt.yscale("log")
sns.regplot(x = weather_data.groupby("Date")["IRRADIATION"].sum(), y = total_power.groupby("Date")["DC_POWER"].sum())
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(weather_data.groupby("Date")["IRRADIATION"].sum().values.reshape(-1,1),total_power.groupby("Date")["DC_POWER"].sum())

model.intercept_,model.coef_
sns.regplot(x = total_power.groupby("Date")["DC_POWER"].sum(),y = total_power.groupby("Date")["AC_POWER"].sum())
from sklearn.linear_model import LinearRegression
a = LinearRegression()

a.fit(total_power.groupby("Date")["DC_POWER"].sum().values.reshape(-1,1),total_power.groupby("Date")["AC_POWER"].sum())

a.intercept_,a.coef_
gen_data_2 = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv",index_col = "DATE_TIME",parse_dates = ["DATE_TIME"])
gen_data_2["Date"] = gen_data_2.index.map(lambda x : x.date())

gen_data_2["Time"] = gen_data_2.index.map(lambda x : x.time())
gen_data_2.groupby("SOURCE_KEY").count()
[any(pd.isnull(gen_data_2[column])) for column in gen_data_2.columns]
inverters = gen_data_2["SOURCE_KEY"].unique()

fig = plt.figure(figsize = (25,16))

for i,inverter in enumerate(inverters,1):

    plt.subplot(6,4,i)

    plt.yscale("log")

    gen_data_2.loc[(gen_data_2["Date"] == pd.to_datetime("2020-05-15")) &  (gen_data_2["SOURCE_KEY"] == inverter),"DC_POWER"].plot(label = inverter + " DC")

    gen_data_2.loc[(gen_data_2["Date"] == pd.to_datetime("2020-05-15")) & (gen_data_2["SOURCE_KEY"] == inverter),"AC_POWER"].plot(label = inverter + " AC")

    plt.legend()
inverters
average_power_2 = gen_data_2.reset_index().groupby("DATE_TIME").mean()[["DC_POWER","AC_POWER"]]

total_power_2 = average_power_2 * gen_data_2["SOURCE_KEY"].nunique()

total_power_2["Date"] = total_power_2.index.map(lambda x : x.date())
weather_data_2 = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv",index_col = "DATE_TIME",parse_dates = True)

weather_data_2["Date"] = weather_data_2.index.map(lambda x : x.date())

weather_data_2["Time"] = weather_data_2.index.map(lambda x : x.time())
sns.regplot(x = weather_data_2.groupby("Date")["IRRADIATION"].sum(),y = total_power_2.groupby("Date").sum()["DC_POWER"])
sns.regplot(x = total_power_2.groupby("Date").sum()["DC_POWER"], y = total_power_2.groupby("Date").sum()["AC_POWER"])
model_2 = LinearRegression()

model_2.fit(total_power_2.groupby("Date").sum()["DC_POWER"].values.reshape(-1,1), y = total_power_2.groupby("Date").sum()["AC_POWER"])

model_2.coef_,model_2.intercept_