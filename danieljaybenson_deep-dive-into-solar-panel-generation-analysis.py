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
# /kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv

# /kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv

# /kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv

# /kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv

plant_1_generation_df = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv")

plant_1_sensor_df = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")

plant_2_generation_df = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv")

plant_2_sensor_df = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
def df_exploration(df, df_name):

    print(f"First five rows of {df_name}:\n{df.head()}\n")

    print(f"Columns in {df_name}:\n{df.columns}\n")

    print(f"The dataframe contains {df.shape[0]} rows and {df.shape[1]} columns.\n")

    print(f"Statistical summary of {df_name}:\n{df.describe()}\n")

    print(f"Statistic summary of object data of {df_name}:\n{df.describe(exclude='number')}\n")
# Explore plant_1_generation dataset

df_exploration(plant_1_generation_df, "plant_1_generation_df")
df_exploration(plant_1_sensor_df, "plant_1_sensor_df")
df_exploration(plant_2_generation_df, "plant_2_generation_df")
df_exploration(plant_2_sensor_df, "plant_2_sensor_df")
import matplotlib.pyplot as plt

import seaborn as sns



sns.pairplot(plant_1_generation_df)

plt.show()
sns.pairplot(plant_1_sensor_df)

plt.show()
sns.pairplot(plant_2_generation_df)

plt.show()
sns.pairplot(plant_2_sensor_df)

plt.show()
import datetime

# Let's take a look at some of the non-numeric data compared to the numeric data now

# Start by converting the date and time column to datetime format

plant_1_generation_df["DATE_TIME"] = pd.to_datetime(plant_1_generation_df["DATE_TIME"])

plant_1_sensor_df["DATE_TIME"] = pd.to_datetime(plant_1_sensor_df["DATE_TIME"])

plant_2_generation_df["DATE_TIME"] = pd.to_datetime(plant_2_generation_df["DATE_TIME"])

plant_2_sensor_df["DATE_TIME"] = pd.to_datetime(plant_2_sensor_df["DATE_TIME"])



# Check the dtype of the DATE_TIME column to make sure we reformatted correctly

plant_1_generation_df['DATE_TIME'].dtype
# Check a few DATE_TIME rows

plant_1_generation_df[["DATE_TIME", "DC_POWER"]]
# Create 4 scatter plots comparing DATE_TIME to DC_POWER and IRRADIATION levels

# We want to see how the data looks across time.

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

sns.scatterplot(plant_1_generation_df["DATE_TIME"], plant_1_generation_df["DC_POWER"], ax=axes[0,0])

sns.scatterplot(plant_1_sensor_df["DATE_TIME"], plant_1_sensor_df["IRRADIATION"], ax=axes[0,1])

sns.scatterplot(plant_2_generation_df["DATE_TIME"], plant_2_generation_df["DC_POWER"], ax=axes[1,0])

sns.scatterplot(plant_2_sensor_df["DATE_TIME"], plant_2_sensor_df["IRRADIATION"], ax=axes[1,1])

plt.show()
# Create month feature in all datesets from DATE_TIME.month

plant_1_generation_df["month"] = pd.DatetimeIndex(plant_1_generation_df["DATE_TIME"]).month

plant_1_sensor_df["month"] = pd.DatetimeIndex(plant_1_sensor_df["DATE_TIME"]).month

plant_2_generation_df["month"] = pd.DatetimeIndex(plant_2_generation_df["DATE_TIME"]).month

plant_2_sensor_df["month"] = pd.DatetimeIndex(plant_2_sensor_df["DATE_TIME"]).month
# Create day feature in all datasets from DATE_TIME.day

plant_1_generation_df["day"] = pd.DatetimeIndex(plant_1_generation_df["DATE_TIME"]).day

plant_1_sensor_df["day"] = pd.DatetimeIndex(plant_1_sensor_df["DATE_TIME"]).day

plant_2_generation_df["day"] = pd.DatetimeIndex(plant_2_generation_df["DATE_TIME"]).day

plant_2_sensor_df["day"] = pd.DatetimeIndex(plant_2_sensor_df["DATE_TIME"]).day
# Create hour feature in all datasets from DATE_TIME.hour

plant_1_generation_df["hour"] = pd.DatetimeIndex(plant_1_generation_df["DATE_TIME"]).hour

plant_1_sensor_df["hour"] = pd.DatetimeIndex(plant_1_sensor_df["DATE_TIME"]).hour

plant_2_generation_df["hour"] = pd.DatetimeIndex(plant_2_generation_df["DATE_TIME"]).hour

plant_2_sensor_df["hour"] = pd.DatetimeIndex(plant_2_sensor_df["DATE_TIME"]).hour
# Create minute feature in all datasets from DATE_TIME.minute

plant_1_generation_df["minute"] = pd.DatetimeIndex(plant_1_generation_df["DATE_TIME"]).minute

plant_1_sensor_df["minute"] = pd.DatetimeIndex(plant_1_sensor_df["DATE_TIME"]).minute

plant_2_generation_df["minute"] = pd.DatetimeIndex(plant_2_generation_df["DATE_TIME"]).minute

plant_2_sensor_df["minute"] = pd.DatetimeIndex(plant_2_sensor_df["DATE_TIME"]).minute
# Look at a single inverter id to ensure that our datetime feature engineering worked

plant_1_generation_df[plant_1_generation_df["SOURCE_KEY"]=="1BY6WEcLGh8j5v7"]
# Let's compare our datetimes to the rest of the features again

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

sns.scatterplot(plant_1_generation_df["month"], plant_1_generation_df["DC_POWER"], ax=axes[0,0])

sns.scatterplot(plant_1_sensor_df["month"], plant_1_sensor_df["IRRADIATION"], ax=axes[0,1])

sns.scatterplot(plant_2_generation_df["month"], plant_2_generation_df["DC_POWER"], ax=axes[1,0])

sns.scatterplot(plant_2_sensor_df["month"], plant_2_sensor_df["IRRADIATION"], ax=axes[1,1])

plt.show()
sns.pairplot(plant_1_generation_df)

plt.show()
sns.pairplot(plant_1_sensor_df)

plt.show()
sns.pairplot(plant_2_generation_df)

plt.show()
sns.pairplot(plant_2_sensor_df)

plt.show()
# First order of business, let's check out what the deal is with the month descrepancies across

# all of the datasets

print(plant_1_generation_df["month"].value_counts())

print(plant_1_sensor_df["month"].value_counts())

print(plant_2_generation_df["month"].value_counts())

print(plant_2_sensor_df["month"].value_counts())
# Let's create a dataframe that holds only those rows that were recorded at the 23 hour and 45 minute

# mark - this is the last recording in a day, and thus where we will find our total daily_yield

# for any given day.

plant_1_last_daily_recording = plant_1_generation_df[(plant_1_generation_df["hour"]==23) & (plant_1_generation_df["minute"]==45)]

plant_2_last_daily_recording = plant_2_generation_df[(plant_2_generation_df["hour"]==23) & (plant_2_generation_df["minute"]==45)]



# Now we can find the mean of the two last_daily_recording dataframes

plant_1_mean_daily_yield = plant_1_last_daily_recording["DAILY_YIELD"].mean()

plant_2_mean_daily_yield = plant_2_last_daily_recording["DAILY_YIELD"].mean()

print(f"Mean daily yield for plant site 1: {plant_1_mean_daily_yield}")

print(f"Mean daily yield for plant site 2: {plant_2_mean_daily_yield}")
# What is the total irradiation per day?

# Let's get a look at our dataframe for a refresher

plant_1_sensor_df
plant_1_sensor_may = plant_1_sensor_df[plant_1_sensor_df["month"]==5]

plant_1_sensor_june = plant_1_sensor_df[plant_1_sensor_df["month"]==6]

plant_2_sensor_may = plant_2_sensor_df[plant_2_sensor_df["month"]==5]

plant_2_sensor_june = plant_2_sensor_df[plant_2_sensor_df["month"]==6]
# Now using the newly created dataframes we can come up with the total number of days

# accounted for in the data for each plant

plant_1_days_in_may = len(plant_1_sensor_may["day"].value_counts()) # could also have used .unique

plant_1_days_in_june = len(plant_1_sensor_june["day"].value_counts()) # could also have used .unique

plant_2_days_in_may = len(plant_2_sensor_may["day"].value_counts()) # could also have used .unique

plant_2_days_in_june = len(plant_2_sensor_june["day"].value_counts()) # could also have used .unique

plant_1_total_days = plant_1_days_in_may + plant_1_days_in_june

plant_2_total_days = plant_2_days_in_may + plant_2_days_in_june

print(plant_1_total_days, plant_2_total_days)
# Now let's find the irradiation sum for each plant

plant_1_irrad_sum = plant_1_sensor_df["IRRADIATION"].sum()

plant_2_irrad_sum = plant_2_sensor_df["IRRADIATION"].sum()

print(plant_1_irrad_sum, plant_2_irrad_sum)
# Now to find the total irradiation per day we divide our irradiation sums by the number of days

# accounted for in the two plant's datasets

plant_1_rad_per_day = plant_1_irrad_sum / plant_1_total_days

plant_2_rad_per_day = plant_2_irrad_sum / plant_2_total_days

print(f"Plant 1 has a total irradiation per day of {plant_1_rad_per_day}")

print(f"Plant 2 has a total irradiation per day of {plant_2_rad_per_day}")
# What is the max ambient and module temperature

# Let's start by visualizing the ambient and module temperature for 

# each plant site

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Graph for plant 1 sensor ambient temperature data in may and june

sns.lineplot(x=plant_1_sensor_may["day"] ,y=plant_1_sensor_may["AMBIENT_TEMPERATURE"], ax=axes[0, 0])

sns.lineplot(x=plant_1_sensor_june["day"] ,y=plant_1_sensor_june["AMBIENT_TEMPERATURE"], ax=axes[0, 0])

axes[0,0].set_title("Plant 1 Ambient Temperature")

axes[0,0].legend(("may", "june"), loc='upper right', shadow=True)



# Graph for plant 1 sensor module temperature data in may and june

sns.lineplot(x=plant_1_sensor_may["day"], y=plant_1_sensor_may["MODULE_TEMPERATURE"], ax=axes[0, 1])

sns.lineplot(x=plant_1_sensor_june["day"], y=plant_1_sensor_june["MODULE_TEMPERATURE"], ax=axes[0, 1])

axes[0,1].set_title("Plant 1 Module Temperature")

axes[0,1].legend(("may", "june"), loc='upper right', shadow=True)



# Graph for plant 2 sensor ambient temperature data in may and june

sns.lineplot(x=plant_2_sensor_may["day"], y=plant_2_sensor_may["AMBIENT_TEMPERATURE"], ax=axes[1, 0])

sns.lineplot(x=plant_2_sensor_june["day"], y=plant_2_sensor_june["AMBIENT_TEMPERATURE"], ax=axes[1, 0])

axes[1,0].set_title("Plant 2 Ambient Temperature")

axes[1,0].legend(("may", "june"), loc='upper right', shadow=True)



# Graph for plant 1 sensor module temperature data in may and june

sns.lineplot(x=plant_2_sensor_may["day"], y=plant_2_sensor_may["MODULE_TEMPERATURE"], ax=axes[1, 1])

sns.lineplot(x=plant_2_sensor_june["day"], y=plant_2_sensor_june["MODULE_TEMPERATURE"], ax=axes[1, 1])

axes[1,1].set_title("Plant 2 Module Temperature")

axes[1,1].legend(("may", "june"), loc='upper right', shadow=True)



plt.show()
# Let's quantify the max temperatures now

plant_1_max_amb_temp = plant_1_sensor_df["AMBIENT_TEMPERATURE"].max()

plant_1_max_mod_temp = plant_1_sensor_df["MODULE_TEMPERATURE"].max()

plant_2_max_amb_temp = plant_2_sensor_df["AMBIENT_TEMPERATURE"].max()

plant_2_max_mod_temp = plant_2_sensor_df["MODULE_TEMPERATURE"].max()



print(f"For Plant site 1 the max ambient temperature recorded was {plant_1_max_amb_temp}")

print(f"For Plant site 1 the max module temperature recorded was {plant_1_max_mod_temp}")

print(f"For Plant site 2 the max ambient temperature recorded was {plant_2_max_amb_temp}")

print(f"For Plant site 2 the max module temperature recorded was {plant_2_max_mod_temp}")
# How many inverters are there for each plant?

# This should be a simple calculation. All we need to do is create a list

# of unique values for the SOURCE_KEY in both sites' generation datasets

plant_1_inverter_count = plant_1_generation_df["SOURCE_KEY"].unique()

plant_2_inverter_count = plant_2_generation_df["SOURCE_KEY"].unique()

print(f"Plant 1 site has a total of {len(plant_1_inverter_count)} inverters.")

print(f"Plant 2 site has a total of {len(plant_2_inverter_count)} inverters.")
# What is the maximum/minimum amount of DC/AC Power generated in a time interval/day?

# To solve this problem we need to find the sum of the last daily recording of the DAILY_YIELD column 

# for each unique inverter per day. For this we can use the may and june dataframes we created earlier 

# as well as the inverter list we made to solve the problem just prior to this one

# Create empty graph for storing total daily yields by day

plant_1_daily_total_yield_may = {}

plant_1_daily_total_yield_june = {}

plant_2_daily_total_yield_may = {}

plant_2_daily_total_yield_june = {}



# Populate key values with keys that will match up to the days in our dataframe

for i in range(1, 32):

    plant_1_daily_total_yield_may[i] = 0  # Set the value of these keys to 0

    plant_1_daily_total_yield_june[i] = 0

    plant_2_daily_total_yield_may[i] = 0

    plant_2_daily_total_yield_june[i] = 0

    

# iterate through each day in plant_1_daily_recording list

for day in plant_1_daily_total_yield_may.keys():

    temp_df = plant_1_last_daily_recording[(plant_1_last_daily_recording["day"]==day) & (plant_1_last_daily_recording["month"]==5)]

    plant_1_daily_total_yield_may[day] = temp_df["DAILY_YIELD"].sum()

    temp_df = plant_1_last_daily_recording[(plant_1_last_daily_recording["day"]==day) & (plant_1_last_daily_recording["month"]==6)]

    plant_1_daily_total_yield_june[day] = temp_df["DAILY_YIELD"].sum()

    

    temp_df = plant_2_last_daily_recording[(plant_2_last_daily_recording["day"]==day) & (plant_2_last_daily_recording["month"]==5)]

    plant_2_daily_total_yield_may[day] = temp_df["DAILY_YIELD"].sum()

    temp_df = plant_2_last_daily_recording[(plant_2_last_daily_recording["day"]==day) & (plant_2_last_daily_recording["month"]==6)]

    plant_2_daily_total_yield_june[day] = temp_df["DAILY_YIELD"].sum()
# Now that we have the daily_yield for every day with a record we can find the maximum and minimum

plant_1_may_daily_yield_max = 0

plant_1_june_daily_yield_max = 0

plant_2_may_daily_yield_max = 0

plant_2_june_daily_yield_max = 0



for val in plant_1_daily_total_yield_may.values():

    if val > plant_1_may_daily_yield_max:

        plant_1_may_daily_yield_max = val



for val in plant_1_daily_total_yield_june.values():

    if val > plant_1_june_daily_yield_max:

        plant_1_june_daily_yield_max = val



for val in plant_2_daily_total_yield_may.values():

    if val > plant_2_may_daily_yield_max:

        plant_2_may_daily_yield_max = val

        

for val in plant_2_daily_total_yield_june.values():

    if val > plant_2_june_daily_yield_max:

        plant_2_june_daily_yield_max = val

        

print(f"The max daily power yield for the month of May at plant site 1 was {plant_1_may_daily_yield_max}")

print(f"The max daily power yield for the month of June at plant site 1 was {plant_1_june_daily_yield_max}")

print(f"The max daily power yield for the month of May at plant site 2 was {plant_2_may_daily_yield_max}")

print(f"The max daily power yield for the month of June at plant site 2 was {plant_2_june_daily_yield_max}")
# Now find the absolute max for each plant site

plant_1_abs_max = max([plant_1_may_daily_yield_max, plant_1_june_daily_yield_max])

plant_2_abs_max = max([plant_2_may_daily_yield_max, plant_2_june_daily_yield_max])



print(f"The absolute max daily yield at plant site 1 was {plant_1_abs_max}")

print(f"The absolute max daily yield at plant site 2 was {plant_2_abs_max}")
# For funsies let's look at the daily yield numbers visually

fig, axes = plt.subplots(2, 2, figsize=(15, 10))



axes[0,0].plot(list(plant_1_daily_total_yield_may.keys()), list(plant_1_daily_total_yield_may.values()))

axes[0,0].set_title("Plant 1 Daily Yield Totals for May")



axes[0,1].plot(list(plant_1_daily_total_yield_june.keys()), list(plant_1_daily_total_yield_june.values()))

axes[0,1].set_title("Plant 1 Daily Yield Totals for June")



axes[1,0].plot(list(plant_2_daily_total_yield_may.keys()), list(plant_2_daily_total_yield_may.values()))

axes[1,0].set_title("Plant 2 Daily Yield Totals for May")



axes[1,1].plot(list(plant_2_daily_total_yield_june.keys()), list(plant_2_daily_total_yield_june.values()))

axes[1,1].set_title("Plant 2 Daily Yield Totals for June")



plt.show()
# Which inverter (source_key) has produced maximum DC/AC power?

# To solve this problem we will need the last daily yield dataframes

# we created earlier.

plant_1_source_key_yields = {}

plant_2_source_key_yields = {}



for inverter in plant_1_last_daily_recording["SOURCE_KEY"].unique():

    temp_df = plant_1_last_daily_recording[plant_1_last_daily_recording["SOURCE_KEY"]==inverter]

    plant_1_source_key_yields[inverter] = temp_df["DAILY_YIELD"].sum()

    

for inverter in plant_2_last_daily_recording["SOURCE_KEY"].unique():

    temp_df2 = plant_2_last_daily_recording[plant_2_last_daily_recording["SOURCE_KEY"]==inverter]

    plant_2_source_key_yields[inverter] = temp_df2["DAILY_YIELD"].sum()

    

plant_1_source_key_yields
plant_1_max_yield = 0

plant_1_max_yield_inverter = ""



for inverter_val in plant_1_source_key_yields.items():

    if inverter_val[1] > plant_1_max_yield:

        plant_1_max_yield = inverter_val[1]

        plant_1_max_yield_inverter = inverter_val[0]



plant_2_max_yield = 0

plant_2_max_yield_inverter = ""



for inverter_val in plant_2_source_key_yields.items():

    if inverter_val[1] > plant_2_max_yield:

        plant_2_max_yield = inverter_val[1]

        plant_2_max_yield_inverter = inverter_val[0]

        

print(f"The plant 1 site inverter {plant_1_max_yield_inverter} had the greatest max yield of {plant_1_max_yield}")

print(f"The plant 2 site inverter {plant_2_max_yield_inverter} had the greatest max yield of {plant_2_max_yield}")
# Let's visualize that

# For funsies let's look at the daily yield numbers visually

fig, axes = plt.subplots(1, 2, figsize=(20, 10))



axes[0].plot(list(plant_1_source_key_yields.keys()), list(plant_1_source_key_yields.values()))

axes[0].set_title("Plant 1 Max Yield Totals for Individual Inverters")



axes[1].plot(list(plant_2_source_key_yields.keys()), list(plant_2_source_key_yields.values()))

axes[1].set_title("Plant 2 Max Yield Totals for Individual Inverters")



plt.show()

# Rank the inverters based on the DC/AC power they produce

# Using the data prior to this we can determine a ranking system

# for our inverters

plant_1_source_key_yields_df = pd.DataFrame(plant_1_source_key_yields.items(), columns=["Inverter_ID", "Max_Power_Yield"])

plant_2_source_key_yields_df = pd.DataFrame(plant_2_source_key_yields.items(), columns=["Inverter_ID", "Max_Power_Yield"])



# The following is an ordered list of best inverter to worst inverter based on max power yield

print("PLANT 1")

print(plant_1_source_key_yields_df.sort_values("Max_Power_Yield", ascending=False).reset_index(drop=True))

print()

print("PLANT 2")

print(plant_2_source_key_yields_df.sort_values("Max_Power_Yield", ascending=False).reset_index(drop=True))
# Are there any empty values?

# This is an easy problem to answer

print(plant_1_generation_df.isnull().sum())

print(plant_1_sensor_df.isnull().sum())

print(plant_2_generation_df.isnull().sum())

print(plant_2_sensor_df.isnull().sum())
# There are no obvious empty values in any of the dataframes.

# However, from all of the observations and graphs made above we can see that for each site there appears to be a number of days

# at the beginning of the month in May and at the end of the month in June that show no data collection, with the exception of a single

# day in May at plant site 1 on May 6th - perhaps this was simply just a test to prepare for the actual data collection process that would begin

# on May 16?
# Let's look at the data from the months prior to May and the months after June at plant site 1.

print(plant_1_generation_df[plant_1_generation_df["month"] < 5]["day"].value_counts())

print(plant_1_generation_df[plant_1_generation_df["month"] > 6]["day"].value_counts())

print(plant_1_generation_df[(plant_1_generation_df["month"] == 5)]["day"].value_counts())

print(plant_1_generation_df[(plant_1_generation_df["month"] == 6)]["day"].value_counts())
# Some interesting observations there. Data was gathered on the 6th day of every month of the year at plant site 1, with the only months 

# in which data was obtained outside of that 6th day being the months May and June. This could be for testing purposes or year-long research on a chosen day of the month. 

# If we wanted to include year-long research we would need to remove all days except for the 6th day of each month, and even then we would only be able to 

# use data from the first plant site. Whenever we are comparing data between the two plant sites I think it will be important to remove the months preceeding May and

# the months proceeding June.