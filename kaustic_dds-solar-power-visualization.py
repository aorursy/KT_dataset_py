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
import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use("seaborn-darkgrid")
df_p1_gen = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")

df_p1_sen = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")

df_p2_gen = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")

df_p2_sen = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
df_p1_gen["DC_POWER"] = df_p1_gen["DC_POWER"].apply(lambda x: x / 10.)
print(df_p1_gen.head())

print(df_p2_gen.head())
# Transform to native datetime formats for easier manipulation



df_p1_gen["DATE_TIME"] = pd.to_datetime(df_p1_gen["DATE_TIME"], format="%d-%m-%Y %H:%S")

df_p2_gen["DATE_TIME"] = pd.to_datetime(df_p2_gen["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")



transform_to_date = lambda x: x.date()

transform_to_time = lambda x: x.time()



df_p1_gen["DATE"] = df_p1_gen["DATE_TIME"].apply(transform_to_date)

df_p1_gen["TIME"] = df_p1_gen["DATE_TIME"].apply(transform_to_time)



df_p2_gen["DATE"] = df_p2_gen["DATE_TIME"].apply(transform_to_date)

df_p2_gen["TIME"] = df_p2_gen["DATE_TIME"].apply(transform_to_time)
# Transform to native datetime formats for easier manipulation



df_p1_sen["DATE_TIME"] = pd.to_datetime(df_p1_sen["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")

df_p2_sen["DATE_TIME"] = pd.to_datetime(df_p2_sen["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")



df_p1_sen["DATE"] = df_p1_sen["DATE_TIME"].apply(transform_to_date)

df_p1_sen["TIME"] = df_p1_sen["DATE_TIME"].apply(transform_to_time)



df_p2_sen["DATE"] = df_p2_sen["DATE_TIME"].apply(transform_to_date)

df_p2_sen["TIME"] = df_p2_sen["DATE_TIME"].apply(transform_to_time)
# Let's check DC_POWER and AC_POWER by the timestamp



plt.figure(figsize=(12, 8))



plt.plot(

    df_p1_gen["DATE_TIME"].unique(),

    df_p1_gen.groupby("DATE_TIME")["DC_POWER"].sum(),

    label="DC Power",

)

plt.plot(

    df_p1_gen["DATE_TIME"].unique(),

    df_p1_gen.groupby("DATE_TIME")["AC_POWER"].sum(),

    label="AC_POWER",

)



plt.legend()

plt.xlabel("Date and time")

plt.show()
# Boilerplate



def plt_init(figsize=(12, 8)):

    _, ax = plt.subplots(1, 1, figsize=figsize)

    return ax
# Plotting irradiation against time



ax = plt_init()



ax.plot(

    df_p1_sen["DATE_TIME"], df_p1_sen["IRRADIATION"], "o--",

    alpha=0.75, label="Irradiation",

)



plt.legend()

plt.show()
# Plotting module temperature and ambient temperature against time



ax = plt_init()



ax.plot(

    df_p1_sen["DATE_TIME"], df_p1_sen["MODULE_TEMPERATURE"],

    label="Module Temperature",

)

ax.plot(

    df_p1_sen["DATE_TIME"], df_p1_sen["AMBIENT_TEMPERATURE"],

    label="Ambient Temperature"

)



plt.legend()

plt.show()
# Now let's look at irradiation and difference between module temp and ambient temp



ax = plt_init()



ax.plot(df_p1_sen["DATE_TIME"], df_p1_sen["IRRADIATION"], label="Irradiation")

ax.plot(

    df_p1_sen["DATE_TIME"],

    df_p1_sen["MODULE_TEMPERATURE"] - df_p1_sen["AMBIENT_TEMPERATURE"],

    label="Module temp. - Ambient temp."

)



plt.legend()
# Let's magnify



ax = plt_init()



ax.plot(df_p1_sen["DATE_TIME"], df_p1_sen["IRRADIATION"] * 20., label="Irradiation * 20")

ax.plot(

    df_p1_sen["DATE_TIME"],

    df_p1_sen["MODULE_TEMPERATURE"] - df_p1_sen["AMBIENT_TEMPERATURE"],

    label="Module temp. - Ambient temp."

)



plt.legend()

plt.show()
# Daily Yield vs Date time



ax = plt_init()



ax.plot(df_p1_gen["DATE_TIME"], df_p1_gen["DAILY_YIELD"], "o--", alpha=0.5)



plt.show()
# Daily Yield vs Date Time per inverter



ax = plt_init((16, 10))



for key, rows in df_p1_gen.groupby("SOURCE_KEY"):

    ax.plot(

        rows["DATE_TIME"], rows["DAILY_YIELD"], "o--", alpha=0.5, label=key

    )



plt.legend()

plt.show()
# Now to the second plant



plt.figure(figsize=(12, 8))



plt.plot(

    df_p2_gen["DATE_TIME"].unique(),

    df_p2_gen.groupby("DATE_TIME")["DC_POWER"].sum(),

    label="DC Power",

)

plt.plot(

    df_p2_gen["DATE_TIME"].unique(),

    df_p2_gen.groupby("DATE_TIME")["AC_POWER"].sum(),

    label="AC_POWER",

)



plt.legend()

plt.xlabel("Date and time")

plt.show()
ax = plt_init()



ax.plot(

    df_p2_sen["DATE_TIME"], df_p2_sen["IRRADIATION"], "o--",

    alpha=0.75, label="Irradiation",

)



plt.legend()

plt.show()
ax = plt_init()



ax.plot(

    df_p2_sen["DATE_TIME"], df_p2_sen["MODULE_TEMPERATURE"],

    label="Module Temperature",

)

ax.plot(

    df_p2_sen["DATE_TIME"], df_p2_sen["AMBIENT_TEMPERATURE"],

    label="Ambient Temperature"

)



plt.legend()

plt.show()
ax = plt_init()



ax.plot(df_p2_sen["DATE_TIME"], df_p2_sen["IRRADIATION"] * 20., label="Irradiation * 20")

ax.plot(

    df_p2_sen["DATE_TIME"],

    df_p2_sen["MODULE_TEMPERATURE"] - df_p2_sen["AMBIENT_TEMPERATURE"],

    label="Module temp. - Ambient temp."

)



plt.legend()

plt.show()
ax = plt_init()



ax.plot(df_p2_gen["DATE_TIME"], df_p2_gen["DAILY_YIELD"], "o--", alpha=0.5)



plt.show()
ax = plt_init((16, 10))



for key, rows in df_p2_gen.groupby("SOURCE_KEY"):

    ax.plot(

        rows["DATE_TIME"], rows["DAILY_YIELD"], "o--", alpha=0.5, label=key

    )



ax.legend()

plt.show()
_, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))





ax1.plot(

    df_p1_gen["DC_POWER"], df_p1_gen["AC_POWER"], "bo",

    alpha=0.5, label="Plant1 | AC Power vs. DC Power",

)



ax2.plot(

    df_p2_gen["DC_POWER"], df_p2_gen["AC_POWER"], "go",

    alpha=0.5, label="Plant2 | AC Power vs. DC Power",

)



ax1.legend()

ax2.legend()



plt.show()
# Module Temperature vs. Ambient Temperature



_, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))





ax1.plot(

    df_p1_sen["AMBIENT_TEMPERATURE"], df_p1_sen["MODULE_TEMPERATURE"], "bo",

    alpha=0.5,

)



ax1.set_title("Plant 1 | Module Temperature vs Ambient Temperature")

ax1.set_xlabel("Ambient Temperature")

ax1.set_ylabel("Module Temperature")

ax1.legend()



ax2.plot(

    df_p2_sen["AMBIENT_TEMPERATURE"], df_p2_sen["MODULE_TEMPERATURE"], "go",

    alpha=0.5,

)



ax2.set_title("Plant 2 | Module Temperature vs Ambient Temperature")

ax2.set_xlabel("Ambient Temperature")

ax2.set_ylabel("Module Temperature")

ax2.legend()



plt.show()
# (Module Temperature - Ambient Temperature) vs. Irradiation



_, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))





ax1.plot(

    df_p1_sen["IRRADIATION"],

    df_p1_sen["MODULE_TEMPERATURE"] - df_p1_sen["AMBIENT_TEMPERATURE"],

    "bo",

    alpha=0.5,

)



ax1.set_title("Plant 1 | Module Temp - Ambient Temp vs. Irradiation")

ax1.set_xlabel("Irradiation")

ax1.set_ylabel("$T_M - T_A$")

ax1.legend()



ax2.plot(

    df_p2_sen["IRRADIATION"],

    df_p2_sen["MODULE_TEMPERATURE"] - df_p2_sen["AMBIENT_TEMPERATURE"],

    "go",

    alpha=0.5,

)



ax2.set_title("Plant 2 | Module Temp - Ambient Temp vs. Irradiation")

ax2.set_xlabel("Irradiation")

ax2.set_ylabel("$T_M - T_A$")

ax2.legend()



plt.show()
# The mean of AC_POWER per day for each SOURCE_KEY



ac_summary_p1 = df_p1_gen.groupby(["SOURCE_KEY", "DATE"]).agg(

    AC_MEAN=("AC_POWER", "mean"), INV=("SOURCE_KEY", "max")

)



ax = plt_init((10, 12))



sns.boxplot(x="AC_MEAN", y="INV", data=ac_summary_p1)



ax.set(xlabel="Mean AC Power produced per day", ylabel="Inverters")



plt.show()
ac_summary_p2 = df_p2_gen.groupby(["SOURCE_KEY", "DATE"]).agg(

    AC_MEAN=("AC_POWER", "mean"), INV=("SOURCE_KEY", "max")

)



ax = plt_init((10, 12))



sns.boxplot(x="AC_MEAN", y="INV", data=ac_summary_p2)



ax.set(xlabel="Mean AC Power produced per day", ylabel="Inverters")



plt.show()