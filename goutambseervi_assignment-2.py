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
plant1_generation_data = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")

plant1_weather_sensor_data = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")

plant2_generation_data = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")

plant2_weather_sensor_data = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
# Mean value of daily yeild

todays_yeild = plant1_generation_data["DAILY_YIELD"].mean()

print(todays_yeild)
# Total irradation per day

plant1_weather_sensor_data["DATE_TIME"] = pd.to_datetime(plant1_weather_sensor_data["DATE_TIME"])

todays_data = plant1_weather_sensor_data[plant1_weather_sensor_data["DATE_TIME"] < "2020-05-16 00:00:00"]

todays_data["IRRADIATION"].sum()
# Max ambient temperature

plant1_weather_sensor_data["AMBIENT_TEMPERATURE"].max()
# Max module temperature

plant1_weather_sensor_data["MODULE_TEMPERATURE"].max()
# Number of inverters for plant 1

len(plant1_generation_data["SOURCE_KEY"].unique())
# Maximum DC Power

plant1_generation_data["DC_POWER"].max()
# Minimum DC Power

plant1_generation_data["DC_POWER"].min()
plant1_generation_data[plant1_generation_data["DC_POWER"] == plant1_generation_data["DC_POWER"].max()]["SOURCE_KEY"]
plant1_generation_data.sort_values(by=["DC_POWER"], ascending=False)
# Is there missing data?

plant1_generation_data["DATE_TIME"] = pd.to_datetime(plant1_generation_data["DATE_TIME"])

plant1_generation_data["DATE_TIME"].dt.date.value_counts()
22*24*4