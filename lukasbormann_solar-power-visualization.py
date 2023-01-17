import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# load data
plant_1_g = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")
plant_1_w = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
plant_2_g = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")
plant_2_w = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
# convert date format to match other tables
plant_1_g['DATE_TIME'] = [ x[6:10] + x[2:6] + x[:2] + x[10:] + ":00" for x in plant_1_g['DATE_TIME']]

plant_1_g['DATE_TIME'] = pd.to_datetime(plant_1_g['DATE_TIME'], format="%Y-%m-%d %H:%M:%S")
plant_1_w['DATE_TIME'] = pd.to_datetime(plant_1_w['DATE_TIME'], format="%Y-%m-%d %H:%M:%S")
plant_2_g['DATE_TIME'] = pd.to_datetime(plant_2_g['DATE_TIME'], format="%Y-%m-%d %H:%M:%S")
plant_2_w['DATE_TIME'] = pd.to_datetime(plant_2_w['DATE_TIME'], format="%Y-%m-%d %H:%M:%S")

# add plants together
plant_g = plant_1_g.append(plant_2_g)
plant_w = plant_1_w.append(plant_2_w)
print(plant_g.isna().sum(), "\n"); print(plant_w.isna().sum())
plant_g_day_yield = plant_g[plant_g["DATE_TIME"].dt.time == pd.to_datetime("23:45:00", format="%H:%M:%S").time()]
plant_w_day_yield = plant_w[plant_w["DATE_TIME"].dt.time == pd.to_datetime("23:45:00", format="%H:%M:%S").time()]

# merge both tables to evaluate weather features
plant_gw_day = plant_g_day_yield.merge(plant_w_day_yield, how="inner", on=["DATE_TIME", "PLANT_ID"])
plot_temp = sns.lmplot(data=plant_gw_day, x="AMBIENT_TEMPERATURE", y="DAILY_YIELD")
# plot_temp.set(xlim=(20, 30))
ax = sns.scatterplot(data=plant_gw_day, x=plant_gw_day["DATE_TIME"].dt.day, y="DAILY_YIELD", hue="AMBIENT_TEMPERATURE")
# Put a legend to the right side
ax.legend(loc='center right', bbox_to_anchor=(1.5, 0.78), ncol=1)
plt.xlabel("DAY")

plt.show()
fig, ax = plt.subplots(figsize=(12,9))

sns.boxplot(data=plant_g_day_yield, x="DAILY_YIELD", y="SOURCE_KEY", ax=ax)
plant_1_g_error = plant_1_g.copy()

plant_1_g_error["TIME"] = plant_1_g_error["DATE_TIME"].dt.time
plant_1_g_error = plant_1_g_error.groupby(["TIME", "SOURCE_KEY"])["DC_POWER"].mean().unstack()

cmap = sns.color_palette("Spectral", n_colors=12)

fig,ax = plt.subplots(dpi=100)
plant_1_g_error.plot(ax=ax, color=cmap)
ax.legend(loc='center right', bbox_to_anchor=(1.5, 0.78), ncol=1)
plt.ylabel("DC_POWER_AVERAGE")
plt.show()