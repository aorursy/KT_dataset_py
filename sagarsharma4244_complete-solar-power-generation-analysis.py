# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:

        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import Data
power1 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv")
weather1 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")

power2 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv")
weather2 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")

# Look at data 
power2.head()
weather2.head()
# Notice the total entries (Not 34)
power2.info()
weather2.info()
# Let's visualize it anyway.
fig, ax = plt.subplots(2,1, figsize=(20,8))
ax[0].set_title("POWER PLANT 2 DAILY CAPACITY")  
sns.lineplot(data=[power2["DC_POWER"],power2["AC_POWER"]], ax=ax[0], palette="tab20", linewidth=1)
ax[1].set_title("WEATHER ON POWER PLANT 2")  
sns.lineplot(data=[weather2["AMBIENT_TEMPERATURE"],weather2["MODULE_TEMPERATURE"],weather2["IRRADIATION"]], ax=ax[1], palette="tab20", linewidth=2.5)
##### Combine the data on the basis of dates only using "DATE_TIME" column #####

## 1. Create a DATE (date format) column using "DATE_TIME"(object format) column 
df_date= weather2["DATE_TIME"].str.split()
x=[]
for d in df_date:
    val= d[0]
    x.append(val)
weather2["DATE"]= pd.DataFrame(x)
weather2.drop(["DATE_TIME"], axis=1, inplace=True)


df_date= power2["DATE_TIME"].str.split()
x=[]
for d in df_date:
    val= d[0]
    x.append(val)
power2["DATE"]= pd.DataFrame(x)
power2.drop(["DATE_TIME"], axis=1, inplace=True)



## 2. Group by date and MEan the rest
groupMeanWeather2 = weather2.groupby("DATE").max().reset_index()
groupMeanPower2 = power2.groupby("DATE").max().reset_index()

# Look at the 34 Entries for both the tables
groupMeanWeather2.info()
groupMeanPower2.info()
def getNormalized(num):
    return num/(num.max()+num.min())
fig, ax = plt.subplots(1,1, figsize=(20,4))
w_amb = getNormalized(groupMeanWeather2["AMBIENT_TEMPERATURE"])
mod_temp = getNormalized(groupMeanWeather2["MODULE_TEMPERATURE"])
w_irr = getNormalized(groupMeanWeather2["IRRADIATION"]) 

sns.lineplot(data=[w_amb,mod_temp,w_irr], ax=ax, palette="tab20", linewidth=1.5,dashes=False).set_title("Scaled Weather parameters)")
plt.show()
fig, ax = plt.subplots(1,1, figsize=(20,4))
p_dc = getNormalized(groupMeanPower2["DC_POWER"])
p_ac = getNormalized(groupMeanPower2["AC_POWER"])
p_dy = getNormalized(groupMeanPower2["DAILY_YIELD"])
p_ty = getNormalized(groupMeanPower2["TOTAL_YIELD"]) 

sns.lineplot(data=[p_ac,p_dc-0.01,p_dy,p_ty], ax=ax, palette="tab20", linewidth=1.5,dashes=False).set_title("Scaled Power Parameters")
plt.show()
fig, ax = plt.subplots(1,1, figsize=(20,8))

sns.lineplot(data=[p_ac,w_amb,mod_temp,w_irr], ax=ax, palette="tab20", linewidth=1,dashes=False).set_title("Scaled Weather Vs Power parameters")
plt.show()



sns.jointplot(x=p_ac,y=w_irr ,kind="reg")
sns.jointplot(x=w_amb,y=mod_temp, kind="kde")
newdata =  pd.DataFrame({"AC": p_ac.values,
                        "AMB": w_amb.values,
                        "MOD":mod_temp.values,
                        "IRR":w_irr.values})

newdata.describe()
sns.pairplot(data =newdata)
# drawthis = newdata.pivot("DATE", "AMBIENT_TEMPERATURE", "DC_POWER")

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(newdata, annot=True, fmt="d", linewidths=.5, ax=ax)
# Group by date and Sum the rest
groupweather2 = weather2.groupby("DATE").sum().reset_index()
grouppower2 = power2.groupby("DATE").sum().reset_index()
# Group beydate and mean the reset 
groupweather2mean = weather2.groupby("DATE").mean().reset_index()
grouppower2mean = power2.groupby("DATE").mean().reset_index()
groupweather2mean.info()
grouppower2mean.info()
groupweather2.info()
grouppower2.info()
# Merged onDate
merged2 = pd.merge(groupweather2,grouppower2,on="DATE")
merged2.info()

# Mean Merged Date
meanmerged2 = pd.merge(groupweather2mean,grouppower2mean,on="DATE")
meanmerged2.info()
meanmerged2.head()
meanmerged2['DATE'] = pd.to_datetime(meanmerged2['DATE'], format='%Y-%m-%d') 
meanmerged2.info()

fig, ax = plt.subplots(1,1, figsize=(20,4))
p_dc = (meanmerged2["DC_POWER"]/280)-0.4
p_ac = (meanmerged2["AC_POWER"]/280)-0.4

p_dy = meanmerged2["DAILY_YIELD"]/10000
w_amb = ((meanmerged2["AMBIENT_TEMPERATURE"]/35 )-0.6)*3
mod_temp = ((meanmerged2["MODULE_TEMPERATURE"]/60 ) - 0.3)*3
w_irr = meanmerged2["IRRADIATION"]*3
sns.lineplot(data=[p_ac,p_dc,w_amb,mod_temp,w_irr], ax=ax, palette="tab20", linewidth=2.5)
plt.show()
def getNormalized(num):
    return num/(num.max()+num.min())
    

fig, ax = plt.subplots(1,1, figsize=(20,4))
p_dc = getNormalized(meanmerged2["DC_POWER"])
p_ac = getNormalized(meanmerged2["AC_POWER"])
# p_dy = meanmerged2["DAILY_YIELD"]/10000
w_amb = getNormalized(meanmerged2["AMBIENT_TEMPERATURE"])
mod_temp = getNormalized(meanmerged2["MODULE_TEMPERATURE"])
w_irr = getNormalized(meanmerged2["IRRADIATION"]) 
p_dy = getNormalized(meanmerged2["DAILY_YIELD"])
p_ty = getNormalized(meanmerged2["TOTAL_YIELD"]) 

sns.lineplot(data=[p_ac,p_dc,w_amb,mod_temp,w_irr,p_dy,p_ty], ax=ax, palette="tab20", linewidth=1.5,dashes=False)
plt.show()
meanmerged2.describe()

p_dy = getNormalized(meanmerged2["DAILY_YIELD"])
p_ty =  getNormalized(meanmerged2["TOTAL_YIELD"]) 

# w_amb = ((meanmerged2["AMBIENT_TEMPERATURE"]/35 )-0.6)*3
# mod_temp = ((meanmerged2["MODULE_TEMPERATURE"]/60 ) - 0.3)*3
# w_irr = meanmerged2["IRRADIATION"]*3
sns.lineplot(data=[p_dy,p_ty], ax=ax, palette="tab20", linewidth=2.5)
plt.show()
meanScaledData = pd.DataFrame({"DC_POWER":p_dc,
                                   "AC_POWER":p_ac,
                                   "AMBIENT_TEMPERATURE":w_amb,
                                   "MODULE_TEMPERATURE":mod_temp,
                                   "IRRADIATION":w_irr})
meanScaledData.info()
meanScaledData.describe()
sns.pairplot(data = meanScaledData)
fig_dims = (8, 8)
fig, ax = plt.subplots(figsize=fig_dims)

corr = meanScaledData.corr()
sns.heatmap(corr,ax = ax, annot=True)

plt.show()
# fig, ax = plt.subplots(1,2, figsize=(8,8))
sns.jointplot(x="IRRADIATION",y="DC_POWER",data=meanScaledData ,kind="reg")
sns.jointplot(x="AMBIENT_TEMPERATURE",y="MODULE_TEMPERATURE",data=meanScaledData, kind="kde")
# plt.show()

