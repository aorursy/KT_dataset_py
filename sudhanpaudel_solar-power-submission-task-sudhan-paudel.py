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
df_pgen1=pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")
df_psen1=pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
df_pgen2=pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")
df_psen2=pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")

df_pgen1
df_pgen1.info()
df_psen1
df_psen1.info()
df_pgen2[['PLANT_ID',"DATE_TIME",'AC_POWER']].value_counts().sort_index()
df_pgen1.describe()
len(df_psen1)
(df_psen1.columns)
len(df_psen1.columns)
df_psen1.describe()
df_psen1["IRRADIATION"].max()
df_pgen1["AC_POWER"][670:5678].mean()
df_pgen1.shape
df_pgen1.shape[0]  #takes rows
df_pgen1.shape[1]  #takes columns
df_psen1
#What is the mean value of daily yield?

print("The mean value of daily yield for plant 1 is " + str(df_pgen1["DAILY_YIELD"].mean()))
print("The mean value of daily yield for plant 2 is " + str(df_pgen2["DAILY_YIELD"].mean()))
#What is the total irradiation per day?

# for this we need to separate the irradiation according to dates and then group by day to get the daily sum.

#For plant1

#DATE_TIME column is object type. We need to change to datetime type.
df_psen1["DATE_TIME"]=pd.to_datetime(df_psen1["DATE_TIME"], format='%Y-%m-%d %H:%M:%S')  
df_psen1["DATE_TIME"].dtype # checking data type
#creating a new column for date for separation according to dates
df_psen1["DATE"]=df_psen1["DATE_TIME"].dt.date
df_psen1                               
group1=df_psen1.groupby(["DATE"]).sum()      # sum of all attributes with respect to date
group1[["IRRADIATION"] ]# we need only IRRADIATION column from group1
#For plant2

df_psen2["DATE_TIME"]=pd.to_datetime(df_psen2["DATE_TIME"], format='%Y-%m-%d %H:%M:%S')  
df_psen2["DATE_TIME"].dtype
df_psen2["DATE"]=df_psen2["DATE_TIME"].dt.date
df_psen2    
group2=df_psen2.groupby(["DATE"]).sum()
group2[["IRRADIATION"] ]
#What is the max ambient and module temperature?
#For plant1
print("The maximum value of ambient temperature for plant 1 is " + str(df_psen1["AMBIENT_TEMPERATURE"].max()))
print("The minimum value of ambient temperature for plant 1 is " + str(df_psen1["AMBIENT_TEMPERATURE"].min()))
print("The maximum value of module temperature for plant 1 is " + str(df_psen1["MODULE_TEMPERATURE"].max()))
print("The minimum value of module temperature for plant 1 is " + str(df_psen1["MODULE_TEMPERATURE"].min()))

#For plant2
print("The maximum value of ambient temperature for plant 2 is " + str(df_psen2["AMBIENT_TEMPERATURE"].max()))
print("The minimum value of ambient temperature for plant 2 is " + str(df_psen2["AMBIENT_TEMPERATURE"].min()))
print("The maximum value of module temperature for plant 2 is " + str(df_psen2["MODULE_TEMPERATURE"].max()))
print("The minimum value of module temperature for plant 2 is " + str(df_psen2["MODULE_TEMPERATURE"].min()))
# How many inverters are there for each plant? 

print("There are "+str(len(df_pgen1["SOURCE_KEY"].unique()))+ " unique inverters for plant 1.")
print("There are "+str(len(df_pgen2["SOURCE_KEY"].unique()))+ " unique inverters for plant 2.")
df_pgen1
# What is the maximum/minimum amount of DC/AC Power generated in a time interval/day?

# we need to group by DC/AC Power and then find the minimum/maximum

#For plant1    found the maximun AC power for a day

#DATE_TIME column is object type. We need to change to datetime type.
df_pgen1["DATE_TIME"]=pd.to_datetime(df_pgen1["DATE_TIME"], format='%d-%m-%Y %H:%M')  
# storing date to different column
df_pgen1["DATE"]=df_pgen1['DATE_TIME'].dt.date
# displaying to check 
df_pgen1 
# grouping according to date to find the value for a particular day
group3=df_pgen1.groupby(["DATE"]).max()
#Displaying the AC_POWER column
group3[[ "AC_POWER"]]
#For plant2   found the minimun DC power for a day
df_pgen2["DATE_TIME"]=pd.to_datetime(df_pgen2["DATE_TIME"], format='%Y-%m-%d %H:%M:%S')  
df_pgen2["DATE"]=df_pgen2['DATE_TIME'].dt.date
df_pgen2
group4=df_pgen2.groupby(["DATE"]).min()
group4[["DC_POWER"]]
# Which inverter (source_key) has produced maximum DC/AC power? 
# For plant1
print(df_pgen1[['SOURCE_KEY','AC_POWER']][df_pgen1.AC_POWER == df_pgen1.AC_POWER.max()])
# For plant2
print(df_pgen2[['SOURCE_KEY','AC_POWER']][df_pgen2.AC_POWER == df_pgen2.AC_POWER.max()])
#Rank the inverters based on the DC/AC power they produce

# For plant1
group5=df_pgen1.groupby(["SOURCE_KEY"]).sum()
group5

group5.iloc[:,[2,3]].rank()
# For plant2
group6=df_pgen2.groupby(["SOURCE_KEY"]).sum()
group6
group6.iloc[:,[2,3]].rank()
# Is there any missing data?

df_pgen1.isnull()
df_pgen2.isnull()
df_psen1.isnull()
df_psen2.isnull()
# There seems that no data is missing.