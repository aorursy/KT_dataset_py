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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plant1=pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")

print(plant1.head())
plant2=pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
print(plant2.head())
plant1.isnull().sum()
plant2.isnull().sum()
plant1.info()
plant2.info()
AC_POWER_df2=pd.DataFrame(plant1["AC_POWER"])
#print(AC_POWER_df2)


plant1_df3=pd.DataFrame(plant1["DC_POWER"])
#print(plant1_df3)

DATE_TIME_df3=pd.to_datetime(plant1["DATE_TIME"])
#print(DATE_TIME_df3)

f,axarr = plt.subplots(2,figsize=(10,7))
axarr[0].plot(DATE_TIME_df3,AC_POWER_df2, label="AC power variation",color="r",linewidth=0.8)
axarr[0].set_title("AC POWER VARIATION")
axarr[1].plot(DATE_TIME_df3,plant1_df3,label="DC power variation",color="b",linewidth=0.8)
axarr[1].set_title("DC POWER VARIATION")


plt.xlabel('Month Number')
plt.ylabel('Sales units in number')
plt.show()
# FOR DATETIME VS TOTAL YIELD 

TOTAL_YIELD_df4=plant1["TOTAL_YIELD"]
DATE_TIME_df3=pd.to_datetime(plant1["DATE_TIME"])

plt.figure(figsize=(15,10))
plt.bar(DATE_TIME_df3,TOTAL_YIELD_df4,color="g")
plt.title("Total Yield variation")
plt.xlabel("TOTAL_YIELD")
plt.ylabel("TOTAL_YIELD")
plt.grid(True, linewidth= 1, linestyle="--",axis="y",alpha=0.7)

plt.show()

#FOR PLANT 2
DATE_TIME_df3=pd.to_datetime(plant2["DATE_TIME"])
AMBIENT_TEM=plant2["AMBIENT_TEMPERATURE"]
MODULE_TEM=plant2["MODULE_TEMPERATURE"]
mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']      
columns = ["AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION"]


fig,ax=plt.subplots(1,1,figsize=(15,8),dpi=80)
ax.fill_between(DATE_TIME_df3,AMBIENT_TEM,label=columns[0],alpha=0.40,
                color=mycolors[0],linewidth=1)
ax.fill_between(DATE_TIME_df3,MODULE_TEM,label=columns[1],alpha=0.35,
                color=mycolors[1],linewidth=1)

ax.set_xlabel("Date")
ax.set_ylabel("Temperatures")
ax.legend(loc='upper right')
ax.set_title('TEMPERATURE VARIATION')

ax.grid(True, linewidth= 1, linestyle="--")

plt.show()
RADIATION=plant2["IRRADIATION"]

plt.figure(figsize=(15,8))
plt.bar(DATE_TIME_df3, RADIATION, width= 0.2, label = "RADIATION DATA",color="orange", align='edge')

plt.xlabel('DATE',fontsize=10)
plt.ylabel('RADIATION',fontsize=10)
plt.legend(loc='upper left',fontsize=15)
plt.title('VARIATION OF RADIATION')

plt.grid(True, linewidth= 1, linestyle="--")

plt.show()
