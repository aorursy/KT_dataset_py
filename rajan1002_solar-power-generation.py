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


import seaborn as sns

from sklearn.preprocessing import LabelEncoder 

from statsmodels.formula.api import ols
plant1=pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv")

plant2=pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv")

sensor1=pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")

sensor2=pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
plant1.info()
plant2.info()
sensor1.info()
sensor2.info()
plant1["SOURCE_KEY"].nunique()
plant2["SOURCE_KEY"].nunique()
plant1["SOURCE_KEY"].value_counts()
plant2["SOURCE_KEY"].value_counts()
plant1.groupby("SOURCE_KEY")["DC_POWER","AC_POWER","DAILY_YIELD","TOTAL_YIELD"].mean()
plant2.groupby("SOURCE_KEY")["DC_POWER","AC_POWER","DAILY_YIELD","TOTAL_YIELD"].mean()
plant1["TOTAL_YIELD"]=(plant1["TOTAL_YIELD"]-min(plant1["TOTAL_YIELD"]))/(max(plant1["TOTAL_YIELD"])-min(plant1["TOTAL_YIELD"]))



plant1["DAILY_YIELD"]=(plant1["DAILY_YIELD"]-min(plant1["DAILY_YIELD"]))/(max(plant1["DAILY_YIELD"])-min(plant1["DAILY_YIELD"]))



plant1["DC_POWER"]=(plant1["DC_POWER"]-min(plant1["DC_POWER"]))/(max(plant1["DC_POWER"])-min(plant1["DC_POWER"]))



plant1["AC_POWER"]=(plant1["AC_POWER"]-min(plant1["AC_POWER"]))/(max(plant1["AC_POWER"])-min(plant1["AC_POWER"]))
sns.lmplot(x="DC_POWER",y="TOTAL_YIELD",hue="SOURCE_KEY",col="SOURCE_KEY",height=3,col_wrap=4,data=plant1,fit_reg=False)
sns.lmplot(x="AC_POWER",y="TOTAL_YIELD",hue="SOURCE_KEY",col="SOURCE_KEY",height=3,col_wrap=4,data=plant1,fit_reg=False)
sns.lmplot(x="DC_POWER",y="DAILY_YIELD",hue="SOURCE_KEY",col="SOURCE_KEY",height=3,col_wrap=4,data=plant1,fit_reg=False)
sns.lmplot(x="AC_POWER",y="DAILY_YIELD",hue="SOURCE_KEY",col="SOURCE_KEY",height=3,col_wrap=4,data=plant1,fit_reg=False)
plant1["DATE_TIME"]=pd.to_datetime(plant1.DATE_TIME)

plant1=plant1.assign(minute=plant1.DATE_TIME.dt.minute,hour=plant1.DATE_TIME.dt.hour,day=plant1.DATE_TIME.dt.day,month=plant1.DATE_TIME.dt.month)
sns.countplot(x="month",data=plant1)
plant1["month"].value_counts()
plant1.drop("DATE_TIME",axis=1,inplace=True)

plant1.drop("PLANT_ID",axis=1,inplace=True)
plant1.groupby("month")["DC_POWER","AC_POWER","DAILY_YIELD","TOTAL_YIELD"].mean()
key_le=LabelEncoder()

plant1["SOURCE_KEY"]=key_le.fit_transform(plant1["SOURCE_KEY"])
model=ols('TOTAL_YIELD~minute+hour+day+month+hour*day+minute*hour+SOURCE_KEY+day*minute+month*day+minute*month+month*hour-1',data=plant1).fit()

model.summary()
model=ols('DAILY_YIELD~hour+day+month+SOURCE_KEY-1',data=plant1).fit()

model.summary()

sensor1["DATE_TIME"]=pd.to_datetime(sensor1["DATE_TIME"])

sensor1=sensor1.assign(minute=sensor1.DATE_TIME.dt.minute, hour=sensor1.DATE_TIME.dt.hour, day=sensor1.DATE_TIME.dt.day, month=sensor1.DATE_TIME.dt.month)







model=ols('IRRADIATION~AMBIENT_TEMPERATURE+MODULE_TEMPERATURE+day',data=sensor1).fit()

model.summary()