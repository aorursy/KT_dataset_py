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
df=pd.read_csv(r"../input/solar-power-plant-datasets/Plant_1_Generation_Data.csv")
df1=pd.read_csv(r"../input/solar-power-plant-datasets/Plant_1_Weather_Sensor_Data.csv")
df2=pd.read_csv(r"../input/solar-power-plant-datasets/Plant_2_Generation_Data.csv")
df3=pd.read_csv(r"../input/solar-power-plant-datasets/Plant_2_Weather_Sensor_Data.csv")
df.head()
missing_values_count = df.isnull().sum()
missing_values_count[0:9]
total_cells = np.product(df.shape)
total_missing = missing_values_count.sum()
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)
df.duplicated().sum()
df.duplicated().any()
df.columns
df.count(axis=0)
df.count(axis=1)
df['DAILY_YIELD'].mean()
df['SOURCE_KEY'].count()
df['AC_POWER'].max()
df["DC_POWER"].max()
df['AC_POWER'].min()
df['DC_POWER'].min()
df["SOURCE_KEY"][df.AC_POWER==df["AC_POWER"].max()]
df["SOURCE_KEY"][df.DC_POWER==df["DC_POWER"].max()]
# Plant 1 wether sensor data analysis#

df1.head()
df1.isnull().sum()
df1=df1.dropna()
df1.head()
df1.isnull().sum()
df1["IRRADIATION"].mean()
df1["AMBIENT_TEMPERATURE"].max()
df1['MODULE_TEMPERATURE'].max()
# plant 2 #
df2.head()
df2.columns
df2.count(axis=0)
df2.count(axis=1)
df2.isnull().sum()
df2.duplicated().sum()
df2['DAILY_YIELD'].mean()
df2["AC_POWER"].count()
df2["DC_POWER"].count()
df2['SOURCE_KEY'].count()
df2["AC_POWER"].max()
df2['DC_POWER'].max()
df2["AC_POWER"].min()
df2["DC_POWER"].min()
df2['SOURCE_KEY'][df2.AC_POWER==df2["AC_POWER"].max()]
df2['SOURCE_KEY'][df2.DC_POWER==df2["DC_POWER"].max()]
# PLANT 2 WETHER SENSOR DATA #
df3.columns
df3.count(axis=0)
df3.isnull().sum()
df3.duplicated().sum()
df3['IRRADIATION'].mean()
df3['AMBIENT_TEMPERATURE'].max()
df3['MODULE_TEMPERATURE'].max()
