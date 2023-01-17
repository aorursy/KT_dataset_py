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
df_pow_gen1 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")   #Module Function
# df_pow_gen1 is a data frame object
df_wthr_gen1 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
df_pow_gen2 = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")
df_wthr_gen2 = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
type(df_pow_gen1)
df_pow_gen1.describe() 
df_pow_gen2.describe() 
df_wthr_gen1.describe()
df_wthr_gen2.describe()
df_pow_gen1['DAILY_YIELD'].mean()
df_pow_gen2['DAILY_YIELD'].mean()
df_wthr_gen1['IRRADIATION'].count()
df_wthr_gen2['IRRADIATION'].count()
df_wthr_gen1[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']].max()
df_wthr_gen2[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']].max()
len(df_pow_gen1['SOURCE_KEY'].unique())
len(df_pow_gen2['SOURCE_KEY'].unique())
df_pow_gen1[['AC_POWER','DC_POWER']].max()
df_pow_gen1[['AC_POWER','DC_POWER']].min()
df_pow_gen1[['AC_POWER','DC_POWER']].idxmax()
df_pow_gen1['SOURCE_KEY'][61624]
df_pow_gen2[['AC_POWER','DC_POWER']].idxmax()
df_pow_gen1['SOURCE_KEY'][41423]
df_pow_gen1.isnull()
df_pow_gen2.isnull()
df_wthr_gen1.isnull()
df_wthr_gen2.isnull()
