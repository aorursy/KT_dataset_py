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
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_wtsen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_pgen1
df_wtsen1
type(df_pgen1)
type(df_wtsen1)
df_pgen1.columns

df_wtsen1.columns
df_pgen1.info()
df_wtsen1.info()
df_pgen1.describe()
df_wtsen1.describe()
df_pgen1['DAILY_YIELD'].mean()
df_wtsen1['IRRADIATION'].count()
df_wtsen1[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']].max()
df_pgen1['SOURCE_KEY'].unique()
len(df_pgen1['SOURCE_KEY'].unique())
df_pgen1[['DC_POWER','AC_POWER']].max()

df_pgen1[['DC_POWER','AC_POWER']].min()

df_pgen1[['AC_POWER','DC_POWER']].idxmax()
df_pgen1['SOURCE_KEY'][61624]

df_pgen1.isnull()
df_wtsen1.isnull()
df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_wtsen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df_pgen2
df_wtsen2
type(df_pgen2)
type(df_wtsen2)
df_pgen2.info()
df_wtsen2.info()
df_pgen2.describe()
df_wtsen2.describe()
df_pgen2['DAILY_YIELD'].mean()
df_wtsen2['IRRADIATION'].count()
df_wtsen2[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']].max()
df_pgen2['SOURCE_KEY'].unique()
len(df_pgen2['SOURCE_KEY'].unique())
df_pgen2[['DC_POWER','AC_POWER']].max()

df_pgen2[['DC_POWER','AC_POWER']].min()

df_pgen2[['AC_POWER','DC_POWER']].idxmax()
df_pgen1['SOURCE_KEY'][41423]

df_pgen2.isnull()
df_wtsen2.isnull()