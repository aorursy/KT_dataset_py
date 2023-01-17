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

df1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

df3 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df4 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df1.head() # Displays Dataset of Generation data
df2.head() # Dataset of Weather Sensor Data
df1.columns
df2.columns
df3.columns
df4.columns
df1.nunique()
df2.nunique()
df1.describe()
df2.describe()
dy_mean = df1['DAILY_YIELD'].mean()

print(f'The mean value of Daily Yield is {dy_mean}')
dy_ambient_temp = df3['AMBIENT_TEMPERATURE'].max()

print(f'The maximum ambient temperature is {dy_ambient_temp}')
dy_module_temp = df3['MODULE_TEMPERATURE'].max()

print(f'The maximum module temperature is {dy_module_temp}')
df1['SOURCE_KEY'].nunique()
dy_max_dc = df1['DC_POWER'].max()

print(f'The Maximum DC Power Generated is {dy_max_dc}')
dy_min_dc = df1['DC_POWER'].min()

print(f'The Minimum DC Power Generated is {dy_min_dc}')
dy_max_ac = df1['AC_POWER'].max()

print(f'The Maximum AC Power Generated is {dy_max_ac}')
dy_min_ac = df1['AC_POWER'].min()

print(f'The Minimum AC Power Generated is {dy_min_ac}')
df1[df1['DC_POWER']==df1['DC_POWER'].max()]
df1['DC_POWER']
df1['DC_POWER']==df1['DC_POWER'].max()
df1[df1['DC_POWER']==df1['DC_POWER'].max()]['SOURCE_KEY']
import pandas as pd

data = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

data["DC_POWER"] = data ["DC_POWER"].rank()

data.sort_values("DC_POWER", ascending = False)

print(data)
data = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

data["AC_POWER"] = data ["AC_POWER"].rank(numeric_only=None)

data.sort_values("AC_POWER", ascending = False)

print(data)
print("Is there a null value in the Generation data? {}".format(df1.isnull().sum().any()))

print("Is there a null value in the Generation data? {}".format(df2.isnull().sum().any()))
print("Is there a null value in the Weather data? {}".format(df3.isnull().sum().any()))

print("Is there a null value in the Weather data? {}".format(df4.isnull().sum().any()))