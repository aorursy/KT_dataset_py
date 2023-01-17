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
raw_data = pd.read_csv('/kaggle/input/ashrae-global-thermal-comfort-database-ii/ashrae_db2.01.csv')
data = raw_data.copy()
drop_col = ['Climate','Publication (Citation)','Data contributor','Operative temperature (F)','Radiant temperature (F)','Globe temperature (F)','Outdoor monthly air temperature (F)','Velocity_l (fpm)','Velocity_m (fpm)','Velocity_h (fpm)','Tg_l (F)','Tg_m (F)','Tg_h (F)','Ta_l (F)','Ta_m (F)','Ta_h (F)','Air temperature (F)','Air velocity (fpm)']
data = data.drop(drop_col,axis=1)
data.head()
import seaborn as sns
sns.set()
data['Season'].value_counts().plot(kind='barh', figsize=(20,6))
data['Koppen climate classification'].value_counts().plot(kind='barh', figsize=(20,6))
ax = data['Year'].plot(kind='hist')
ax = data['Age'].plot(kind='hist')
data['Building type'].value_counts().plot(kind='barh', figsize=(20,6))
data = data.rename(columns={'PMV': 'Predicted Mean Vote', 'PPD': 'Predicted Percentage Disastisfied', 'SET':'Standard Effective Temp', 'CLO': 'Clothing Insulation', 'Ta_h (C)': 'tempfloor_high (C)', 'Ta_m (C)':'tempfloor_med (C)', 'Ta_l (C)':'tempfloor_low (C)', 'Tg_h (C)':'globetemp_high (C)', 'Tg_m (C)':'globetemp_med (C)','Tg_l (C)':'globetemp_low (C)','velocity_h (m/s)':'velocity_high (m/s)','velocity_m (m/s)':'velocity_med (m/s)','velocity_l (m/s)':'velocity_low (m/s)'})
data.columns
data.corr()['Thermal sensation'].sort_values(ascending=False).head(10)
#positive correlation
data.corr()['Thermal sensation'].sort_values().head(10)
#negative correlation
data['Koppen climate classification'].value_counts()
data['Koppen climate classification'].unique()
tropical_A = []
dry_B = []
temperate_C = []
continental_D = []
polar_E = []

for climate in data['Koppen climate classification'].unique():
    if climate[0] == 'A':
        tropical_A.append(climate)
    elif climate[0] == 'B':
        dry_B.append(climate)
    elif climate[0] == 'C':
        temperate_C.append(climate)
    elif climate[0] == 'D':
        continental_D.append(climate)
    elif climate[0] == 'E':
        polar_E.append(climate)
print(tropical_A)
print(dry_B)
print(temperate_C)
print(continental_D)
print(polar_E)
data.loc[data['Koppen climate classification'].isin(tropical_A), 
             'Climate'] = 'Tropical'
data.loc[data['Koppen climate classification'].isin(dry_B), 
             'Climate'] = 'Dry'
data.loc[data['Koppen climate classification'].isin(temperate_C), 
             'Climate'] = 'Temperate'
data.loc[data['Koppen climate classification'].isin(continental_D), 
             'Climate'] = 'Continental'
data.loc[data['Koppen climate classification'].isin(polar_E), 
             'Climate'] = 'Polar'
data.Climate.value_counts()
data['Koppen climate classification'].isin(tropical_A).sum()
data['Koppen climate classification'].isin(dry_B).sum()
data['Koppen climate classification'].isin(temperate_C).sum()
data['Koppen climate classification'].isin(continental_D).sum()
data.columns
data = data.rename(columns={'Cooling startegy_building level':'Cooling_strategy_building_level','Cooling startegy_operation mode for MM buildings': 'Cooling_strategy_operation_mode_for_MM' })
data.select_dtypes(exclude='number').columns
data['City'].unique()
data.select_dtypes(include='number').columns
import missingno as msno

msno.matrix(data.select_dtypes(include='number'));
# We are going to use data_no_na for the rest of handling missing columns
data_no_na = data.copy() 
print('mean: ' + str(data_no_na['Age'].mean()))
print('median: '+ str(data_no_na['Age'].median()))
data['Age'].describe()
data_no_na['Age'] = data_no_na['Age'].fillna(data_no_na['Age'].mean())
print('mean: ' + str(data_no_na['Thermal sensation'].mean()))
print('median: '+ str(data_no_na['Thermal sensation'].median()))
data_no_na['Thermal sensation'].describe()
data_no_na['Thermal sensation'] = data_no_na['Thermal sensation'].fillna(data_no_na['Thermal sensation'].mean())
print('mean: ' + str(data_no_na['Clo'].mean()))
print('median: '+ str(data_no_na['Clo'].median()))
data_no_na['Clo'].describe()
data_no_na['Clo'] = data_no_na['Clo'].fillna(data_no_na['Clo'].mean())
print('mean: ' + str(data_no_na['Met'].mean()))
print('median: '+ str(data_no_na['Met'].median()))
data_no_na['Met'].describe()
data_no_na['Met'] = data_no_na['Met'].fillna(data_no_na['Met'].mean())
print('mean: ' + str(data_no_na['Air temperature (C)'].mean()))
print('median: '+ str(data_no_na['Air temperature (C)'].median()))
data_no_na['Air temperature (C)'].describe()
data_no_na['Air temperature (C)'] = data_no_na['Air temperature (C)'].fillna(data_no_na['Air temperature (C)'].mean())
print('mean: ' + str(data_no_na['Relative humidity (%)'].mean()))
print('median: '+ str(data_no_na['Relative humidity (%)'].median()))
data_no_na['Relative humidity (%)'].describe()
data_no_na['Relative humidity (%)'] = data_no_na['Relative humidity (%)'].fillna(data_no_na['Relative humidity (%)'].mean())
print('mean: ' + str(data_no_na['Air velocity (m/s)'].mean()))
print('median: '+ str(data_no_na['Air velocity (m/s)'].median()))
data_no_na['Air velocity (m/s)'].describe()
data_no_na['Air velocity (m/s)'] = data_no_na['Air velocity (m/s)'].fillna(data_no_na['Air velocity (m/s)'].median())
print('mean: ' + str(data_no_na['Outdoor monthly air temperature (C)'].mean()))
print('median: '+ str(data_no_na['Outdoor monthly air temperature (C)'].median()))
data_no_na['Outdoor monthly air temperature (C)'].describe()
data_no_na['Outdoor monthly air temperature (C)'] = data_no_na['Outdoor monthly air temperature (C)'].fillna(data_no_na['Outdoor monthly air temperature (C)'].mean())
msno.matrix(data_no_na.select_dtypes(include='number'));
data_no_na['Thermal sensation acceptability'].value_counts()
data_no_na['Air movement acceptability'].value_counts()
data_no_na['Air movement acceptability'] = data_no_na['Air movement acceptability'].astype('category')
data_no_na['Thermal sensation acceptability'] = data_no_na['Thermal sensation acceptability'].astype('category')
data_no_na.select_dtypes(include='number').columns
msno.matrix(data_no_na.select_dtypes(include='object'));
