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
drop_col = ['Data contributor','Operative temperature (F)','Radiant temperature (F)','Globe temperature (F)','Outdoor monthly air temperature (F)','Velocity_l (fpm)','Velocity_m (fpm)','Velocity_h (fpm)','Tg_l (F)','Tg_m (F)','Tg_h (F)','Ta_l (F)','Ta_m (F)','Ta_h (F)','Air temperature (F)','Air velocity (fpm)']
data = data.drop(drop_col,axis=1)
data.head()
data = data.rename(columns={'PMV': 'Predicted Mean Vote', 
                            'PPD': 'Predicted Percentage Disatisfied', 
                            'SET':'Standard Effective Temp', 
                            'CLO': 'Clothing Insulation', 
                            'Ta_h (C)': 'tempfloor_high (C)', 
                            'Ta_m (C)':'tempfloor_med (C)', 
                            'Ta_l (C)':'tempfloor_low (C)', 
                            'Tg_h (C)':'globetemp_high (C)', 
                            'Tg_m (C)':'globetemp_med (C)',
                            'Tg_l (C)':'globetemp_low (C)',
                            'velocity_h (m/s)':'velocity_high (m/s)',
                            'velocity_m (m/s)':'velocity_med (m/s)',
                            'velocity_l (m/s)':'velocity_low (m/s)', 
                            'Cooling startegy_building level':'cooling_strategy_building',
                            'Cooling startegy_operation mode for MM buildings':'cooling_strategy_for_mm_buildings',
                            'Building type': 'Building_type'})
data.columns
import seaborn as sns
sns.set()
data['Building_type'].value_counts().plot(kind='barh', figsize=(20,6))
import seaborn as sns
sns.set()
data['Sex'].value_counts().plot(kind='barh', figsize=(30,10))
import missingno as msno

msno.matrix(data.select_dtypes(include='number'));
msno.matrix(data.select_dtypes(include='O'));

data_retirement = data[data.Building_type == 'Senior center']
data_family = data[data.Building_type == 'Multifamily housing']
msno.matrix(data_retirement.select_dtypes(include='number'))
msno.matrix(data_retirement.select_dtypes(include='O'))
msno.matrix(data_family.select_dtypes(include='number'))
msno.matrix(data_family.select_dtypes(include='O'))
msno.matrix(data_family.select_dtypes(include='O'))
drop_cols = ['Thermal sensation acceptability',
             'Air movement acceptability',
             'activity_10',
             'activity_20',
             'activity_30',
             'activity_60',
             'globetemp_high (C)',
             'globetemp_med (C)',
             'globetemp_low (C)',
             'Subject«s height (cm)',
             'Subject«s weight (kg)',
             'Blind (curtain)',
             'Door',
             'Air movement preference',
             'Humidity preference',
             'Humidity sensation',
             'Publication (Citation)',
             'Database']
            
data_Retirement = data_retirement.drop(drop_cols,axis =1)
data_Retirement = data_Retirement.dropna()
msno.matrix(data_Retirement.select_dtypes(include='number'))
msno.matrix(data_Retirement.select_dtypes(include='O'))

keep_col_1 = ['Year', 
            'Thermal sensation', 
            'Clo', 
            'Met', 
            'Air temperature (C)', 
            'Relative humidity (%)', 
            'Air velocity (m/s)', 
            'Outdoor monthly air temperature (C)',
            'Season', 
            'Climate', 
            'City', 
            'Country',
            'Building_type', 
            'cooling_strategy_building', 
            'Thermal preference', 
            'Thermal sensation acceptability',
            'Sex']
data_Family = data_family[keep_col_1]
data_Family = data_Family.dropna()
msno.matrix(data_Family.select_dtypes(include='number'))
msno.matrix(data_Family.select_dtypes(include='O'))
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
object_cols_retirement = ['Season', 'Koppen climate classification', 'Climate','City','Country','Building_type',
                          'cooling_strategy_building','Heating strategy_building level', 'Sex','Thermal preference','Thermal comfort']

OH_cols_retirement = pd.DataFrame(OH_encoder.fit_transform(data_Retirement[object_cols_retirement]))
OH_cols_retirement.index = data_Retirement.index
column_name_retirement = OH_encoder.get_feature_names(object_cols_retirement)
OH_cols_retirement.columns = column_name_retirement
OH_cols_retirement
other_data_retirement = data_Retirement.drop(object_cols_retirement, axis=1)
OH_data_retirement = pd.concat([other_data_retirement, OH_cols_retirement], axis=1)
OH_data_retirement
object_cols_family = ['Season','Climate','City','Country','Building_type','cooling_strategy_building','Thermal preference','Sex']
OH_cols_family = pd.DataFrame(OH_encoder.fit_transform(data_Family[object_cols_family]))
OH_cols_family.index = data_Family.index
column_name_family = OH_encoder.get_feature_names(object_cols_family)
OH_cols_family.columns = column_name_family
OH_cols_family
other_data_family = data_Family.drop(object_cols_family, axis=1)
OH_data_family = pd.concat([other_data_family, OH_cols_family], axis=1)
OH_data_family
OH_data_retirement.columns
columns_retirement = OH_data_retirement.select_dtypes(include='number').drop('Thermal sensation',axis=1).columns

data_grouped_retirement = OH_data_retirement.groupby(['Sex_Female','Thermal sensation'])[columns_retirement].mean()
data_grouped_retirement
OH_data_family.columns
columns_family = OH_data_family.select_dtypes(include='number').drop('Thermal sensation',axis=1).columns

data_grouped_family = OH_data_family.groupby(['Sex_Female','Thermal sensation'])[columns_family].mean()
data_grouped_family
OH_data_retirement.corr()['Sex_Female'].sort_values(ascending=False).head(10)
OH_data_family.corr()['Sex_Female'].sort_values(ascending=False).head(10)
violin_1 = sns.violinplot(x="Sex_Female", y="Thermal sensation", data=OH_data_family)
violin_2 = sns.violinplot(x="Sex_Female", y="Thermal sensation", data=OH_data_retirement)