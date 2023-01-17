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
data['Cooling startegy_building level'].value_counts().plot(kind='barh', figsize=(20,6))
data['Building type'].value_counts().plot(kind='barh', figsize=(20,6))
data['Thermal preference'].value_counts().plot(kind='barh', figsize=(20,6))
data = data.rename(columns={'PMV': 'Predicted Mean Vote', 
                            'PPD': 'Predicted Percentage Disastisfied', 
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
                            'Cooling startegy_operation mode for MM buildings':'cooling_strategy_for_mm_buildings'})
data.columns
data['Koppen climate classification'].isna().sum()
tropical_A = []
dry_B = []
temperate_C = []
continental_D = []
polar_E = []


# Koppen climate classifcation that starts with A always refers to tropical, B refers to dty and so on...
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
data['Air movement acceptability'] = data['Air movement acceptability'].astype('object')
data['Thermal sensation acceptability'] = data['Thermal sensation acceptability'].astype('object')
import missingno as msno

msno.matrix(data.select_dtypes(include='number'));
msno.matrix(data.select_dtypes(include='O'));
data.cooling_strategy_building.unique()
data_mechanical = data[data.cooling_strategy_building == 'Mechanically Ventilated']
data_aircon = data[data.cooling_strategy_building == 'Air Conditioned']
data_natural = data[data.cooling_strategy_building == 'Naturally Ventilated']
data_mm = data[data.cooling_strategy_building == 'Mixed Mode']
data_na = data[data.cooling_strategy_building.isna() == True]
msno.matrix(data_mechanical.select_dtypes(include='number'))
msno.matrix(data_mm.select_dtypes(include='number'))
msno.matrix(data_aircon.select_dtypes(include='number'))
msno.matrix(data_natural.select_dtypes(include='number'))
keep_col = ['Year', 
            'Thermal sensation', 
            'Predicted Mean Vote', 
            'Predicted Percentage Disastisfied', 
            'Standard Effective Temp', 
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
            'Building type', 
            'cooling_strategy_building', 
            'Thermal preference', 
            'Thermal sensation acceptability']
# We are going to use this variable
data_no_na = data[keep_col]
data_no_na.select_dtypes(include='number').columns
data_no_na.select_dtypes(include='O').columns
print('mean: ' + str(data_no_na['Clo'].mean()))
print('median: '+ str(data_no_na['Clo'].median()))
data_no_na['Clo'].describe()
sns.distplot(data_no_na['Clo'])
data_no_na['Clo'] = data_no_na['Clo'].fillna(data_no_na['Clo'].mean())
print('mean: ' + str(data_no_na['Met'].mean()))
print('median: '+ str(data_no_na['Met'].median()))
data_no_na['Met'].describe()
sns.distplot(data_no_na['Met'])
data_no_na['Met'] = data_no_na['Met'].fillna(data_no_na['Met'].median())
data_simple = data_no_na.copy()
data_simple = data_simple.dropna()
msno.matrix(data_simple)
data_simple.cooling_strategy_building.unique()
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
object_cols = ['Season','Climate','Building type','Thermal sensation acceptability','Thermal preference']
OH_cols = pd.DataFrame(OH_encoder.fit_transform(data_simple[object_cols]))
OH_cols.index = data_simple.index
column_name = OH_encoder.get_feature_names(object_cols)
OH_cols.columns = column_name
OH_cols
data_simple.select_dtypes(include="O").columns
# Get list of categorical variables
object_cols = ['Season','Climate','Building type','Thermal sensation acceptability','Thermal preference']
data_simple
other_data = data_simple.drop(object_cols, axis=1)
OH_data = pd.concat([other_data, OH_cols], axis=1)
OH_data
OH_data.columns
columns = OH_data.select_dtypes(include='number').drop('Year',axis=1).columns

data_grouped = OH_data.groupby(['cooling_strategy_building','Year'])[columns].mean()
data_grouped
OH_data.dtypes
OH_data.cooling_strategy_building = OH_data.cooling_strategy_building.replace('Air Conditioned', 1)
OH_data.cooling_strategy_building = OH_data.cooling_strategy_building.replace('Mixed Mode', 2)
OH_data.cooling_strategy_building = OH_data.cooling_strategy_building.replace('Naturally Ventilated', 3)
OH_data.cooling_strategy_building = OH_data.cooling_strategy_building.astype('float64')
from sklearn.model_selection import train_test_split

train_columns = OH_data.drop(['City', 'Country','cooling_strategy_building'], axis=1).columns

OH_data.cooling_strategy_building

X_train, X_test, y_train, y_test = train_test_split(OH_data[train_columns], OH_data.cooling_strategy_building, test_size=0.33, random_state=42)
from sklearn.cluster import KMeans

# We know beforehand that we have three class
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train)
predictions = kmeans.predict(X_train)
from sklearn.metrics import accuracy_score

accuracy_score(y_train,predictions )
kmeans.inertia_
predictions_test = kmeans.predict(X_test)
accuracy_score(y_test,predictions_test)
from sklearn.metrics import (
                            homogeneity_completeness_v_measure, 
                            adjusted_rand_score, 
)
homogeneity, completeness, v_measure =  homogeneity_completeness_v_measure(y_test, predictions_test)

print(f"""Cluster Model Performance:
      Homogeneity: {homogeneity}
      Completeness: {completeness}
      V-Measure: {v_measure}""")
OH_data.columns
OH_data.groupby(['cooling_strategy_building'])['Thermal sensation acceptability_1.0'].mean()
OH_data_natural_ventilaton= OH_data[OH_data.cooling_strategy_building == 3]['Thermal sensation']
OH_data_not_natural_ventilaton= OH_data[OH_data.cooling_strategy_building != 3]['Thermal sensation']
sns.distplot(OH_data_natural_ventilaton)
sns.distplot(OH_data_not_natural_ventilaton)
sns.scatterplot(x="Air temperature (C)", y="Thermal sensation", data=OH_data, hue='cooling_strategy_building')
OH_data[['Season_Autumn',
       'Season_Spring', 'Season_Summer', 'Season_Winter']]
OH_data_summer = OH_data[OH_data.Season_Summer == 1]
OH_data_winter = OH_data[OH_data.Season_Winter == 1]
OH_data_spring = OH_data[OH_data.Season_Spring == 1]
OH_data_autumn = OH_data[OH_data.Season_Autumn == 1]
sns.distplot(OH_data_summer['Thermal sensation acceptability_1.0'])
sns.distplot(OH_data_autumn['Thermal sensation acceptability_1.0'])
sns.distplot(OH_data_winter['Thermal sensation acceptability_1.0'])
sns.distplot(OH_data_spring['Thermal sensation acceptability_1.0'])
