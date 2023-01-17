# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd # For parallel dataprocessing to avoid extreme memory consumption
import matplotlib.pyplot as plt #To generate plots
from haversine import haversine, Unit # To calculate the distance between origin and destination states
import glob # to extract file locations
import seaborn as sns #To generate plots
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.concat([pd.read_csv(f) for f in glob.glob("/kaggle/input/projectdata/FAF4.5.1_Reprocessed_*.csv")])
df.info()
state_location = pd.read_csv("/kaggle/input/projectdata/state_distance.csv")
state_location.info()
df = df.merge(state_location, left_on = 'dms_destst', right_on = 'FIPS')
df = df.rename(columns = {'Latitude': 'dest_lat','Longitude':'dest_lon'})
df = df.drop(['Name','FIPS'],axis = 1) # removing unrequired columns to avoid conflict in the next merge
df = df.merge(state_location, left_on = 'dms_origst', right_on = 'FIPS')
df = df.rename(columns = {'Latitude': 'orig_lat','Longitude':'orig_lon'})
df = df.drop(['Name','FIPS'],axis = 1)# removing unrequired columns
df['ori_lat_lon'] = list(zip(df['orig_lat'], df['orig_lon']))
df['des_lat_lon'] = list(zip(df['dest_lat'], df['dest_lon']))
df['Distance'] = [haversine(x,y) for x,y in zip(df['ori_lat_lon'],df['des_lat_lon'])]  # Calculating distance between origin and destination
df = df.drop(['orig_lon','orig_lat','dest_lat','dest_lon','ori_lat_lon','des_lat_lon'],axis = 1) # removing unrequired columns
df = df[['dms_origst','dms_destst', 'dms_mode', 'sctg2', 'trade_type','value_1997', 'value_2002', 'value_2007' ,'value_2012', 'value_2017', 'Distance']]
df = pd.melt(df, id_vars=list(df.columns[0:5]) + list(df.columns[10:11]), var_name = 'var_type')
df.shape
df['Year'] = df['var_type'].str.replace(r'\D','')
df = df.drop('var_type', axis =1)
df.drop_duplicates(inplace = True)
df.rename(columns = {'value':'Trade'}, inplace = True)
path ="/kaggle/input/projectdata/"
crops = ['sorghum','wheat','barley','animals_total','oats','milk','rice','corn','rye','honey','aquaculture','diaryplants']
file_path = [path + x + ".csv" for x in crops]
file_path

sctg1_4 = pd.concat([pd.read_csv(f, usecols=['Year','State', 'State ANSI','Commodity','Data Item','Value']) for f in file_path])
sctg1_4.head()
sctg1_4.dtypes
sctg1_4['Value'] = sctg1_4['Value'].str.replace(r'\D','')
sctg1_4.info()
sctg1_4.dropna(inplace = True)
sctg1_4['State ANSI'] =sctg1_4['State ANSI'].astype('int64')
sctg1_4.rename(columns ={'Value':'Production'}, inplace = True)
sctg1_4['Year'] = sctg1_4['Year'].astype(str)
sctg1_4['Production'] = pd.to_numeric(sctg1_4['Production'])
sctg1_4.dtypes

income =  pd.read_csv('/kaggle/input/projectdata/income.csv')
income = income[income['Description'] == 'Personal income (millions of dollars)']
income.drop(['LineCode'],axis = 1, inplace=True)
Years = ['1997','2002','2007','2012','2017']
income.reset_index(drop = True)
income.dtypes
income['GeoFips'] = income['GeoFips']/1000
income = pd.melt(income, id_vars=['GeoFips','GeoName','Description'], var_name = 'Year', value_name= 'Income')
income.head()
income.dtypes
income['GeoFips'] = income['GeoFips'].astype('int64')
income['Income'] = income['Income']*1000000 # converting income unit from millions of dollars to dollars
income.drop('Description', axis = 1, inplace = True)
income = income[income["Year"].isin(Years)]
income.shape
gdp = pd.read_csv('/kaggle/input/projectdata/state_gdp_1997_2018.csv')
gdp = pd.melt(gdp, id_vars=['GeoFIPS','GeoName','Unit'], var_name = 'Year', value_name= 'Gdp')
gdp.dtypes
gdp['Gdp'] = gdp['Gdp']*1000000
gdp.drop('Unit', axis = 1, inplace = True)
gdp = gdp[gdp["Year"].isin(Years)]
gdp.shape
df2 = pd.merge(gdp, income, how = 'left', left_on = ['GeoFIPS','Year'],right_on = ['GeoFips', 'Year'])
df2.dtypes
df2 = df2[df2["Year"].isin(Years)]
df2.drop(['GeoName_x','GeoName_y','GeoFips'],axis = 1, inplace=True) #dropping non-required columns
sctg1_4.dtypes
df2 = pd.merge(df2, sctg1_4, how = 'left', left_on = ['GeoFIPS','Year'],right_on = ['State ANSI', 'Year'])
df2.dtypes
df2.dropna(inplace = True)
df2.dtypes
df2 = df2[['GeoFIPS','Year','State','Commodity','Data Item','Production', 'Income','Gdp']]
df2.info()
df.dtypes
#df = dd.from_pandas(df, npartitions=5)
df = pd.merge(df,df2, how = 'left', left_on = ['dms_origst','Year'],right_on = ['GeoFIPS', 'Year'])
df.columns
df.dropna(inplace = True)
df.shape
value_features = ['Production','Trade','Income', 'Gdp', 'Distance']
df[value_features].describe()
plt.rcParams['figure.figsize'] = (10,8)
df[value_features].hist()
sns.boxplot(x='Trade', data=df);
sns.boxplot(x='Production', data=df);
plt.rcParams['figure.figsize'] = (10,8)
df[['Income','Gdp']].boxplot()
df[value_features].corr()
sns.heatmap(df[value_features].corr())