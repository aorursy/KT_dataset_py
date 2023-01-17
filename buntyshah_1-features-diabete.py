import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# List sheetnames in excel file from USDA
xl = pd.ExcelFile('../input/DataDownload.xls')
xl.sheet_names  # see all sheet names
# Read in county-level data about stores
stores=pd.read_excel('../input/DataDownload.xls', sheet_name='STORES')
stores.head()
stores.columns
stores=pd.read_excel('../input/DataDownload.xls', sheet_name='STORES')
stores=stores[['FIPS', 'State', 'County', 'GROCPTH14',  'WICSPTH12','SNAPSPTH16','SPECSPTH14','CONVSPTH14']]
print(stores.shape)
stores.head()
# Import from Excel
restaur=pd.read_excel('../input/DataDownload.xls', sheet_name='RESTAURANTS')
restaur=restaur[['FIPS', 'State', 'County', 'FFRPTH14', 'FSRPTH14', 'PC_FFRSALES12', 'PC_FSRSALES12']]
print(restaur.shape)
restaur.head()
soceco=pd.read_excel('../input/DataDownload.xls', sheet_name='SOCIOECONOMIC')
soceco=soceco[['FIPS', 'State', 'County', 'PCT_NHWHITE10', 'PCT_65OLDER10', 'PCT_18YOUNGER10', 'PERPOV10', 'METRO13', 'POPLOSS10', 'MEDHHINC15']]
print(soceco.shape)
soceco.head()
# 2015 population
pop15=pd.read_excel('../input/DataDownload.xls', sheet_name='Supplemental Data - County')
pop15.rename(columns={'FIPS ':'FIPS'}, inplace=True)
pop15=pop15[['FIPS','State', 'County', 'Population Estimate, 2015']]
print(pop15.shape)
pop15.head()
print(pop15['FIPS'].nunique())
print(stores['FIPS'].nunique())
access=pd.read_excel('../input/DataDownload.xls', sheet_name='ACCESS')
access=access[['FIPS', 'State', 'County', 'PCT_LACCESS_POP15']]
access.head()
snap=pd.read_excel('../input/DataDownload.xls', sheet_name='ASSISTANCE')
snap=snap[['FIPS', 'State', 'County', 'SNAP_PART_RATE13', 'PCT_NSLP15', 'PCT_WIC15', 'PCT_CACFP15']]
snap.head()
health=pd.read_excel('../input/DataDownload.xls', sheet_name='HEALTH')
health=health[['FIPS', 'State', 'County', 'PCT_DIABETES_ADULTS13', 'PCT_OBESE_ADULTS13', 'RECFACPTH14']]
health.head()
# What is the national average rate of diabetes and obesity?
print('diabetes:', health['PCT_DIABETES_ADULTS13'].mean())
print('obesity:', health['PCT_OBESE_ADULTS13'].mean())
# Create the target variable (diabetes)
health['hi_diabetes']=0
health.loc[health['PCT_DIABETES_ADULTS13']>health['PCT_DIABETES_ADULTS13'].mean(), 'hi_diabetes']=1
health['hi_diabetes'].value_counts()
# Create the target variable (obesity)
health['hi_obesity']=0
health.loc[health['PCT_OBESE_ADULTS13']>health['PCT_OBESE_ADULTS13'].mean(), 'hi_obesity']=1
health['hi_obesity'].value_counts()
# Drop the continuous variable
health=health.drop(['PCT_DIABETES_ADULTS13', 'PCT_OBESE_ADULTS13'], axis=1)
health=health[['FIPS', 'State', 'County', 'hi_diabetes', 'hi_obesity', 'RECFACPTH14']]
health.columns
df=health
df=pd.merge(df, restaur, on=['FIPS', 'State', 'County'])
df=pd.merge(df, soceco, on=['FIPS', 'State', 'County'])
df=pd.merge(df, access, on=['FIPS', 'State', 'County'])
df=pd.merge(df, snap, on=['FIPS', 'State', 'County'])
df=pd.merge(df, stores, on=['FIPS', 'State', 'County'])
df.head()
# drop missing valued
print('before:', df.shape)
df=df.dropna(how='any')
print('after:', df.shape)
df.to_csv('counties.csv', index=False)