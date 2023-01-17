import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
# Read relevent data from categry Access and Proximity to Grocery Store
Access = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='ACCESS')
Access= Access[['FIPS', 'State', 'County','LACCESS_POP10','LACCESS_LOWI10','LACCESS_HHNV10']]
Access.shape
Access.head()
Access.info()
#Create test data of 2014 for prediction
Access_test = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='ACCESS')
Access_test = Access_test[['FIPS', 'State', 'County','LACCESS_POP15','LACCESS_LOWI15','LACCESS_HHNV15']]
Access_test.head()
# Read relevent data from Store Sheet
Stores = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='STORES')
#find retail store decline by difference of 2009 to 2014

Stores['total_2009'] = Stores['GROC09'] + Stores['SUPERC09'] + Stores['CONVS09'] + Stores['SPECS09']
Stores['total_2014'] = Stores['GROC14'] + Stores['SUPERC14'] + Stores['CONVS14'] + Stores['SPECS14']
Stores['is_decline'] = Stores['total_2014']- Stores['total_2009']
Stores['is_store_decline'] = Stores['is_decline'].apply(lambda x : 1 if (x<0) else 0   ) 
Stores=Stores[['FIPS', 'State', 'County','GROC09','GROCPTH09','SUPERC09','SUPERCPTH09','CONVS09','CONVSPTH09','SPECS09','SPECSPTH09','SNAPS12','SNAPSPTH12','is_store_decline']]
Stores.head()
#Prepare store data for 2014
Stores_test = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='STORES')
Stores_test=Stores_test[['FIPS', 'State', 'County','GROC14','GROCPTH14','SUPERC14','SUPERCPTH14','CONVS14','CONVSPTH14','SPECS14','SPECSPTH14','SNAPS16','SNAPSPTH16']]
Stores_test.head()
# Read relevent data from Health Sheet
health = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='HEALTH')
health=health[['FIPS', 'State', 'County','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS08','RECFAC09']]
health.head()
# Create test data for 2014
health_test = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='HEALTH')
health_test=health_test[['FIPS', 'State', 'County','PCT_DIABETES_ADULTS13','PCT_OBESE_ADULTS13','RECFAC14']]
health_test.head()
# Read relevent data from SOCIOECONOMIC Sheet
social = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='SOCIOECONOMIC')
social=social[['FIPS', 'State', 'County','MEDHHINC15','METRO13','POPLOSS10','POVRATE15','PERPOV10']]
social.head()
# Create Data for 2014
social_test = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='SOCIOECONOMIC')
social_test=social_test[['FIPS', 'State', 'County','MEDHHINC15','METRO13','POPLOSS10','POVRATE15','PERPOV10']]
social_test.head()
BEA = pd.read_excel("../input/bureau-of-economic-analysis-data/Pop_2009.xlsx",sheet_name='2009')
BEA.head()
BEA.info()
#Read data for 2014
BEA_test = pd.read_excel('../input/us-bureau-of-economic-analysis-bea-2014-data/Pop_2014.xlsx')
BEA_test.head()
#Merging all the data

df=Stores
df=pd.merge(df, Access, on=['FIPS', 'State', 'County'])
df=pd.merge(df, health, on=['FIPS', 'State', 'County'])
df=pd.merge(df, social, on=['FIPS', 'State', 'County'])
df.head()
df1= pd.merge(df, BEA, on='FIPS',how='left')
df1.head()
df1 = df1.rename(columns={'State_x': 'State', 'County_x': 'County'})
df1.head()
df_test=Stores_test
df_test=pd.merge(df_test, Access_test, on=['FIPS', 'State', 'County'])
df_test=pd.merge(df_test, health_test, on=['FIPS', 'State', 'County'])
df_test=pd.merge(df_test, social_test, on=['FIPS', 'State', 'County'])
df_test.head()
df_test= pd.merge(df_test, BEA_test, on='FIPS',how='left')
df_test.head()
df_test = df_test.rename(columns={'State_x': 'State', 'County_x': 'County'})
df_test = df_test.drop(['County_y','State_y'],axis=1)
df_test.head()
#drop missing valued
#print('before:', df1.shape)
#df1=df1.dropna(how='any')
#print('after:', df1.shape)
df1.to_csv('counties.csv', index=False)
df_test.to_csv('test.csv',index=False)