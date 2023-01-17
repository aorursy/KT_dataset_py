import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
df=pd.read_csv("../input/existing-buildings-energy-water-efficiency-ebewe-program.csv")
df.head()
df.count()
df=df.drop(['BUILDING ADDRESS'],axis=1)
df_RN = df.rename(columns={'BUILDING ID': 'BuildingId','POSTAL CODE':'PostalCode',
                        'ENTITY RESPONSIBLE FOR BENCHMARK':'EntityResponseForBenchmark',
                        'COMPLIANCE STATUS':'CompilanceStatus',
                        'PROPERTY TYPE':'PropertyType','ENERGY STAR SCORE':'EnergyStarScore',
                        'SITE ENERGY USE INTENSITY (EUI) (kBtu/ft²)':'SiEUI',
                        'SOURCE ENERGY USE INTENSITY (EUI) (kBtu/ft²)':'SoEUI',
                        'WEATHER NORMALIZED SITE ENERGY USE INTENSITY (EUI) (kBtu/ft²)':'WNSiEUI',
                        'WEATHER NORMALIZED SOURCE ENERGY USE INTENSITY (EUI) (kBtu/ft²)':'WNSoEUI',
                        'CARBON DIOXIDE EMISSIONS (Metric Ton CO2e)':'CDE',
                        'INDOOR WATER USE (kgal)':'IWU',
                        'INDOOR WATER INT (gal/ft²)':'IWI',
                        'OUTDOOR WATER USE (kgal)':'OWU',
                        'TOTAL WATER USE (kgal)':'TWU',
                        'GROSS BUILDING FLOOR AREA (ft²)':'GBFA',
                        'PROGRAM YEAR': 'PROGRAM_YEAR'})
df_RN.head()
df_RN_clear=df_RN.dropna(how='any')
df_RN_clear.head()
df_RN_clear.count()
df_RN_clear = df_RN_clear.drop(df_RN_clear[df_RN_clear.EnergyStarScore=='Not Available'].index)
df_RN_clear = df_RN_clear.drop(df_RN_clear[df_RN_clear.SiEUI=='Not Available'].index)
df_RN_clear = df_RN_clear.drop(df_RN_clear[df_RN_clear.SoEUI=='Not Available'].index)
df_RN_clear = df_RN_clear.drop(df_RN_clear[df_RN_clear.WNSiEUI=='Not Available'].index)
df_RN_clear = df_RN_clear.drop(df_RN_clear[df_RN_clear.WNSoEUI=='Not Available'].index)
df_RN_clear = df_RN_clear.drop(df_RN_clear[df_RN_clear.CDE=='Not Available'].index)
df_RN_clear = df_RN_clear.drop(df_RN_clear[df_RN_clear.IWU=='Not Available'].index)
#df_RN_clear = df_RN_clear.drop(df_RN_clear[df_RN_clear.IWI=='Not Available'].index)
df_RN_clear.head()
print(df_RN_clear.EntityResponseForBenchmark.unique().shape)
print(df_RN_clear.CompilanceStatus.unique().shape)
print(df_RN_clear.PropertyType.unique().shape)
df_RN_clear['BuildingId']=df_RN_clear['BuildingId'].astype(str) 
df_RN_clear['BuildingId']= df_RN_clear['BuildingId'].str.extract('(\d+)', expand=False)
df_RN_clear['BuildingId']=df_RN_clear.BuildingId.astype(int) 
df_RN_clear['PROGRAM_YEAR']=df_RN_clear.PROGRAM_YEAR.astype(int)
df_RN_clear['GBFA']=df_RN_clear['GBFA'].astype(str) 
df_RN_clear['GBFA']= df_RN_clear['GBFA'].str.extract('(\d+)', expand=False)
df_RN_clear['GBFA']=df_RN_clear['GBFA'].astype(int) 
df_RN_clear['TWU']=df_RN_clear['TWU'].astype(str) 
df_RN_clear['TWU']= df_RN_clear['TWU'].str.extract('(\d+)', expand=False)
df_RN_clear['TWU']=df_RN_clear['TWU'].astype(float) 
#df_RN_clear['TWU']=df_RN_clear.TWU.astype(int) 
#df_RN_clear['EnergyStarScore']=df_RN_clear['EnergyStarScore'].astype(str) 
#df_RN_clear['EnergyStarScore']= df_RN_clear['EnergyStarScore'].str.extract('(\d+)', expand=False)
df_RN_clear['EnergyStarScore']=df_RN_clear['EnergyStarScore'].astype(float) 
#df_RN_clear['EnergyStarScore']=df_RN_clear['EnergyStarScore'].astype(int) 
#df_RN_clear['SiEUI']=df_RN_clear.TWU.astype(str) 
#df_RN_clear['SiEUI']= df_RN_clear['SiEUI'].str.extract('(\d+)', expand=False)
df_RN_clear['SiEUI']=df_RN_clear['SiEUI'].astype(float) 
#df_RN_clear['SiEUI']=df_RN_clear['SiEUI'].astype(int) 
#df_RN_clear['SoEUI']=df_RN_clear.TWU.astype(str) 
#df_RN_clear['SoEUI']= df_RN_clear['SoEUI'].str.extract('(\d+)', expand=False)
df_RN_clear['SoEUI']=df_RN_clear['SoEUI'].astype(float) 
#df_RN_clear['SoEUI']=df_RN_clear['SoEUI'].astype(int) 
#df_RN_clear['WNSiEUI']=df_RN_clear.TWU.astype(str) 
#df_RN_clear['WNSiEUI']= df_RN_clear['WNSiEUI'].str.extract('(\d+)', expand=False)
df_RN_clear['WNSiEUI']=df_RN_clear['WNSiEUI'].astype(float)
#df_RN_clear['WNSiEUI']=df_RN_clear['WNSiEUI'].astype(int)
#df_RN_clear['WNSoEUI']=df_RN_clear.TWU.astype(str) 
#df_RN_clear['WNSoEUI']= df_RN_clear['WNSoEUI'].str.extract('(\d+)', expand=False)
df_RN_clear['WNSoEUI']=df_RN_clear['WNSoEUI'].astype(float) 
#df_RN_clear['WNSoEUI']=df_RN_clear['WNSoEUI'].astype(int) 
#df_RN_clear['CDE']=df_RN_clear.TWU.astype(str) 
#df_RN_clear['CDE']= df_RN_clear['CDE'].str.extract('(\d+)', expand=False)
df_RN_clear['CDE']=df_RN_clear['CDE'].astype(float) 
#df_RN_clear['CDE']=df_RN_clear['CDE'].astype(int) 
#df_RN_clear['IWU']=df_RN_clear.TWU.astype(str) 
#df_RN_clear['IWU']= df_RN_clear['IWU'].str.extract('(\d+)', expand=False)
df_RN_clear['IWU']=df_RN_clear['IWU'].astype(float) 
#df_RN_clear['IWU']=df_RN_clear['IWU'].astype(int) 
df_RN_clear['OWU']=df_RN_clear['OWU'].astype(str) 
df_RN_clear['OWU']= df_RN_clear['OWU'].str.extract('(\d+)', expand=False)
df_RN_clear['OWU']=df_RN_clear['OWU'].astype(float) 
#df_RN_clear['OWU']=df_RN_clear['OWU'].astype(int) 
#df_RN_clear['IWI']=df_RN_clear.TWU.astype(str) 
#df_RN_clear['IWI']= df_RN_clear['IWI'].str.extract('(\d+)', expand=False)
df_RN_clear['IWI']=df_RN_clear['IWI'].astype(float) 
#df_RN_clear['IWI']=df_RN_clear['IWI'].astype(int) 
print(np.isfinite(df_RN_clear['SiEUI']).size)
print(np.isfinite(df_RN_clear['SoEUI']).size)
print(np.isfinite(df_RN_clear['GBFA']).size)
print(np.isfinite(df_RN_clear['BuildingId']).size)
df_RN_clear.dtypes
df_RN_clear= pd.get_dummies(df_RN_clear, prefix='CS_', columns=['CompilanceStatus'])
df_RN_clear= pd.get_dummies(df_RN_clear, prefix='PT_', columns=['PropertyType'])
df_RN_clear=df_RN_clear.drop(['EntityResponseForBenchmark'],axis=1)
df_RN_clear.head()
import seaborn as sns
import matplotlib.pyplot as pl

f, ax = pl.subplots(figsize=(10, 10))
corr = df_RN_clear.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

df_1=df_RN_clear[['EnergyStarScore','SiEUI','SoEUI','WNSiEUI','WNSoEUI','CDE','IWU','IWI','TWU','PT__Hospital (General Medical & Surgical), Parking','PT__Hotel']].copy()
f, ax = pl.subplots(figsize=(10, 10))
corr = df_1.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
from sklearn.preprocessing import StandardScaler
X=df_RN_clear.loc[:,['SiEUI','SoEUI','WNSiEUI','WNSoEUI','CDE','IWU','TWU']].values
Y=df_RN_clear.loc[:,'PT__Hospital (General Medical & Surgical), Parking'].values

import statsmodels.formula.api as sm

#reg=sm.OLS(endog=Y,exog=X).fit()
#reg.summary()
