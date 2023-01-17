import pandas as pd 
import numpy as np 
from pandas import Series,DataFrame

import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('whitegrid')
%matplotlib inline 
boston_df = pd.read_csv('../input/property-assessment-fy2017.csv')
boston_df.info()
from IPython.display import IFrame
IFrame("https://data.boston.gov/dataset/property-assessment/resource/d195dc47-56f6-437c-80a8-7acbb8a2aa6d/view/a7b8e255-42da-4266-b022-65da0f6a8d61", width=700, height=400)
boston = pd.DataFrame(boston_df, columns=['PID','ZIPCODE','LU','OWNER','AV_TOTAL','GROSS_TAX','YR_BUILT','YR_REMOD','GROSS_AREA','LIVING_AREA','NUM_FLOORS'])
# A quick view of the new dataframe 
boston.head()
boston.describe()
boston['GROSS_TAX'] = boston['GROSS_TAX']/100
boston.sort_values('YR_BUILT', ascending=True).head()
boston.sort_values('AV_TOTAL', ascending=True).head()
boston = boston.replace(0,np.nan)
boston.describe()
boston.sort_values('NUM_FLOORS',ascending = True).head()
boston = boston.drop(boston.index[75595],axis=0)
boston['NUM_FLOORS'].value_counts()
boston.loc[boston['NUM_FLOORS'] == 2.7]
# I found this house on google map and it is actually 2.5 floors
boston.loc[boston['NUM_FLOORS'] == 2.7, 'NUM_FLOORS'] = 2.5
boston.loc[boston['NUM_FLOORS'] == 8.7]
# The floor number can not be observed from outside 

boston.loc[boston['NUM_FLOORS'] == 8.7, 'NUM_FLOORS'] = np.nan
boston.sort_values('YR_REMOD', ascending = True).head()
boston.loc[boston['YR_REMOD'] < boston['YR_BUILT'], 'YR_REMOD'] = np.nan
order = boston['GROSS_AREA'].value_counts()
order.sort_index().head()
boston.loc[boston['GROSS_AREA'] == 1, 'GROSS_AREA'] = np.nan
boston.loc[boston['LIVING_AREA'] == 1, 'LIVING_AREA'] = np.nan
boston.loc[boston['GROSS_AREA'] == 2, 'GROSS_AREA'] = np.nan
boston.loc[boston['LIVING_AREA'] == 2, 'LIVING_AREA'] = np.nan
boston.loc[boston['GROSS_AREA'] == 3, 'GROSS_AREA'] = np.nan
boston.loc[boston['LIVING_AREA'] == 3, 'LIVING_AREA'] = np.nan
boston['LU'].value_counts().sort_values()
sns.factorplot('LU',data=boston,kind='count', size=6, order=['CD','R1','R2','R3','CM','E','RL','CP','C','A','RC','R4','CL','CC','EA','I','AH'])
boston_A = boston[boston['LU'] == 'A']
boston_CD = boston[boston['LU'] == 'CD']
boston_R1 = boston[boston['LU'] == 'R1']
boston_R2 = boston[boston['LU'] == 'R2']
boston_R3 = boston[boston['LU'] == 'R3']
boston_R4 = boston[boston['LU'] == 'R4']
boston_Residential = pd.concat([boston_A, boston_CD, boston_R1, boston_R2, boston_R3, boston_R4])
boston_C = boston[boston['LU'] == 'C']
boston_CC = boston[boston['LU'] == 'CC']
boston_Commercial = pd.concat([boston_C, boston_CC])
boston_Residential['PID'].nunique()
boston_Commercial['PID'].nunique()
Total_Residential = boston_Residential['AV_TOTAL'].sum()
print('The total assessed value of the residential properties in the Boston area is ${:,.2f}'.format(Total_Residential))

Total_Commercial = boston_Commercial['AV_TOTAL'].sum()
print('The total assessed value of the commercial properties in the Boston area is ${:,.2f}'.format(Total_Commercial))
Tax_Residential = boston_Residential['GROSS_TAX'].sum()
print('The total recorded gross tax for residential properties in the Boston area is ${:,.2f}'.format(Tax_Residential))
Tax_Commercial = boston_Commercial['GROSS_TAX'].sum()
print('The total recorded gross tax for commercial properties in the Boston area is ${:,.2f}'.format(Tax_Commercial))
boston_Residential['AV_TOTAL'].describe()
# in order to eliminate the extreme values, we would find an assessed value for our histogram. We are only going to visaulize a part of it
boston_Residential['AV_TOTAL'].quantile(0.90)
plt.hist('AV_TOTAL',data=boston_Residential, range=[0, 1000000], rwidth=0.8)
boston_Commercial['AV_TOTAL'].quantile(0.90)
plt.hist('AV_TOTAL',data=boston_Commercial, range=[0, 20000000], rwidth=0.8)
Top_Properties_Residential = boston_Residential.sort_values('AV_TOTAL', ascending = False)[:10]
Top_Properties_Residential
Top_Properties_Commercial = boston_Commercial.sort_values('AV_TOTAL', ascending = False)[:10]
Top_Properties_Commercial 
corrmat = boston_Residential.corr()
f, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(corrmat, vmax=1, vmin=-1, square=True);
plt.title('Residential', fontsize=20)
corrmat = boston_Commercial.corr()
f, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(corrmat, vmax=1,vmin=-1, square=True);
plt.title('Commercial',fontsize=20)

f, ax = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(x='LU', y="NUM_FLOORS", data=boston)
fig.axis(ymin=0, ymax=62);
# Extract and clean up th values 

boston_year_built = boston[boston['YR_BUILT'].notnull()]
construction_year = boston_year_built['YR_BUILT'].value_counts()
df_years_built = pd.DataFrame({
    construction_year.name: construction_year.index.tolist(), 
    'COUNTS': construction_year.values
})

df_years_built = df_years_built.sort_values('YR_BUILT', ascending = True)
df_years_built = df_years_built.reset_index(drop = True)
df_years_built['YR_BUILT'] = df_years_built['YR_BUILT'].astype(int)
df_years_built = DataFrame(df_years_built, columns=['YR_BUILT', 'COUNTS'])
df_years_built.describe()
df_hot_construction = df_years_built.sort_values('COUNTS', ascending=False)
df_hot_construction[:10]
boston_year_remodel = boston[boston['YR_REMOD'].notnull()]
remodel_year = boston_year_remodel['YR_REMOD'].value_counts()
df_years_remodel = pd.DataFrame({
    remodel_year.name: remodel_year.index.tolist(), 
    'COUNTS': remodel_year.values
})

df_years_remodel = df_years_remodel.sort_values('YR_REMOD', ascending = True)
df_years_remodel = df_years_remodel.reset_index(drop = True)
df_years_remodel['YR_REMOD'] = df_years_remodel['YR_REMOD'].astype(int)
df_years_remodel = DataFrame(df_years_remodel, columns=['YR_REMOD', 'COUNTS'])
df_years_remodel.describe()
df_hot_remodel = df_years_remodel.sort_values('COUNTS', ascending=False)
df_hot_remodel[:10]
