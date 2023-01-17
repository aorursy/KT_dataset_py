import pandas as pd

import numpy as np

from pprint import pprint
df_orig = pd.read_csv("../input/road_eqr_carpda_1_Data.csv", encoding="ISO-8859-1",na_values=[":"])

df = df_orig.copy()

print(df.head())

print(df.tail())

print(df.columns)
print(df["Flag and Footnotes"].unique())

df[df["Flag and Footnotes"] == 'd']
# 'd' means "definition differs"

# Assume for simplicity that it is the same



df = df.drop("Flag and Footnotes",axis=1)
df.UNIT.unique()
df = df.drop("UNIT",axis=1)
df.columns = ["time","country","motor","value"]

df.head()
df.country.unique()
df.country.replace(to_replace='Germany (until 1990 former territory of the FRG)', value='Germany', inplace=True)

df.country.unique()
df[[pd.isnull(x) for x in df.value]]
df.fillna(0,inplace=True)
df.dtypes
def remove_comma(x):

    return str(x).replace(',','')

df.value = df.value.apply(remove_comma).astype(int)
df.dtypes
df['motor'].unique()
# Will also remove \xa0 etc.

df.motor = df.motor.str.strip()



df.motor.unique()
mot_num = df[['motor','value']].groupby('motor').sum().loc[:,'value'].squeeze()

mot_num.plot.bar()
print(mot_num['Diesel'])

print(mot_num['Diesel (excluding hybrids)'] + mot_num['Hybrid diesel-electric'] + mot_num['Plug-in hybrid diesel-electric'])
print(df[df.motor == 'Diesel (excluding hybrids)'])
df[(df.time==2017) & (df.country=='Germany')]
#df.to_csv('../data/road_eqr_carpda_pre.csv',header=True,index=False)
print(len(df.query('motor == "Diesel (excluding hybrids)" and value == 0')))

len(df.query('motor == "Diesel" and value==0'))
df2 = df.copy()
df2=df2[~df.motor.isin(['Diesel (excluding hybrids)','Petrol (excluding hybrids)','Alternative Energy'])]

df2.motor.unique()
df2.motor.replace('Petroleum products', 'petroleum', inplace=True)

df2.motor.replace('Diesel', 'diesel', inplace=True)

df2.motor.replace('Electricity', 'electricity', inplace=True)



df2.motor.replace(['Hybrid electric-petrol', 'Plug-in hybrid petrol-electric', 'Hybrid diesel-electric', 'Plug-in hybrid diesel-electric'], 'hybrid', inplace=True)

df2.motor.replace(['Liquefied petroleum gases (LPG)', 'Natural Gas', 'Hydrogen and fuel cells', 'Bioethanol', 'Biodiesel', 'Bi-fuel', 'Other'], 'other', inplace=True)



df2.motor.unique()
df2.describe()
df2.head()
df3 = pd.DataFrame(columns=df2.columns)

for (i,row) in df2.drop('value',axis=1).drop_duplicates().iterrows():

    partial = df2[(df2.time==row.time) & (df2.country==row.country) & (df2.motor==row.motor)]

    row['value'] = partial.value.sum()

    #print(row.motor, len(partial))

    df3 = df3.append(row)

df3.head()
df3[df3.country.isin(['Netherlands','Belgium','Norway'])].head()
elec_by_country = df3[df3.motor=='electricity'].groupby('country').value.max()

elec_by_country[elec_by_country == 0]
#df3.to_csv('../data/road_eqr_carpda_cleaned.csv',index=False,header=True)