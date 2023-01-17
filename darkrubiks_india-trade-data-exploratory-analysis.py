import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

df_export = pd.read_csv("../input/india-trade-data/2018-2010_export.csv")

df_import = pd.read_csv("../input/india-trade-data/2018-2010_import.csv")
df_export.head()
df_import.head()
df_export.isna().sum()
df_export['value'].fillna(0,inplace=True)
df_import.isna().sum()
df_import['value'].fillna(0,inplace=True)
df_export[df_export['country']=='UNSPECIFIED'].head()
df_import[df_import['country']=='UNSPECIFIED'].head()
df_export['year'].unique()
df_import['year'].unique()
df_export.duplicated().sum()
df_import.duplicated().sum()
df_import.drop_duplicates(keep="first",inplace=True)
print('Total exportation value: {:.2f}'.format(df_export['value'].sum())) 

print('Total importation value: {:.2f}'.format(df_import['value'].sum())) 
pd.DataFrame(df_export['country'].value_counts()[:10])
pd.DataFrame(df_import['country'].value_counts()[:10])
pd.DataFrame(df_export['Commodity'].value_counts()[:10])
pd.DataFrame(df_import['Commodity'].value_counts()[:10])
pd.DataFrame(df_export.groupby(df_export['Commodity'])['value'].sum().sort_values(ascending=False)[:10])
pd.DataFrame(df_import.groupby(df_import['Commodity'])['value'].sum().sort_values(ascending=False)[:10])
df_export[df_export['value']==df_export['value'].max()]
df_import[df_import['value']==df_import['value'].max()]
plt.figure(figsize=(10,10))

ax = sns.lineplot(data=df_export,x='year',y='value',err_style=None)

ax = sns.lineplot(data=df_import,x='year',y='value',err_style=None)

ax = plt.legend(['Export','Import'])