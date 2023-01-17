import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/countries of the world.csv')
df.shape
# get the data info
df.info()
df.head()
import string
# Convert the object datatypes to float, replace the ',', with '.' and then convert
df['Pop. Density (per sq. mi.)'] = df['Pop. Density (per sq. mi.)'].str.replace(',','.').apply(float)
df['Coastline (coast/area ratio)']= df['Coastline (coast/area ratio)'].str.replace(',','.').apply(float)
df['Net migration']= df['Net migration'].str.replace(',','.').apply(float)
df['Infant mortality (per 1000 births)']= df['Infant mortality (per 1000 births)'].str.replace(',','.').apply(float)
df['Literacy (%)']= df['Literacy (%)'].str.replace(',','.').apply(float)
df['Phones (per 1000)']= df['Phones (per 1000)'].str.replace(',','.').apply(float)
df['Arable (%)']= df['Arable (%)'].str.replace(',','.').apply(float)
df['Crops (%)']= df['Crops (%)'].str.replace(',','.').apply(float)
df['Other (%)']= df['Other (%)'].str.replace(',','.').apply(float)
df['Birthrate']= df['Birthrate'].str.replace(',','.').apply(float)
df['Deathrate']= df['Deathrate'].str.replace(',','.').apply(float)
df['Agriculture']= df['Agriculture'].str.replace(',','.').apply(float)
df['Industry']= df['Industry'].str.replace(',','.').apply(float)
df['Service']= df['Service'].str.replace(',','.').apply(float)
df['Climate']= df['Climate'].str.replace(',','.').apply(float)
df.head()
df.describe()
df.columns
df['Region'] =   df['Region'].str.replace(' ','') # trim the white spaces
df['Region'].unique()
sns.countplot(y='Region',data=df)
df_new = df.fillna(value=0.0)
plt.subplots(figsize=(14,6))
sns.boxplot(x='Region',y='Population',data=df_new)
#skipping the two outliers, to get a clear picture about other regions
df_skipAsia_NA = df[(df['Region'] != 'ASIA(EX.NEAREAST)') & (df['Region'] != 'NORTHERNAMERICA' )]
plt.subplots(figsize=(12,6))
sns.boxplot(x='Region',y='Population',data=df_skipAsia_NA)
plt.figure(figsize=(12,6))
sns.relplot(x='Population',y='Area (sq. mi.)',data=df_new,hue='Region')
#df_new['Pop. Density (per sq. mi.)'] = df_new['Pop. Density (per sq. mi.)'].str.replace(',','.').apply(float)
#### trying to Cluster the data into various groups, starting with all the fields
from sklearn.cluster import KMeans
kms = KMeans(n_clusters=4)
kms.fit(df_new.drop(['Country','Region'],axis=1))
kms.cluster_centers_
df_new['Cluster'] = kms.labels_
df_new.head()
plt.subplots(figsize=(12,6))
sns.scatterplot(x='Arable (%)',y='Crops (%)',hue='Cluster',data=df_new,markers='*',palette='coolwarm')
df_new.columns
### Trying to create a dataframe for classfying based on the agriculture data from the dataframe.
df_agri =  df_new[['Country','Region','Literacy (%)','Arable (%)','Crops (%)', 'Climate','Agriculture']]
kms2 = KMeans(n_clusters=2)
kms2.fit(df_agri.drop(['Country','Region'],axis=1))
kms2.labels_
df_agri['Cluster'] = kms2.labels_
df_agri.head(10)
plt.subplots(figsize=(12,6))
sns.scatterplot(x='Agriculture',y='Crops (%)',hue='Cluster',data=df_agri,markers='*',palette='plasma')
