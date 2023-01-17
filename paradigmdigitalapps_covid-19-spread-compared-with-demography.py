import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go

from fbprophet import Prophet

import pycountry

import plotly.express as px

from datetime import timedelta



from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)



countries_of_the_world = pd.read_csv("../input/countries-demographic-and-economic-data/countries of the world.csv")
df.shape
countries_of_the_world.shape
countries_of_the_world["Country"] = countries_of_the_world["Country"].str.strip()
list(countries_of_the_world["Country"])[:10]
df_countries = df.groupby(["Country","Last Update"])[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df_countries
df_countries_start = df_countries.groupby("Country", as_index=False).nth(0)

df_countries_start.reset_index(drop=True, inplace=True)

df_countries_start
days_later = 15 # x days after the first observation of a country

df_countries_start["xDays Later Confirmed"] = 0 # create a new column to register number of observations

countries_list = list(df_countries_start["Country"].drop_duplicates()) # Check if there is any duplicate data



for item in countries_list:

    

    try:

        print(item)

        

        first_date = min(df_countries_start[df_countries_start["Country"] == item]["Last Update"])

        xdays_later = first_date + timedelta(days=days_later)

        announce_date = min(df_countries[(df_countries["Country"] == item) & (df_countries["Last Update"] > xdays_later)]["Last Update"])

        confirmed_sick = df_countries[(df_countries["Country"] == item) & (df_countries["Last Update"] == announce_date)]["Confirmed"]

        death_sick = df_countries[(df_countries["Country"] == item) & (df_countries["Last Update"] == announce_date)]["Deaths"]

        recovered_sick = df_countries[(df_countries["Country"] == item) & (df_countries["Last Update"] == announce_date)]["Recovered"]

        

        print(confirmed_sick.values)

        

        df_countries_start.loc[df_countries_start['Country'] == item, 'xDays Later Confirmed'] = confirmed_sick.values

        df_countries_start.loc[df_countries_start['Country'] == item, 'xDays Later Death'] = death_sick.values

        df_countries_start.loc[df_countries_start['Country'] == item, 'xDays Later Recovered'] = recovered_sick.values



    except:

        continue

    

df_countries_start.head(20)
df_countries_dem = df_countries_start.merge(countries_of_the_world, left_on='Country', right_on='Country')
df_countries_dem.shape
df_countries_dem[df_countries_dem["xDays Later Confirmed"] > 0].head(15)
df_research = df_countries_dem[df_countries_dem["xDays Later Confirmed" ] > 0]
df_research.shape
df_research = df_research.drop(['Country', 'Last Update','Region'], axis=1)
df_research.dropna(inplace=True)
df_research.reset_index(drop=True, inplace=True)
for i in range (0,23):

    if df_research.iloc[:,i].dtypes == object:

        df_research.iloc[:,i] = df_research.iloc[:,i].str.replace(',', '.').astype(float)

        
df_research.corr()
plt.figure(figsize=(13,13))



sns.heatmap(df_research.corr().round(1), vmax=1, square=True,annot=True,cmap='coolwarm')



plt.title('Correlation between different fearures')
pd.set_option('display.max_columns', None)
df_research = df_countries_dem[df_countries_dem["xDays Later Confirmed" ] > 0]
df_research = df_research.drop(['Country', 'Last Update','Region'], axis=1)

df_research = df_research.drop(['Confirmed', 'Deaths','Recovered'], axis=1)

df_research = df_research.drop(['xDays Later Confirmed', 'xDays Later Death','xDays Later Recovered'], axis=1)
df_research.dropna(inplace=True)
df_research.head(10)
df_research.reset_index(drop=True, inplace=True)
df_research.shape
for i in range (0,18):

    if df_research.iloc[:,i].dtypes == object:

        df_research.iloc[:,i] = df_research.iloc[:,i].str.replace(',', '.').astype(float)

        pass

 
X = df_research.iloc[:,:]
X.shape
from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(X)

print(standardized_data.shape)
covar_matrix = np.matmul(standardized_data.T , standardized_data)
covar_matrix.shape
from scipy.linalg import eigh 
values, vectors = eigh(covar_matrix,eigvals=(16,17))
values = values.real

print(values)
#transpose

vectors = vectors.T

vectors.shape
new_coordinates = np.matmul(vectors, standardized_data.T)

print ("Resultant at new data shape: ", vectors.shape, "*", standardized_data.T.shape," = ", new_coordinates.shape)
df_research.head()
df_research.reset_index(drop=True, inplace=True)
new_coordinates.shape
new_coordinates = np.vstack((new_coordinates)).T



df = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal"))
df_pca_ext = pd.concat([df, df_research], axis=1).reindex(df.index)
df_pca_ext.head()
sns.set(style="ticks")





sns.FacetGrid(df_pca_ext, height=10, hue="Climate").map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.title('PCA visualization of sequences')

plt.show()
df_research = df_countries_dem[df_countries_dem["xDays Later Confirmed" ] > 0]
df_research = df_research.drop(['Country', 'Last Update','Region'], axis=1)
df_research.reset_index(drop=True, inplace=True)
for i in range (0,24):

    if df_research.iloc[:,i].dtypes == object:

        df_research.iloc[:,i] = df_research.iloc[:,i].str.replace(',', '.').astype(float)

        pass

df_research.head(5)
df_research.dropna(inplace=True)
df_research.reset_index(drop=True, inplace=True)
X = df_research.iloc[:,:]
standardized_data = StandardScaler().fit_transform(X)

covar_matrix = np.matmul(standardized_data.T , standardized_data)

values, vectors = eigh(covar_matrix,eigvals=(22,23))

vectors = vectors.T

new_coordinates = np.matmul(vectors, standardized_data.T)

new_coordinates = np.vstack((new_coordinates)).T

df = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal"))
df_pca_ext = pd.concat([df, df_research], axis=1).reindex(df.index)
df_pca_ext.head()
df_pca_ext['Rel Death'] = df_pca_ext['xDays Later Death']/df_pca_ext['Population']

df_pca_ext['Rel Migr'] = df_pca_ext['xDays Later Death']/df_pca_ext['Net migration']

df_pca_ext['Rel Migr range'] = (df_pca_ext['Net migration']/2).round(0)*2
sns.set(style="ticks")

sns.FacetGrid(df_pca_ext, height=8, hue="Rel Death").map(plt.scatter, '1st_principal', '2nd_principal')

plt.title('PCA visualization of sequences')

plt.show()
sns.set(style="ticks")

sns.FacetGrid(df_pca_ext, height=8, hue="Rel Migr").map(plt.scatter, '1st_principal', '2nd_principal')

plt.title('PCA visualization of sequences')

plt.show()
sns.set(style="ticks")

sns.FacetGrid(df_pca_ext, height=8, hue="Rel Migr range").map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.title('PCA visualization of sequences')

plt.show()
list(df_countries_dem[df_countries_dem["Net migration"] > '4']["Country"])