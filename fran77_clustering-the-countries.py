# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/unsupervised-learning-on-country-data/Country-data.csv')

df_dic = pd.read_csv('/kaggle/input/unsupervised-learning-on-country-data/data-dictionary.csv')
for i, row in df_dic.iterrows():

    print(row['Column Name'], '--->', row['Description'])
data.head()
# Number of countries

print(len(data))
plt.figure(figsize=(12,8))

sns.kdeplot(data.child_mort)
# Country with min child mortality

print(data[data.child_mort == data.child_mort.min()].country.values[0], '(%d)' %(data.child_mort.min()))
# Country with min child mortality

print(data[data.child_mort == data.child_mort.max()].country.values[0], '(%d)' %(data.child_mort.max()))
# Countries that are not in the ISO Code



data['country'] = data['country'].str.replace('Cape Verde', 'Cabo Verde')

data['country'] = data['country'].str.replace('Congo, Dem. Rep.', 'Congo, The Democratic Republic of the')

data['country'] = data['country'].str.replace('Congo, Rep.', 'Republic of the Congo')

data['country'] = data['country'].str.replace('Macedonia, FYR', 'North Macedonia')

data['country'] = data['country'].str.replace('Micronesia, Fed. Sts.', 'Micronesia, Federated States of')

data['country'] = data['country'].str.replace('South Korea', 'Korea, Republic of')

data['country'] = data['country'].str.replace('St. Vincent and the Grenadines', 'Saint Vincent and the Grenadines')
import pycountry

import plotly.express as px

import pandas as pd



list_countries = data['country'].unique().tolist()



d_country_code = {}  # To hold the country names and their ISO

for country in list_countries:

    try:

        country_data = pycountry.countries.search_fuzzy(country)

        country_code = country_data[0].alpha_3

        d_country_code.update({country: country_code})

    except:

        print('could not add ISO 3 code for ->', country)

        # If could not find country, make ISO code ' '

        d_country_code.update({country: ' '})



# create a new column iso_alpha in the data

# and fill it with appropriate iso 3 code

for k, v in d_country_code.items():

    data.loc[(data.country == k), 'iso_alpha'] = v
data[data['iso_alpha'].duplicated(keep=False)]
data.loc[112,'iso_alpha'] = 'NER'
fig = px.choropleth(data_frame = data,

                    locations= "iso_alpha",

                    color= "child_mort",  # value in column 'Confirmed' determines color

                    hover_name= "country",

                    color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green

                    )



fig.show()
plt.figure(figsize=(12,8))

sns.kdeplot(data.exports)
# Country with min exports

print(data[data.exports == data.exports.min()].country.values[0], '(%d)' %(data.exports.min()))
# Country with max exports

print(data[data.exports == data.exports.max()].country.values[0], '(%d)' %(data.exports.max()))
fig = px.choropleth(data_frame = data,

                    locations= "iso_alpha",

                    color= "exports",  # value in column 'Confirmed' determines color

                    hover_name= "country",

                    color_continuous_scale= 'RdYlGn',  #  color scale red, yellow green

                    )



fig.show()
plt.figure(figsize=(12,8))

sns.kdeplot(data.health)
# Country with min health

print(data[data.health == data.health.min()].country.values[0], '(%d)' %(data.health.min()))
# Country with max health

print(data[data.health == data.health.max()].country.values[0], '(%d)' %(data.health.max()))
fig = px.choropleth(data_frame = data,

                    locations= "iso_alpha",

                    color= "health",  # value in column 'Confirmed' determines color

                    hover_name= "country",

                    color_continuous_scale= 'RdYlGn',  #  color scale red, yellow green

                    )



fig.show()
# Country with min imports

print(data[data.imports == data.imports.min()].country.values[0], '(%d)' %(data.imports.min()))
# Country with max health

print(data[data.imports == data.imports.max()].country.values[0], '(%d)' %(data.imports.max()))
fig = px.choropleth(data_frame = data,

                    locations= "iso_alpha",

                    color= "imports",  # value in column 'Confirmed' determines color

                    hover_name= "country",

                    color_continuous_scale= 'RdYlGn',  #  color scale red, yellow green

                    )



fig.show()
plt.figure(figsize=(12,8))

sns.kdeplot(data.income)
# Country with min income

print(data[data.income == data.income.min()].country.values[0], '(%d)' %(data.income.min()))
# Country with income income

print(data[data.income == data.income.max()].country.values[0], '(%d)' %(data.income.max()))
fig = px.choropleth(data_frame = data,

                    locations= "iso_alpha",

                    color= "income",  # value in column 'Confirmed' determines color

                    hover_name= "country",

                    color_continuous_scale= 'RdYlGn',  #  color scale red, yellow green

                    )



fig.show()
plt.figure(figsize=(12,8))

sns.kdeplot(data.inflation)
# Country with min inflation

print(data[data.inflation == data.inflation.min()].country.values[0], '(%d)' %(data.inflation.min()))
# Country with min inflation

print(data[data.inflation == data.inflation.max()].country.values[0], '(%d)' %(data.inflation.max()))
data[data.inflation == data.inflation.max()]
fig = px.choropleth(data_frame = data,

                    locations= "iso_alpha",

                    color= "inflation",  # value in column 'Confirmed' determines color

                    hover_name= "country",

                    color_continuous_scale= 'RdYlGn',  #  color scale red, yellow green

                    )



fig.show()
plt.figure(figsize=(12,8))

sns.kdeplot(data.life_expec)
# Country with min life_expec

print(data[data.life_expec == data.life_expec.min()].country.values[0], '(%d)' %(data.life_expec.min()))
# Country with min life_expec

print(data[data.life_expec == data.life_expec.max()].country.values[0], '(%d)' %(data.life_expec.max()))
fig = px.choropleth(data_frame = data,

                    locations= "iso_alpha",

                    color= "life_expec",  # value in column 'Confirmed' determines color

                    hover_name= "country",

                    color_continuous_scale= 'RdYlGn',  #  color scale red, yellow green

                    )



fig.show()
plt.figure(figsize=(12,8))

sns.kdeplot(data.total_fer)
# Country with min total_fer

print(data[data.total_fer == data.total_fer.min()].country.values[0], '(%d)' %(data.total_fer.min()))
# Country with max total_fer

print(data[data.total_fer == data.total_fer.max()].country.values[0], '(%d)' %(data.total_fer.max()))
fig = px.choropleth(data_frame = data,

                    locations= "iso_alpha",

                    color= "total_fer",  # value in column 'Confirmed' determines color

                    hover_name= "country",

                    color_continuous_scale= 'RdYlGn',  #  color scale red, yellow green

                    )



fig.show()
plt.figure(figsize=(12,8))

sns.kdeplot(data.gdpp)
# Country with min gdpp

print(data[data.gdpp == data.gdpp.min()].country.values[0], '(%d)' %(data.gdpp.min()))
# Country with max total_fer

print(data[data.gdpp == data.gdpp.max()].country.values[0], '(%d)' %(data.gdpp.max()))
fig = px.choropleth(data_frame = data,

                    locations= "iso_alpha",

                    color= "gdpp",  # value in column 'Confirmed' determines color

                    hover_name= "country",

                    color_continuous_scale= 'RdYlGn',  #  color scale red, yellow green

                    )



fig.show()
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
# Normalize the columns, the majority are skawed

# Normalize the columns (right skew)



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

data['child_mort'] = scaler.fit_transform(data['child_mort'].values.reshape(-1, 1))

data['exports'] = scaler.fit_transform(data['exports'].values.reshape(-1, 1))

data['health'] = scaler.fit_transform(data['health'].values.reshape(-1, 1))

data['imports'] = scaler.fit_transform(data['imports'].values.reshape(-1, 1))

data['income'] = scaler.fit_transform(data['income'].values.reshape(-1, 1))

data['inflation'] = scaler.fit_transform(data['inflation'].values.reshape(-1, 1))

data['life_expec'] = scaler.fit_transform(data['life_expec'].values.reshape(-1, 1))

data['total_fer'] = scaler.fit_transform(data['total_fer'].values.reshape(-1, 1))

data['gdpp'] = scaler.fit_transform(data['gdpp'].values.reshape(-1, 1))
# Remove country and iso_code

countries = data['country']

iso_alpha = data['iso_alpha']

data = data.drop(['country', 'iso_alpha'], axis=1)
data.head()
# Calculate sum of squared distances

ssd = []

K = range(1,10)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(data)

    ssd.append(km.inertia_)
# Plot sum of squared distances / elbow method

plt.figure(figsize=(10,6))

plt.plot(K, ssd, 'bx-')

plt.xlabel('k')

plt.ylabel('ssd')

plt.title('Elbow Method For Optimal k')

plt.show()
# Best number of clusters is 3
# Create and fit model

kmeans = KMeans(n_clusters=3)

model = kmeans.fit(data)
pred = model.labels_

data['cluster'] = pred
data.head()
# Create PCA for data visualization / Dimensionality reduction to 2D graph

from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca_model = pca.fit_transform(data)

data_transform = pd.DataFrame(data = pca_model, columns = ['PCA1', 'PCA2'])

data_transform['Cluster'] = pred
data_transform.head()
plt.figure(figsize=(8,8))

g = sns.scatterplot(data=data_transform, x='PCA1', y='PCA2', palette=sns.color_palette()[:3], hue='Cluster')

title = plt.title('Countries Clusters with PCA')
data['country'] = countries

data['iso_alpha'] = iso_alpha
fig = px.choropleth(data_frame = data,

                    locations= "iso_alpha",

                    color= "cluster",  # value in column 'Confirmed' determines color

                    hover_name= "country",

                    color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green

                    )



fig.show()