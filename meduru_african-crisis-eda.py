import matplotlib.pyplot as plt

import plotly.express as px

import pandas as pd

import seaborn as sns

import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

sns.set(style="whitegrid")

baslik_font = {'family': 'arial', 'color': 'darkred','weight': 'bold','size': 13 }

eksen_font  = {'family': 'arial', 'color': 'darkblue','weight': 'bold','size': 10 }
df = pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv', index_col='case')

df.head()
print("Checking the columns in the dataset.")

df.columns
df.info()
df.isnull().sum()
df.describe()
# Dataset is non-uniform and recently formed countries have least data

plt.figure(figsize=(8,8))

counts= df['country'].value_counts()

country=counts.index

explode = (0.2, 0.1, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0)

plt.pie(counts, explode=explode,labels=country,autopct='%1.1f%%')

plt.show()
# Let me visualize the country that has high number of banking_crisis 

plt.figure(figsize=(10,5))

sns.set_style('whitegrid')

sns.countplot(x='country', data=df, palette='gist_rainbow', hue='banking_crisis')

plt.title('Graph of Country Representation based on Banking Crisis')

plt.xticks(rotation = 60)

plt.xlabel(None)

plt.show()
plt.figure(figsize=(20,14))

plt.subplot(221)

sns.set_style('whitegrid')

sns.countplot(x='country', data=df, palette='gist_rainbow', hue='systemic_crisis')

plt.title('Graph of Country Representation based on systemic_crisis')

plt.xticks(rotation = 90)

plt.xlabel(None)

plt.subplot(222)

sns.set_style('whitegrid')

sns.countplot(x='country', data=df, palette='gist_rainbow', hue='inflation_crises')

plt.title('Graph of Country Representation based on inflation_crises')

plt.xticks(rotation = 90)

plt.xlabel(None)

plt.subplot(223)

sns.set_style('whitegrid')

sns.countplot(x='country', data=df, palette='gist_rainbow', hue='banking_crisis')

plt.title('Graph of Country Representation based on Banking Crisis')

plt.xticks(rotation = 90)

plt.xlabel(None)

plt.tight_layout()

plt.show()

plt.figure(figsize=(15,8))

plt.subplot(121)

sns.barplot(x='systemic_crisis',y='country',data=df, palette='Paired')

plt.ylabel(None)

plt.title("Systemic_Crisis", fontdict=baslik_font)

plt.subplot(122)

plt.title("İnflation_Crises", fontdict=baslik_font)

sns.barplot(y='country',x='inflation_crises',data=df,palette='Paired')

plt.ylabel(None)

plt.tight_layout()

plt.show()
#The inflation and exchange rates are good indicator for economic health for the country

plt.figure(figsize=(15,8))

count = 1

for country in df.country.unique():

    plt.subplot(len(df.country.unique())/4,5,count)

    count+=1

    sns.lineplot(df[df.country==country]['year'],df[df.country==country]['exch_usd'], color="darkred")

    sns.lineplot(df[df.country==country]['year'],df[df.country==country]['inflation_annual_cpi'],color="darkblue")

    plt.subplots_adjust(wspace=0.4,hspace=0.5)

    plt.xlabel(None)

    plt.ylabel('İnflation/Exchange Rates')

    plt.title(country,baslik_font)
df["banking_crisis_new"]=df.banking_crisis.replace({'crisis':1,'no_crisis':0})

df.head(1)
df.corr()
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



X = df[['inflation_annual_cpi', 'exch_usd']]

X = StandardScaler().fit_transform(X)



sklearn_pca = PCA(n_components=1)

df["PCA_1"] = sklearn_pca.fit_transform(X)



print(

    'The percentage of total variance in the dataset explained by each',

    'component from Sklearn PCA.\n',

    sklearn_pca.explained_variance_ratio_

)
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



X = df[['systemic_crisis','currency_crises', 'inflation_crises']]

X = StandardScaler().fit_transform(X)



sklearn_pca = PCA(n_components=1)

df["PCA_CRİSİS_2"] = sklearn_pca.fit_transform(X)



print(

    'The percentage of total variance in the dataset explained by each',

    'component from Sklearn PCA.\n',

    sklearn_pca.explained_variance_ratio_

)
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



X = df[['domestic_debt_in_default', 'sovereign_external_debt_default','gdp_weighted_default']]

X = StandardScaler().fit_transform(X)



sklearn_pca = PCA(n_components=1)

df["PCA_DEBT_3"] = sklearn_pca.fit_transform(X)



print(

    'The percentage of total variance in the dataset explained by each',

    'component from Sklearn PCA.\n',

    sklearn_pca.explained_variance_ratio_

)
plt.figure(figsize=(10,7))

sns.heatmap(df.corr(), cmap='magma', annot=True)

plt.ylim(0,15)
a=df.sort_values(by=['year'])



fig = px.choropleth(a,locations="cc3",

                    color="exch_usd",animation_frame="year", 

                    hover_name="country", 

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(

    title_text = 'Africa Döviz Kuru',

    geo_scope='africa', 

)
fig = px.choropleth(a,locations="cc3",

                    color="inflation_annual_cpi",animation_frame="year", 

                    hover_name="country", 

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(

    title_text = 'Africa Enflasyon Durumu',

    geo_scope='africa', 

)
a1=df.groupby('country').sum()

a1['cc2']=['DZA', 'AGO', 'CAF', 'EGY','CIV',  'KEN', 'MUS', 'MAR', 'NGA',

       'ZAF', 'TUN', 'ZMB', 'ZWE']

a1['country1']=['Algeria', 'Angola', 'Central African Republic',

       'Egypt','Ivory Coast', 'Kenya', 'Mauritius', 'Morocco', 'Nigeria',

       'South Africa', 'Tunisia', 'Zambia', 'Zimbabwe']


fig = px.choropleth(a1,locations="cc2",

                    color="banking_crisis_new",

                    hover_name="country1", 

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(

    title_text = 'Africa Enflasyon Durumu',

    geo_scope='africa',

)