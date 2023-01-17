# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import missingno as msno
from datetime import datetime, timedelta


import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
df_kiva_loans = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
df_loc = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
df_themes = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
df_mpi = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")

df_kiva_loans.head(5)
msno.matrix(df_kiva_loans);
df_kiva_loans.describe(include = 'all')
countries = df_kiva_loans['country'].value_counts()[df_kiva_loans['country'].value_counts()>3400]
list_countries = list(countries.index) #this is the list of countries that will be most used.
plt.figure(figsize=(13,8))
sns.barplot(y=countries.index, x=countries.values, alpha=0.6)
plt.title("Number of borrowers per country", fontsize=16)
plt.xlabel("Nb of borrowers", fontsize=16)
plt.ylabel("Countries", fontsize=16)
plt.show();
df_kiva_loans['borrower_genders']=[elem if elem in ['female','male'] else 'group' for elem in df_kiva_loans['borrower_genders'] ]
#to replace values such as "woman, woman, woman, man"

borrowers = df_kiva_loans['borrower_genders'].value_counts()
labels = (np.array(borrowers.index))
values = (np.array((borrowers / borrowers.sum())*100))

trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent',
               textfont=dict(size=20),
                showlegend=True)

layout = go.Layout(
    title="Borrowers' genders"
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="Borrowers_genders")
plt.figure(figsize=(13,8))
sectors = df_kiva_loans['sector'].value_counts()
sns.barplot(y=sectors.index, x=sectors.values, alpha=0.6)
plt.xlabel('Number of loans', fontsize=16)
plt.ylabel("Sectors", fontsize=16)
plt.title("Number of loans per sector")
plt.show();
plt.figure(figsize=(15,10))
activities = df_kiva_loans['activity'].value_counts().head(50)
sns.barplot(y=activities.index, x=activities.values, alpha=0.6)
plt.ylabel("Activity", fontsize=16)
plt.xlabel('Number of loans', fontsize=16)
plt.title("Number of loans per activy", fontsize=16)
plt.show();
plt.figure(figsize=(12,8))
sns.distplot(df_kiva_loans['loan_amount'])
plt.ylabel("density estimate", fontsize=16)
plt.xlabel('loan amount', fontsize=16)
plt.title("KDE of loan amount", fontsize=16)
plt.show();
temp = df_kiva_loans['loan_amount']

plt.figure(figsize=(12,8))
sns.distplot(temp[~((temp-temp.mean()).abs()>3*temp.std())]);
plt.ylabel("density estimate", fontsize=16)
plt.xlabel('loan amount', fontsize=16)
plt.title("KDE of loan amount (outliers removed)", fontsize=16)
plt.show();
plt.figure(figsize=(15,8))
sns.boxplot(x='loan_amount', y="sector", data=df_kiva_loans);
plt.xlabel("Value of loan", fontsize=16)
plt.ylabel("Sector", fontsize=16)
plt.title("Sectors loans' amounts boxplots", fontsize=16)
plt.show();
round(df_kiva_loans.groupby(['sector'])['loan_amount'].median(),2)
temp = df_kiva_loans[df_kiva_loans['loan_amount']<2000]
plt.figure(figsize=(15,8))
sns.boxplot(x='loan_amount', y="sector", data=temp)
plt.xlabel("Value of loan", fontsize=16)
plt.ylabel("Sector", fontsize=16)
plt.title("Sectors loans' amounts boxplots", fontsize=16)
plt.show();
loans_dates = df_kiva_loans.dropna(subset=['disbursed_time', 'funded_time'], how='any', inplace=False)

dates = ['posted_time','disbursed_time','funded_time']
loans_dates[dates] = loans_dates[dates].applymap(lambda x : x.split('+')[0])

loans_dates[dates]=loans_dates[dates].apply(pd.to_datetime)
loans_dates['time_funding']=loans_dates['funded_time']-loans_dates['posted_time']
loans_dates['time_funding'] = loans_dates['time_funding'] / timedelta(days=1) 


#this last line gives us the value for waiting time in days and float format,
# for example: 3 days 12 hours = 3.5
temp = loans_dates['time_funding']

plt.figure(figsize=(12,8))
sns.distplot(temp[~((temp-temp.mean()).abs()>3*temp.std())]);
df_ctime = round(loans_dates.groupby(['country'])['time_funding'].median(),2)
df_camount = round(df_kiva_loans.groupby(['country'])['loan_amount'].median(),2)
df_camount = df_camount[df_camount.index.isin(list_countries)].sort_values()
df_ctime = df_ctime[df_ctime.index.isin(list_countries)].sort_values()

f,ax=plt.subplots(1,2,figsize=(20,10))

sns.barplot(y=df_camount.index, x=df_camount.values, alpha=0.6, ax=ax[0])
ax[0].set_title("Medians of funding amounts per loan country wise ")
ax[0].set_xlabel('Amount in dollars')
ax[0].set_ylabel("Country")

sns.barplot(y=df_ctime.index, x=df_ctime.values, alpha=0.6,ax=ax[1])
ax[1].set_title("Medians of waiting days per loan to be funded country wise  ")
ax[1].set_xlabel('Number of days')
ax[1].set_ylabel("")

plt.tight_layout()
plt.show();
df_repay = round(df_kiva_loans.groupby(['country'])['term_in_months'].median(),2)
df_repay = df_repay[df_repay.index.isin(list_countries)].sort_values()

df_kiva_loans['ratio_amount_duration']= df_kiva_loans['funded_amount']/df_kiva_loans['term_in_months'] 
temp = round(df_kiva_loans.groupby('country')['ratio_amount_duration'].median(),2)
temp = temp[temp.index.isin(list_countries)].sort_values()

f,ax=plt.subplots(1,2,figsize=(20,10))

sns.barplot(y=temp.index, x=temp.values, alpha=0.6, ax=ax[0])
ax[0].set_title("Ratio of amount of loan to repayment period per country", fontsize=16)
ax[0].set_xlabel("Ratio value", fontsize=16)
ax[0].set_ylabel("Country", fontsize=16)

sns.barplot(y=df_repay.index, x=df_repay.values, alpha=0.6,ax=ax[1])
ax[1].set_title("Medians of number of months per repayment, per country",fontsize=16)
ax[1].set_xlabel('Number of months', fontsize=16)
ax[1].set_ylabel("")

plt.tight_layout()
plt.show();
lenders = pd.read_csv('../input/additional-kiva-snapshot/lenders.csv')
lenders.head()
lender_countries = lenders.groupby(['country_code']).count()[['permanent_name']].reset_index()
lender_countries.columns = ['country_code', 'Number of Lenders']
lender_countries.sort_values(by='Number of Lenders', ascending=False,inplace=True)
lender_countries.head(7)
countries_data = pd.read_csv( '../input/additional-kiva-snapshot/country_stats.csv')
countries_data.head()
countries_data = pd.read_csv( '../input/additional-kiva-snapshot/country_stats.csv')
lender_countries = pd.merge(lender_countries, countries_data[['country_name','country_code']],
                            how='inner', on='country_code')

data = [dict(
        type='choropleth',
        locations=lender_countries['country_name'],
        locationmode='country names',
        z=np.log10(lender_countries['Number of Lenders']+1),
        colorscale='Viridis',
        reversescale=False,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='Lenders'),
    )]
layout = dict(
    title = 'Lenders per country in a logarithmic scale ',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='lenders-map')
import matplotlib as mpl 
from wordcloud import WordCloud, STOPWORDS
import imageio

heart_mask = imageio.imread('../input/poverty-indicators/heart_msk.jpg') #because displaying this wordcloud as a heart seems just about right :)

mpl.rcParams['figure.figsize']=(12.0,8.0)    #(6.0,4.0)
mpl.rcParams['font.size']=10                #10 

more_stopwords = {'org', 'default', 'aspx', 'stratfordrec','nhttp','Hi','also','now','much','username'}
STOPWORDS = STOPWORDS.union(more_stopwords)

lenders_reason = lenders[~pd.isnull(lenders['loan_because'])][['loan_because']]
lenders_reason_string = " ".join(lenders_reason.loan_because.values)

wordcloud = WordCloud(
                      stopwords=STOPWORDS,
                      background_color='white',
                      width=3200, 
                      height=2000,
                      mask=heart_mask
            ).generate(lenders_reason_string)

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('./reason_wordcloud.png', dpi=900)
plt.show()
df_mpi.head(7)
mpi_country = df_mpi.groupby('country')['MPI'].mean().reset_index()

data = [dict(
        type='choropleth',
        locations=mpi_country['country'],
        locationmode='country names',
        z=mpi_country['MPI'],
        colorscale='Greens',
        reversescale=True,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='MPI'),
    )]

layout = dict(
    title = 'Average MPI per country',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)

fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='mpi-map')
df_mpi_oxford = pd.read_csv('../input/mpi/MPI_subnational.csv')
temp = df_mpi_oxford.groupby('Country')['Headcount Ratio Regional'].mean().reset_index()

temp = temp[temp.Country.isin(list_countries)].sort_values(by="Headcount Ratio Regional", ascending = False)

plt.figure(figsize=(15,10))
sns.barplot(y=temp.Country, x=temp['Headcount Ratio Regional'], alpha=0.6)
plt.ylabel("Country", fontsize=16)
plt.xlabel('Headcount Ratio National', fontsize=16)
plt.title("Headcount Ratio National per Country", fontsize=16)
plt.show();
#load all needed data
df_household =pd.read_csv('../input/poverty-indicators/household_size.csv',sep=';')
df_indicators = pd.read_csv('../input/poverty-indicators/indicators.csv',
                            sep=';', encoding='latin1', decimal=',').rename(columns={'country_name': 'country'})
df_education= pd.read_csv('../input/additional-kiva-snapshot/country_stats.csv')[['country_name','mean_years_of_schooling']].rename(columns={'country_name': 'country'})
df_mobile = pd.read_csv('../input/poverty-indicators/mobile_ownership.csv',sep=';',encoding='latin1',decimal=',')
df_mobile['mobile_per_100capita']=df_mobile['mobile_per_100capita'].astype('float')

#merge data for most frequent countries
temp = pd.merge(df_indicators, df_household, how='right', on='country')
temp = pd.merge(temp, df_mobile, how='left', on='country')
indicators = pd.merge(temp, df_education, how='left', on='country').round(2)

indicators
palestine_data = ['Palestine','PLS', 4550000, 24.52, 91, 89, 60, 73, 7.4, 0.90, 5.9, 3.1, 97.8, 8]
kyrgyzstan_data = ['Kyrgyzstan','KGZ', 6082700, 64.15, 90, 93.3, 99.8, 58.3, 29.20, 0.73, 4.2, 2.1, 123.7, 10.80]

indicators.loc[35]=palestine_data
indicators.loc[36]=kyrgyzstan_data
indicators_mpi = pd.merge(indicators,mpi_country, how='inner', on='country').drop(['country_code','population'],axis=1)

corre = indicators_mpi.corr()

mask1 = np.zeros_like(corre, dtype=np.bool)
mask1[np.triu_indices_from(mask1)] = True

f, axs = plt.subplots(figsize=(10, 8))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corre, mask=mask1, cmap=cmap, vmax=.3, center=0, square=False, linewidths=.5, cbar_kws={"shrink": .5});
consumption = pd.read_csv('../input/poverty-indicators/consumption.csv',sep=";", decimal=',',encoding='latin1')

df = pd.merge(indicators, consumption[['country','consumption_capita']] , how='left', on='country').round(2).dropna()
df.rename(columns={'rural_population_%': 'rural_ratio','access_water_%':'ratio_wateraccess', 'access_electricity_%':'ratio_electricityaccess',
                  'employment_%':'ratio_employment','agriculture_employment_%':'ratio_agriculture',
                  'access_sanitation_%':'ratio_sanitation'},inplace=True)

from statsmodels.formula.api  import ols
model = ols(formula = 'consumption_capita ~ rural_ratio+ratio_wateraccess+ratio_electricityaccess+ratio_sanitation+ratio_employment+ratio_agriculture+\
            male_headship+average_household_size+avg_children_nb+mean_years_of_schooling+mobile_per_100capita',
          data = df).fit()
print(model.summary())
model2 = ols(formula = 'consumption_capita ~ rural_ratio+ratio_sanitation+mobile_per_100capita+average_household_size ',
          data = df).fit()
print(model2.summary())
df_ratio = round(df_kiva_loans.groupby('country')['ratio_amount_duration'].median(),2)
df_ctime = round(loans_dates.groupby(['country'])['time_funding'].median(),2)
df_camount = round(df_kiva_loans.groupby(['country'])['loan_amount'].median(),2)
df_repay = round(df_kiva_loans.groupby(['country'])['term_in_months'].median(),2)

kiva_indic = pd.concat([df_ratio, df_ctime, df_camount, df_repay], axis=1, join='inner').reset_index()
indicators_mpi_kiva = pd.merge(indicators_mpi,kiva_indic,how='left',on='country')

indicators_mpi_kiva = indicators_mpi_kiva[['MPI','ratio_amount_duration','time_funding','loan_amount','term_in_months']]
indicators_mpi_kiva.corr()
from sklearn.preprocessing import StandardScaler

clusters = pd.read_csv('../input/kivadhsv1/DHS.clusters.csv')

clusters= clusters.drop_duplicates()[['DHSCLUST','DHSCC.x', 'DHS.lat', 'DHS.lon','Country','MPI.median', 'Nb.HH', 'AssetInd.median','URBAN_RURA',
                    'Nb.Electricity', 'Nb.fuel', 'Nb.floor', 'Nb.imp.sanitation', 'Nb.imp.water', 'Median.educ', 'Nb.television', 'Nb.phone']]
clusters.drop(clusters.index[4987],inplace=True)

clusters['Country']=[clusters['Country'].iloc[i] if clusters['DHSCC.x'].iloc[i] !='AM' else 'ARM' for i in range(len(clusters)) ]
clusters[['DHS.lat', 'DHS.lon']] = clusters[['DHS.lat', 'DHS.lon']].astype(float,inplace=True)

for indic in ['Nb.Electricity','Nb.fuel','Nb.floor','Nb.imp.sanitation','Nb.imp.water','Nb.television','Nb.phone'] : 
    clusters[indic]=round(100*clusters[indic].astype(int)/clusters['Nb.HH'].astype(int),2)
    
clusters['URBAN_RURA']=clusters['URBAN_RURA'].apply (lambda x : 1 if x=='U' else 0 )

clusters['DHSCLUST']=[str(clusters['DHSCLUST'].iloc[i])+'_'+clusters['Country'].iloc[i] for i in range(len(clusters)) ]

clusters['AssetInd.median']=clusters['AssetInd.median'].astype(float)
max_asset = max(clusters['AssetInd.median'])
min_asset = min(clusters['AssetInd.median'])
clusters['AssetInd.median']=clusters['AssetInd.median'].apply(lambda x : round((100/(max_asset-min_asset)) * (x-min_asset),2))

clusters.rename(columns={"MPI.median":"MPI_cluster", "AssetInd.median" : "wealth_index", "URBAN_RURA": "urbanity" , 'Nb.Electricity':'ratio_electricity',
                        'Nb.imp.sanitation':'ratio_sanitation','Nb.imp.water':'ratio_water', 'Nb.phone':'ratio_phone',
                        'Nb.floor':'ratio_nakedSoil','Nb.fuel':'ratio_cookingFuel'}, inplace=True)

clusters['MPI_cluster']=clusters['MPI_cluster'].astype(float,inplace=True)

clusters[['urbanity' ,'ratio_sanitation','ratio_phone', 'ratio_electricity', 'ratio_cookingFuel', 'ratio_nakedSoil','ratio_water']] = StandardScaler().fit_transform(clusters[['urbanity' ,'ratio_sanitation','ratio_phone', 'ratio_electricity', 'ratio_cookingFuel', 'ratio_nakedSoil','ratio_water']])

clusters.sample(20)

clusters_clmb = clusters[clusters['Country']=='COL']

model_clmb= ols(formula = 'MPI_cluster ~ urbanity + ratio_sanitation +ratio_phone+ ratio_electricity+ ratio_cookingFuel+ratio_nakedSoil+ratio_water',
          data = clusters_clmb).fit()

print(model_clmb.summary())
loan_coords = pd.read_csv('../input/additional-kiva-snapshot/loan_coords.csv')
loans_extended = pd.read_csv('../input/additional-kiva-snapshot/loans.csv')
loans_with_coords = loans_extended.merge(loan_coords, how='left', on='loan_id')

loans_with_coords=loans_with_coords[['loan_id','country_code','country_name','town_name','latitude','longitude',
                                    'original_language','description','description_translated','tags', 'activity_name','sector_name','loan_use',
                                    'loan_amount','funded_amount',
                                    'posted_time','planned_expiration_time','disburse_time', 'raised_time', 'lender_term', 'num_lenders_total','repayment_interval']]
loans_with_coords = loans_with_coords[np.isfinite(loans_with_coords['latitude'])]

loans = loans_with_coords[loans_with_coords['country_name'].isin(['Philippines','Colombia','Armenia','Kenya','Haiti'])]

loans.sample(10)
#loans_with_coords['country_name'].value_counts() -> #Peru #Cambodia #Uganda #Pakistan #Ecuador

'''Performing Knn with k=1 to find the cluster for each loan'''
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1 , metric='haversine')
neigh.fit(np.radians(clusters[['DHS.lat', 'DHS.lon']]), clusters['DHSCLUST']) 
loans['DHSCLUST'] = neigh.predict(np.radians(loans[['latitude','longitude']]))

'''Build a table to show the coordinates of the loan and coordinates of the cluster it is assigned to, then calculate the Haversine distance in kilometers between cluster and loan'''
precision_knn = loans[['loan_id','country_name','latitude','longitude','DHSCLUST']].merge(clusters[['DHSCLUST','DHS.lat','DHS.lon']], how='left', on='DHSCLUST')
lat1 = np.radians(precision_knn['latitude'])
lat2 = np.radians(precision_knn['DHS.lat'])
lon1 = np.radians(precision_knn['longitude'])
lon2 = np.radians(precision_knn['DHS.lon'])
temp = np.power((np.sin((lat2-lat1)/2)),2) + np.cos(lat1) * np.cos(lat2) * np.power((np.sin((lon2-lon1)/2)),2)
precision_knn['distance_km'] = 6371 * (2 * np.arcsin(np.sqrt(temp))) #6371 is the radius of the earth

precision_knn.sample(10)
print("The median distance in kilometers between a loan and the cluster it's assigned to is : " , round(precision_knn['distance_km'].median(),2))
useful_clusters = list(loans['DHSCLUST'].value_counts().reset_index()['index'])
clusters_kiva = clusters[clusters['DHSCLUST'].isin(useful_clusters)]
temp = loans.dropna(subset=['posted_time','disburse_time', 'raised_time'], how='any', inplace=False)

dates = ['posted_time', 'disburse_time', 'raised_time']
temp[dates] = temp[dates].applymap(lambda x : x.split('+')[0])

temp[dates]=temp[dates].apply(pd.to_datetime)
temp['time_funding'] = temp['raised_time']- temp['posted_time']
temp['time_funding'] = temp['time_funding'] / timedelta(days=1) 
temp['ratio_amount_TimeFunding'] = temp['funded_amount']/temp['time_funding']
temp['ratio_amount_TimeRepay'] = temp['funded_amount']/temp['lender_term']

d = round(temp.groupby(['DHSCLUST'])[['funded_amount','lender_term','ratio_amount_TimeFunding','ratio_amount_TimeRepay']].median(),2).reset_index()

clusters_kiva = clusters_kiva.merge(d, how='left',on='DHSCLUST')

indicators_kiva_col =  clusters_kiva[['MPI_cluster','wealth_index','funded_amount','lender_term','ratio_amount_TimeFunding','ratio_amount_TimeRepay']][clusters_kiva['Country']=='COL']
indicators_kiva_col.corr()
d = loans.groupby(['DHSCLUST','sector_name'])['funded_amount'].count().reset_index()

d['proportion'] = [d['funded_amount'].iloc[i]/len(loans[loans['DHSCLUST']==d['DHSCLUST'].iloc[i]]) for i in range(len(d))]

tmp1 = d[d['sector_name']=='Agriculture'][['DHSCLUST','proportion']]
tmp1 = tmp1.merge(d[d['sector_name']=='Education'][['DHSCLUST','proportion']], how='left', on='DHSCLUST')
tmp1 = tmp1.merge(d[d['sector_name']=='Food'][['DHSCLUST','proportion']], how='left', on='DHSCLUST')
tmp1 = tmp1.merge(d[d['sector_name']=='Health'][['DHSCLUST','proportion']], how='left', on='DHSCLUST')
tmp1.columns=['DHSCLUST','ratio_agriculture', 'ratio_education','ratio_food', 'ratio_health']

clusters_kiva = clusters_kiva.merge(tmp1 , how='left',on='DHSCLUST')
indicators_kiva_col =  clusters_kiva[['MPI_cluster','wealth_index','ratio_agriculture', 'ratio_education','ratio_food', 'ratio_health']][clusters_kiva['Country']=='COL']
indicators_kiva_col.corr()
temp_list = []
for i in range(len(loans)) :
    try : 
        math.isnan(loans['description_translated'].iloc[i])
        temp_list.append(loans['description'].iloc[i])
    except :
        temp_list.append(loans['description_translated'].iloc[i])

loans['Description']=temp_list

df_temp = loans.sample(200000, random_state=42)
desc_by_clust = df_temp[['country_name','DHSCLUST','Description']].replace(np.nan, "").groupby(['country_name','DHSCLUST'])['Description'].apply(lambda x: "\n".join(x)).reset_index() 

plt.style.use('ggplot')
import spacy
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

nlp = spacy.load('en_core_web_lg', disable=["tagger", "parser", "ner"])
nlp.max_length = 5000000
raw_desc_texts = list(desc_by_clust['Description'].values)
processed_desc_texts = [nlp(text) for text in raw_desc_texts]
processed_desc_vectors = np.array([text.vector for text in processed_desc_texts])

tsne = TSNE(n_components=2, metric='cosine', random_state=7777)
fitted = tsne.fit(processed_desc_vectors)
fitted_components = fitted.embedding_
desc_by_clust['cx'] = fitted_components[:, 0]
desc_by_clust['cy'] = fitted_components[:, 1]
desc_by_clust.head()

cluster_cnt = desc_by_clust.groupby('country_name').size()
selected_countries = cluster_cnt[cluster_cnt > 50]
n_selected_countries = len(selected_countries)
selected_country_pos = np.where(cluster_cnt > 50)[0]
id2country = dict(enumerate(selected_countries.index))
country2id = {v: k for k, v in id2country.items()}
selected_desc_by_cluster = desc_by_clust.query('country_name in @selected_countries.index')

fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(selected_desc_by_cluster['cx'], selected_desc_by_cluster['cy'], s=15,
            c=[country2id[x] for x in selected_desc_by_cluster['country_name']],
            cmap=plt.cm.get_cmap('tab20', 5))
formatter = plt.FuncFormatter(lambda val, loc: id2country[val])
plt.colorbar(ticks=np.arange(19), format=formatter);
plt.show()
philippines_cluster_desc = desc_by_clust.query('country_name == "Philippines"')
clusters_phil = KMeans(n_clusters=5, random_state=42)
clusters_phil.fit_transform(philippines_cluster_desc[['cx', 'cy']]);

for c_desc in philippines_cluster_desc['Description'].iloc[clusters_phil.labels_ == 3].iloc[:10]:
    print(c_desc[:min(800, len(c_desc))], end="")
    print('...' if len(c_desc) > 800 else "")
    print('-' * 20)
for c_desc in philippines_cluster_desc['Description'].iloc[clusters_phil.labels_ == 1].iloc[:10]:
    print(c_desc[:min(800, len(c_desc))], end="")
    print('...' if len(c_desc) > 800 else "")
    print('-' * 20)
use_by_clust = loans[['country_name','DHSCLUST','loan_use']].replace(np.nan, "").groupby(['country_name','DHSCLUST'])['loan_use'].apply(lambda x: "\n".join(x)).reset_index() 

raw_use_texts = list(use_by_clust['loan_use'].values)
processed_use_texts = [nlp(text) for text in raw_use_texts]

processed_use_vectors = np.array([text.vector for text in processed_use_texts])

tsne = TSNE(n_components=2, metric='cosine', random_state=7777)
fitted = tsne.fit(processed_use_vectors)
fitted_components = fitted.embedding_
use_by_clust['cx'] = fitted_components[:, 0]
use_by_clust['cy'] = fitted_components[:, 1]

cluster_region_cnt = use_by_clust.groupby('country_name').size()
selected_countries = cluster_region_cnt[cluster_region_cnt > 50]
n_selected_countries = len(selected_countries)
selected_country_pos = np.where(cluster_region_cnt > 50)[0]
id2country = dict(enumerate(selected_countries.index))
country2id = {v: k for k, v in id2country.items()}
selected_use_by_cluster = use_by_clust.query('country_name in @selected_countries.index')

fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(selected_use_by_cluster['cx'], selected_use_by_cluster['cy'], s=15,
            c=[country2id[x] for x in selected_use_by_cluster['country_name']],
            cmap=plt.cm.get_cmap('tab20', 5))
formatter = plt.FuncFormatter(lambda val, loc: id2country[val])
plt.colorbar(ticks=np.arange(19), format=formatter);
plt.show()
philippines_cluster_uses = use_by_clust.query('country_name == "Philippines"')
clusters_phil = KMeans(n_clusters=3, random_state=7777)
clusters_phil.fit_transform(philippines_cluster_uses[['cx', 'cy']]);

for c_uses in philippines_cluster_uses['loan_use'].iloc[clusters_phil.labels_ == 0].iloc[:10]:
    print(c_uses[:min(500, len(c_uses))], end="")
    print('...' if len(c_uses) > 500 else "")
    print('-' * 20)
for c_uses in philippines_cluster_uses['loan_use'].iloc[clusters_phil.labels_ == 1].iloc[:10]:
    print(c_uses[:min(500, len(c_uses))], end="")
    print('...' if len(c_uses) > 500 else "")
    print('-' * 20)
'''First step : load DHS Program clusters data, and keep demographic features after data cleaning. 
That's what we did above to get the dataset "clusters" '''

"""Second step : Running regression model, let's do that for Philippines """
clusters_ph = clusters[clusters['Country']=='PH']
model_ph= ols(formula = 'MPI_cluster ~ urbanity + ratio_sanitation +ratio_phone+ ratio_electricity+ ratio_cookingFuel+ratio_nakedSoil+ratio_water',
          data = clusters_ph).fit()

"""Third step : Given a random new loan from philippines, let's assign it to nearest cluster"""
new_loan = loans[loans['country_name']=='Philippines'].sample(1,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1 , metric='haversine')
neigh.fit(np.radians(clusters_ph[['DHS.lat', 'DHS.lon']]), clusters_ph['DHSCLUST']) 
cluster_newloan = neigh.predict(np.radians(new_loan[['latitude','longitude']]))[0]
mpi_newloan = clusters_ph[clusters_ph.DHSCLUST=='481_PH']['MPI_cluster']   ## measure of MPI

"""Fourth step : See if the sector """
sector_loan = new_loan['sector_name']
indicators_kiva_phil =  clusters_kiva[['MPI_cluster','wealth_index','ratio_agriculture', 'ratio_education','ratio_food', 'ratio_health']][clusters_kiva['Country']=='PH']
indicators_kiva_phil.corr()
# seeing the correlation matrix, agriculture seems to be the sector that reflects poverty the most in Philippines, thus : 
# Agricultre -> 4, food->3 , education ->2, health ->1
# here sector_loan is Agriculture so given two loans coming from the same cluster '481_PH', this loan here should have priority.

"""Fifth step : Kiva score"""
clusters_kiva_phil =clusters_kiva[clusters_kiva['Country']=='PH']
indicators_kiva_phil =  clusters_kiva_phil[['MPI_cluster','wealth_index','funded_amount','lender_term','ratio_amount_TimeFunding','ratio_amount_TimeRepay']]
cor = indicators_kiva_phil.corr()['MPI_cluster']
clusters_kiva_phil[['MPI_cluster','wealth_index','funded_amount','lender_term','ratio_amount_TimeFunding','ratio_amount_TimeRepay']] = StandardScaler().fit_transform(indicators_kiva_phil)
loan_infos = clusters_kiva_phil[clusters_kiva_phil['DHSCLUST']==cluster_newloan]
kiva_score = float(cor['funded_amount']*loan_infos['funded_amount'] + cor['lender_term']*loan_infos['lender_term'] +cor['ratio_amount_TimeFunding']*loan_infos['ratio_amount_TimeFunding']+cor['ratio_amount_TimeRepay']*loan_infos['ratio_amount_TimeRepay'])
