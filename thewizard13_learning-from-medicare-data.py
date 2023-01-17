#Import Base Packages

import pandas as pd

import numpy as np

import os



#Import Visualization Packages

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot, download_plotlyjs

init_notebook_mode(connected=True) #important call otherwise plotly graphs will not show up in notebook 



#Import BigQuery API for Accessing Medicare Data

from google.cloud import bigquery



#print(os.listdir("../input"))
#Load Medicare Part D Prescriber (2014) Dataset

#Aggregated by total beneficiaries, claims, & claim cost per state



client = bigquery.Client() #need to call this to prime the bigquery cursor



query = (

    '''

    select

        nppes_provider_state as state,

        sum(bene_count) as beneficiaries,

        sum(total_claim_count) as total_claims,

        sum(total_drug_cost) as total_claim_cost

    from `bigquery-public-data.cms_medicare.part_d_prescriber_2014`

    group by 1

    order by 2 desc

    '''

    #"limit 100"

)

query_job = client.query(query)



medicare = query_job.to_dataframe()



medicare.head()
#Beneficiaries by State Bar Chart



data = [dict(

    type = 'bar',

    x = medicare.state,

    y = (medicare.beneficiaries/1000000).round(1),

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

    text= medicare.state

)

       ]

layout = dict(title = '2014 Medicare Beneficiaries by State (Millions)',

             )



fig = dict(data = data, layout = layout)

iplot(fig)
#Beneficiaries by State Choropleth Map



data = [dict(

    type = 'choropleth',

    locations = medicare['state'], #spatial coordinates

    z = (medicare['beneficiaries']/1000000).round(1).astype(float), #data to be color-coded

    locationmode = 'USA-states', #set of locations match entries in locations

    colorscale = 'Reds',

    text = medicare['state'],

    colorbar = dict(

            title = "Millions")

)

]



layout = go.Layout(

    title_text = '2014 Medicare Beneficiaries by State',

    geo_scope='usa', #limit map scope to USA

)



fig = dict(data = data, layout = layout)

iplot(fig)



#fig = go.Figure(data = data, layout = layout)

#fig.show() #two ways to show figure (dictionary notation or go.Figure)
#Medicare Beneficiaries Per Capita



#Import 2015 US Pop Estimates by State

pop2015 = pd.read_csv("../input/population.csv")

pop2015.columns = ['state','population']



#Join State medicare info Drugs and Populations

med_pop = pd.merge(medicare, pop2015, how = 'left', left_on = 'state', right_on = 'state')

med_pop = med_pop.dropna(subset = ['population']) #drop rows where population is NaN

med_pop['benef_per_cap'] = med_pop['beneficiaries']/med_pop['population']

med_pop['claims_per_cap'] = med_pop['total_claims']/med_pop['population']

med_pop['cost_per_cap'] = med_pop['total_claim_cost']/med_pop['population']



med_pop.head()
#Medicare Beneficiaries Per Capita Choropleth Map



data = [dict(

    type = 'choropleth',

    locations = med_pop['state'], #spatial coordinates

    z = med_pop['benef_per_cap'].astype(float), #data to be color-coded

    locationmode = 'USA-states', #set of locations match entries in locations

    colorscale = 'Blues',

    text = medicare['state'],

    reversescale = True,

)

]



layout = go.Layout(

    title_text = '2014 Medicare Beneficiaries per Capita by State',

    geo_scope='usa', #limit map scope to USA

)



fig = dict(data = data, layout = layout)

iplot(fig)



#fig = go.Figure(data = data, layout = layout)

#fig.show()
#Most Prescribed Drugs in US

#Aggregated by Total Claims and Cost



query = (

    '''

    select

        generic_name,

        sum(total_claim_count) as total_claims,

        sum(total_drug_cost) as total_claim_cost

    from `bigquery-public-data.cms_medicare.part_d_prescriber_2014`

    group by 1

    order by 2 desc

    '''

    #"limit 100"

)

query_job = client.query(query)



drugs = query_job.to_dataframe() #don't forget this call... very useful!

drugs.head(10)

#Most Prescribed Drugs per Capita by State

#Aggregated by Total Claims and Cost



query = (

    '''

    select

        nppes_provider_state as state,

        generic_name,

        sum(total_claim_count) as total_claims,

        sum(total_drug_cost) as total_claim_cost

    from `bigquery-public-data.cms_medicare.part_d_prescriber_2014`

    group by 1,2

    order by 3 desc

    '''

    #"limit 100"

)

query_job = client.query(query)



state_drugs = query_job.to_dataframe() #don't forget this call... very useful!



#Join Prescribed Drugs and Populations

drug_pop = pd.merge(state_drugs, pop2015, how = 'left', left_on = 'state', right_on = 'state')

drug_pop = drug_pop.dropna(subset = ['population']) #drop rows where population is NaN

drug_pop['claims_per_cap'] = drug_pop['total_claims']/drug_pop['population']

drug_pop['cost_per_cap'] = drug_pop['total_claim_cost']/drug_pop['population']



#Get top drugs for each state

top_drugs = drug_pop.groupby(['state'])

top_drugs = top_drugs.apply(lambda x: x.sort_values(by = ['total_claims'], ascending = False))

top_drugs = top_drugs.reset_index(drop = True)

top5_drugs = top_drugs.groupby(['state']).head(5)
#Graph Top Drugs by Number of States the Drug is within the Top 5

buckets = top5_drugs['generic_name'].nunique()

drug_counts = top5_drugs['generic_name'].count()



data = [dict(

        type = 'histogram',

        x = top5_drugs['generic_name'],

            )

       ]



layout = go.Layout(

        title_text = 'Counts of Top 5 Drugs in Each State',

)



fig = dict(data = data, layout = layout)

iplot(fig)
#Import Nursing Facility Data



query = (

    '''

    select

        *

    from `bigquery-public-data.cms_medicare.nursing_facilities_2014`

    '''

    "limit 100"

)

query_job = client.query(query)



nursing = query_job.to_dataframe() #don't forget this call... very useful!

nursing.head()
#Facility Clustering

from sklearn.cluster import KMeans



nursing = nursing.drop(columns = ['facility_name', 'street_address', 'city', 'state', 'zip_code', 'male_beneficiaries', 'female_beneficiaries', 'dual_beneficiaries', 'nondual_beneficiaries', 'percent_of_beneficiaries_with_hypertension'])

nursing.index = nursing['provider_id']

nursing = nursing.dropna()



cluster = KMeans(

            n_clusters = 4,

            init = 'k-means++',

            n_init = 10,

            max_iter = 300,

                )



nursing['cluster'] = cluster.fit_predict(nursing)



from sklearn.decomposition import PCA #Principal Component Analysis so I can create scatter plot

pca = PCA(n_components=2).fit(nursing)

pca_2d = pca.transform(nursing)

kmeans = cluster.fit(nursing)

y_kmeans = cluster.predict(nursing)

plt.scatter(pca_2d[:,0], pca_2d[:,1], c=y_kmeans, s=50, cmap='viridis')

plt.show()
centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.show()
#Regression Analysis

#Predictive? Correlation?

from sklearn.linear_model import LinearRegression