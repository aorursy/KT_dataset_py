# data manipulation packages
import numpy as np
import pandas as pd
import re

# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns
# Import Yelp and American Community Survey (ACS) datasets
yelp = pd.read_csv('../input/yelp-dataset/yelp_business.csv')
acs = pd.read_csv('../input/ipumsancestryextract/2012-16_ipums_ancestry_extract.csv')
#Import Yelp to ACS mapping
geo_map_yelp = pd.read_csv('../input/ipumsancestryextract/geo_map_yelp.csv')
geo_map_yelp
# Merge Yelp dataset with Yelp to ACS Mapping, dropping entries not in our 6 cities
yelp_geo = pd.merge(yelp, geo_map_yelp, on='city', how='inner', sort=False)
yelp_geo['state'].value_counts()
# Drop out duplicate cities in other states
yelp_geo = yelp_geo.loc[(yelp_geo['state']=='AZ') |
                        (yelp_geo['state']=='NV') |
                        (yelp_geo['state']=='NC') |
                        (yelp_geo['state']=='OH') |
                        (yelp_geo['state']=='PA') |
                        (yelp_geo['state']=='IL')
                        ].reset_index(drop=True)

yelp_geo['state'].value_counts()
# prepare ethnicity in yelp data (by restaurant category)
# mapping source: https://www.yelp.com/developers/documentation/v3/category_list
eth_map_yelp = pd.read_csv('../input/ipumsancestryextract/eth_map_yelp.csv', index_col='Yelp_clean')

eth_map_yelp.head(10)
# create new dataframe for Yelp data with labelled ethnic data
yelp_geo_eth = yelp_geo.copy()
yelp_geo_eth['ethnicity']=""
    
# convert the mapping to a dictionary to label the dataset
eth_dict_yelp = eth_map_yelp.to_dict()['ethnicity']

# Label all Yelp businesses (including restaurants) by ethnicity.  
# Note: this is very slow, using two for loops, which can take up to 1 minute. This can be substantially improved.
for k, v in eth_dict_yelp.items():
    for index, element in yelp_geo_eth.loc[:,'categories'].iteritems():
        if k in element:
            yelp_geo_eth.loc[yelp_geo_eth.index[index],'ethnicity']=v
yelp_grouped=yelp_geo_eth.groupby(['MET2013','ethnicity']).agg({'stars':['mean','median'], 'business_id':'count', 'review_count':'sum', 'Metropolitan area, 2013 OMB delineations':'first'})
yelp_grouped.columns=['mean_stars','median_stars','restaurant_count','review_count','area_name']
yelp_grouped.head(10)
# Import ACS ethnicity mapping
eth_map_acs = pd.read_csv('../input/ipumsancestryextract/eth_map_acs.csv')
eth_map_acs.iloc[[0,1,2,3,-4,-3,-2,-1]]
# Merge ethnicity mapping with the ACS dataset, and drop irrelevant populations (e.g. American)
acs_eth = pd.merge(acs, eth_map_acs, on='ANCESTR1', how='inner', sort=False)
acs_eth = acs_eth[(acs_eth['ethnicity']!='american') & (acs_eth['ethnicity']!='na')]
acs_eth.head(5)
acs_grouped = acs_eth.groupby(['MET2013','ethnicity']).agg({'PERWT':'sum'})
acs_grouped.head(15)
comb_data=pd.merge(acs_grouped,yelp_grouped,left_index=True,right_index=True,how='inner').reset_index()
comb_data
# plot restaurant count by number of people (by ethnicity)
plt.figure(figsize=(20,10))

sns.regplot('PERWT','restaurant_count',data=comb_data)
plt.title("Ethnic restaurant count by population", size=22)
plt.xlabel("Ethnic population")
plt.ylabel("Ethnic restaurant count")
plt.show()
# Create log of variables
comb_data['log_pop']=comb_data['PERWT'].apply(np.log1p)
comb_data['log_rest_cnt']=comb_data['restaurant_count'].apply(np.log1p)

#Plot figure with log of variables
plt.figure(figsize=(20,10))
sns.regplot('log_pop','log_rest_cnt', data=comb_data)
plt.title("Ethnic restaurant count by population", size=22)
plt.xlabel("Log of ethnic population")
plt.ylabel("Log of ethnic restaurant count")
plt.show()
# plot restaurant rating by population
sns.lmplot('PERWT','mean_stars', data=comb_data, 
            fit_reg=True, size=6, aspect=3, legend=False)
plt.title("Ethnic restaurant rating by population", size=22)
plt.xlabel("Ethnic population")
plt.ylabel("Mean ethnic restaurant stars")
plt.show()
# plot restaurant rating by log population
sns.lmplot('log_pop','mean_stars', data=comb_data, 
            fit_reg=True, size=6, aspect=3, legend=False)
plt.title("Ethnic restaurant rating by population", size=22)
plt.xlabel("Log ethnic population")
plt.ylabel("Mean ethnic restaurant stars")
plt.show()
# exclude Brazil and Pacific ethnicities
comb_data_trimmed = comb_data.loc[(comb_data['ethnicity']!='brazil')&(comb_data['ethnicity']!='pacific')]

# Plot facet chart for each ethnicity
ax = sns.lmplot('log_pop','mean_stars', data=comb_data_trimmed, 
               col='ethnicity', hue='ethnicity', col_wrap=3, scatter_kws={'s':100},
               fit_reg=True, size=6, aspect=2, legend=False)
ax.set(ylim=(2.5,5))
ax.set_titles(size=24)

plt.show()

