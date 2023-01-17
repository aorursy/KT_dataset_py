import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# honeyCols = ['state','n_col', 'yield_per_col', 'total_prod', 'stocks', 'price_per_lb', 'prod_value', 'year']
honey = pd.read_csv("/kaggle/input/honey-production/honeyproduction.csv")
honeyCols = {"state":"state_code", "numcol":"n_colony", "yieldpercol":"yield_per_colony", "totalprod" : "total_production", "stocks":"stocks_in", "priceperlb":"price_per_lb", "prodvalue":"production_value"}
honey.rename(columns = honeyCols, inplace = True)
honey.head()
honey.describe(include = "all")
state_name_dict = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming'
}

honey['state_name'] = honey['state_code'].apply(lambda x: state_name_dict[x])
honey['consumption'] = honey['total_production'] - honey['stocks_in'] 

honey.head()
#Reordering the columns
honey = honey[["state_code", "state_name", "n_colony", "yield_per_colony", "total_production", "stocks_in", "consumption", "price_per_lb", "production_value","year"]]
honey.head()
#Adjusting for inflation
inflation_rate = {
    1998: 1.454,
    1999: 1.423,
    2000: 1.376,
    2001: 1.339,
    2002: 1.317,
    2003: 1.288,
    2004: 1.255,
    2005: 1.214,
    2006: 1.176,
    2007: 1.143,
    2008: 1.101,
    2009: 1.105,
    2010: 1.087,
    2011: 1.054,
    2012: 1.032
}

inflation_list = ['price_per_lb', 'production_value']

for year in set(honey['year']):
    for item in inflation_list:
        honey.loc[honey['year']==year, item] = inflation_rate[year]*honey.loc[honey['year']==year, item]
        
honey.head()
#General Production trend from 1998 to 2012
plt.figure(figsize = (12,8))
sns.pointplot(honey["year"], honey["total_production"])
plt.show()
#Check the production trend using boxplots
plt.figure(figsize=(18,8))
sns.boxplot("year", "total_production", data = honey)
plt.show()
#Production trend for each state
sns.FacetGrid(honey, col = "state_name", col_wrap = 5, height = 3).map(plt.plot, "year", "total_production", marker = "*")
#Checking the correlation between variables
honey_corr = honey[['n_colony', 
              'yield_per_colony', 
              'total_production', 
              'stocks_in', 
              'price_per_lb', 
              'production_value']].corr()

honey_corr
#Visualizing the same in Heatmap()
sns.heatmap(honey_corr, annot = True, vmin = -1, vmax = 1, cmap="YlGnBu")
plt.show()
#Visualising the relationship between variables with pairplot
sns.pairplot(honey[['n_colony', 
              'yield_per_colony', 
              'total_production', 
              'stocks_in', 
              'price_per_lb', 
              'production_value']])
#Plotting the relative trend graphs for Number of Colonies, Yield per colony, Total Production etc. 
honey_prod_over_years = honey.groupby("year").mean()
honey_prod_over_years[['n_colony','yield_per_colony','total_production','stocks_in','price_per_lb','consumption','production_value']].plot(ax=plt.subplots(figsize=(15,9))[1])
plt.show()
#Grouping the dataset based on years
honey[["year", "yield_per_colony"]].groupby("year").mean().round().plot()
plt.show()
#Grouping the dataset based on states
plt.figure(figsize=(18,8))
honey[["state_name", "total_production"]].groupby("state_name").mean().round().plot()
plt.show()
#Checking hypothesis: "More the number of colonies, higher the production value"
sns.lmplot(x = "n_colony", y = "production_value", data = honey, hue = 'year')
plt.show()
# Interactive scatter plot using plotly package
import plotly.express as px
px.scatter(honey, x = 'n_colony', y = 'production_value', animation_frame = 'year', size = 'total_production')
