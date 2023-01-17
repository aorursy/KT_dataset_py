## For Data

import numpy as np

import pandas as pd

import datetime

import re



## For Plotting

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go



#Display Options

pd.set_option('display.max_colwidth', None)

pd.options.display.max_rows = 1000



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dfdnm = pd.read_excel("/kaggle/input/silk-road-2-listings/mastersr2.xlsx")

print("This dataset contains",dfdnm.shape[0],"observations across",dfdnm.shape[1],"variables")

dfdnm.head()
# Plot missingness

sns.heatmap(dfdnm.isnull(),

           yticklabels=False,

           cbar=False,

           cmap='viridis')
# Check missingness

dfdnm.isnull().sum()
# Create dictionaries to map alternative values to country names

region_dict = {

    

    'Unknown': 'Unknown',

    'China': 'Asia',

    'Hong Kong, (China)': 'Asia',

    'Undeclared': 'Unknown',

    'India': 'Asia',

    'Canada': 'North America',

    'United States': 'North America',

    'Netherlands': 'EU',

    'United Kingdom': 'EU',

    'Germany': 'EU',

    'Belgium': 'EU',

    'South Africa': 'Africa',

    'Australia': 'Oceania',

    'Spain': 'EU',

    'Czech Republic': 'EU',

    'Sweden': 'EU',

    'Finland': 'EU',

    'New Zealand': 'Oceania',

    'Norway': 'Non-EU Europe',

    'Poland': 'EU',

    'Austria': 'EU',

    'Switzerland': 'Non-EU Europe',

    'Denmark': 'EU',

    'Ireland': 'EU',

    'Italy': 'EU',

    'Bulgaria': 'EU',

    'Slovenia': 'EU',

    'Armenia': 'Non-EU Europe',

    'Slovakia': 'EU',

    'Latvia': 'EU',

    'France': 'EU',

    'Hungary': 'EU',

    'Singapore': 'Asia',

    'Germany\n]': 'EU',

    'Colombia': 'Latin America',

    'Malaysia': 'Asia',

    'Israel': 'Middle East',

    'Japan': 'Asia',

    'Vatican (Holy See)': 'EU',

    'Angola': 'Africa',

    'Greece': 'EU',

    'Paraguay': 'Latin America',

    'Albania': 'Non-EU Europe',

    'Panama': 'Latin America',

    'Luxembourg': 'EU',

    'Kosovo': 'Non-EU Europe',

    'Mexico': 'North America',

    'Monaco': 'EU',

    'Argentina': 'Latin America',

    'Bolivia': 'Latin America',

    'Ukraine': 'Non-EU Europe',

    'Croatia': 'EU',

    'Denmark': 'EU',

    'Lithuania': 'EU',

    'Romania': 'EU',

    'Reunion': 'Africa',

    'Saint Martin': 'Latin America',

    'Moldova': 'Non-EU Europe',

    'Central America': 'Latin America',

    'Tuvalu': 'Oceania',

    'Thailand': 'Asia',

    'Afghanistan': 'Asia',

    'European union': 'EU',

    'Belgium': 'EU',

    'Spain': 'EU', 

    'Hungary': 'EU',

    'Philippines': 'Asia',

    'EU':'EU'

     

}



country_dict = {

    'China': 'China',

    'Hong Kong, (China)': 'China',

    'Undeclared': 'Unknown',

    'India': 'India',

    'Canada': 'Canada',

    'United States': 'United States',

    'Netherlands': 'Netherlands',

    'United Kingdom': 'United Kingdom',

    'Germany': 'Germany',

    'Belgium': 'Belgium',

    'South Africa': 'South Africa',

    'Australia': 'Australia',

    'Spain': 'Spain',

    'Czech Republic': 'Czech Republic',

    'Sweden': 'Sweden',

    'Finland': 'Finland',

    'New Zealand': 'New Zealand',

    'Norway': 'Norway',

    'Poland': 'Poland',

    'Austria': 'Austria',

    'Switzerland': 'Switzerland',

    'Denmark': 'Denmark',

    'Ireland': 'Ireland',

    'Italy': 'Italy',

    'Bulgaria': 'Bulgaria',

    'Slovenia': 'Slovenia',

    'Armenia': 'Armenia',

    'Slovakia': 'Slovakia',

    'Latvia': 'Latvia',

    'France': 'France',

    'Hungary': 'Hungary',

    'Singapore': 'Singapore',

    'Germany\n]': 'Germany',

    'Colombia': 'Colombia',

    'Malaysia': 'Malaysia',

    'Israel': 'Israel',

    'Japan': 'Japan',

    'Vatican (Holy See)': 'Holy See',

    'Angola': 'Angola',

    'Greece': 'Greece',

    'Paraguay': 'Paraguay',

    'Albania': 'Albania',

    'Panama': 'Panama',

    'Luxembourg': 'Luxembourg',

    'Kosovo': 'Kosovo',

    'Mexico': 'Mexico',

    'Monaco': 'Monaco',

    'Argentina': 'Argentina',

    'Bolivia': 'Bolivia',

    'Ukraine': 'Ukraine',

    'Croatia': 'Croatia',

    'Denmark / UK (Top #4 Seller)': 'United Kingdom',

    'Lithuania': 'Lithuania',

    'Romania': 'Romania',

    'Reunion (FR)': 'Reunion',

    'Saint Martin (FR)': 'Saint Martin',

    'Moldova, Republic of': 'Moldova',

    'Central America': 'Central America',

    'Tuvalu': 'Tuvalu',

    'Thailand': 'Thailand',

    'Afghanistan': 'Afghanistan',

    'European union': 'EU',

    'Belgium': 'Belgium',

    'Spain': 'Spain', 

    'Hungary': 'Hungary',

    'Philippines': 'Philippines'

}



destination_dict = {

    'Worldwide': 'Worldwide',

    'Canada': 'Canada',

    'United States': 'United States',

    'Worldwide except Australia': 'Worldwide ex AUS',

    'European Union': 'EU',

    'Undeclared': 'Undeclared',

    'Australia': 'Australia',

    'Sweden': 'Sweden',

    'Germany': 'Germany',

    'United Kingdom': 'United Kingdom',

    'United States &amp; Canada': 'United States and Canada', 

    'New Zealand': 'New Zealand',

    'Norway': 'Norway',

    'Finland': 'Finland',

    'Ireland': 'Ireland',

    'China': 'China',

    'Switzerland': 'Switzerland',

    'Denmark': 'Denmark',

    'Azerbaijan': 'Azerbaijan',

    'Netherlands': 'Netherlands',

    'Italy': 'Italy',

    'France': 'France',

    'European Union / UK / Worldwide': 'Worldwide',

    'Worldwide\n ]': 'Worldwide',

    'United States & Canada': 'United States and Canada',

    'Argentina': 'Argentina',

    'Mexico': 'Mexico',

    'Japan': 'Japan',

    'Tuvalu': 'Tuvalu',

    'Belgium': 'Belgium',

    'Spain': 'Spain',

    'Hungary': 'Hungary',

    'Philippines': 'Philippines'

}



# Map keys to values

dfdnm['Origin'] = dfdnm['Origin'].map(country_dict)

dfdnm['Origin_region'] = dfdnm['Origin'].map(region_dict)

dfdnm['Destination'] = dfdnm['Destination'].map(destination_dict)
# Subset obs to titles containing 'Cocaine' within Cocaine subcategory

key = re.compile('COCAINE')

filt = (dfdnm['Title']

         .str.upper()

         .str.contains(key))



dfcoca = (dfdnm[(filt.fillna(False)) & 

                dfdnm.Category.isin(['Drugs','Stimulants']) & 

                dfdnm.Subcategory.isin(['Cocaine','None'])].copy())



# Remove listings with price of zero

dfcoca= dfcoca[dfcoca.PriceUSD > 0]



# Extract product purity

dfcoca['Purity'] =  (dfcoca['Title']

                     .str.upper()

                     .str.replace('\d+% DISCOUNT', '')

                     .str.replace('\d+% REFUND', '')

                     .str.extract(r'(\d+(?:\.\d+)?%)'))

dfcoca['Purity'] =   (dfcoca['Purity']

                     .str.extract(r'(\d+(?:\.\d+)?)')

                     .astype(float))



dfcoca['Title1'] = (dfcoca['Title']

                    .str.strip()

                    .str.upper()

                    .str.replace('(', '')  

                    .str.replace(')', '') 

                    .str.replace('S', '%') 

                    .str.replace(r'(\d+(?:\-\d+)?%)', '')

                    .str.pad(50, side='left', fillchar=' ')

                    .str.replace('%', 'S'))



# Extract quantity of units

dfcoca['Quantitytext'] = (dfcoca['Title1']

                      .str.replace('ONE', '1')

                      .str.replace(',', '.')

                      .str.replace('1/8', '0.125')

                      .str.replace('1/4', '0.25')

                      .str.replace('1/2', '0.5')

                      .str.replace('HALF', '0.5')

                      .str.replace('3 OZ', '84G')

                      .str.replace('2 OZ', '56G')

                      .str.replace(r'8.?B', '3.5G')

                      .str.replace(r' \.', '0.')

                      .str.extract(r'(\.?\d+(?:\.\d+)?.*?O?K?.K?G?)')

                      .fillna('NA'))



# dfcoca['Quantity'] = (dfcoca.Quantitytext.str.extract('(\d+(?:\.\d+)?)')

#                       .astype(float))

dfcoca['Quantity'] = (dfcoca.Quantitytext

                      .str.pad(10, side='left', fillchar=' ')

                      .str.replace(r' \.', '0.')

                      .str.extract('(\d+(?:[.,-]\d+)?)')

                      .astype(float))





dfcoca['Quantitytext1'] = dfcoca.Quantitytext.str.replace('(\d+(?:[.,-]\d+)?)','')

dfcoca= dfcoca[~dfcoca.Title1.str.contains("CUTTING") & ~dfcoca.Title1.str.contains("RITALIN") & ~dfcoca.Title1.str.contains("LEAVES")]

dfcoca.loc[(dfcoca.Quantitytext1.str.contains('G')),'Grams']=1

dfcoca.loc[(dfcoca.Quantitytext1.str.contains('K')),'Grams']=1000

dfcoca.loc[(dfcoca.Quantitytext1.str.contains('KG')),'Grams']=1000

dfcoca.loc[(dfcoca.Quantitytext1.str.contains('.K')),'Grams']=1000

dfcoca.loc[(dfcoca.Title1.str.contains('GR')),'Grams']=1

dfcoca.loc[(dfcoca.Quantitytext1.str.contains('OZ')),'Grams']=28

dfcoca.loc[(dfcoca.Quantitytext1.str.contains('OUNCE')),'Grams']=28

dfcoca.loc[(dfcoca.Title1.str.contains('MG')),'Grams']=0.001 #Milligram

dfcoca['Quantity'] = dfcoca['Grams'] * dfcoca['Quantity']



dfcoca['Title1'] = dfcoca['Title1'].str.replace('%', 'S')  



# Calculate price per unit

dfcoca['Gram_Price'] = (dfcoca['PriceUSD']/dfcoca['Quantity']).astype(float)





# Drop variables

dfcoca.drop(['Quantitytext','Quantitytext1', 'Rating', 'Reviews', 'Market', 'Category', 'Subcategory','Grams'], axis=1, inplace = True)
# Proportion of observations with missing data by column

dfcoca.isnull().sum() * 100 / len(dfcoca)
# Visually check for missing values

sns.heatmap(dfcoca.isnull(),

           yticklabels=False,

           cbar=False,

           cmap='viridis')
# Drop all observations that don't contain information about our variable of interest

dfcoca = dfcoca.dropna(subset=['Gram_Price'])



# Retain only most recent listing scrape values

dfcoca = dfcoca.drop_duplicates(subset=['Title'], keep='last')
# Histogram of PPG

sns.distplot(dfcoca.Gram_Price, hist=True)
dfcoca[dfcoca.Gram_Price > 500][['Title','Sellerid','Quantity','Purity','PriceUSD','Gram_Price']].sort_values(by=['Gram_Price'], ascending = False)
dfcoca[dfcoca.Gram_Price < 25][['Title','Sellerid','Quantity','Purity','PriceUSD','Gram_Price','Date']].sort_values(by=['Gram_Price'], ascending = True)
# Remove Sample and Lottery listings

dfcoca= dfcoca[~dfcoca.Title1.str.contains("SAMPLE") & ~dfcoca.Title1.str.contains("LOTTERY")]

dfcoca[dfcoca.Gram_Price < 30][['Title','Sellerid','Quantity','Purity','PriceUSD','Gram_Price']].sort_values(by=['Gram_Price'], ascending = True)



# Excluding abnormally high values

dfcoca1 = dfcoca.sort_values(by=['Gram_Price'], ascending = False)[3:]



# Histogram of PPG

sns.distplot(dfcoca1.Gram_Price, hist=True)



# Descriptive Statistics

dfcoca1.Gram_Price.describe()
def donut_plot(data, vari, title1) :

    """Displays distribution of churn status for given variable"""

    trace1 = go.Pie(values  = data[vari].value_counts().values.tolist(),

                    labels  = data[vari].value_counts().keys().tolist(),

                    hoverinfo = "label+percent",

                    domain  = dict(x = [0,1]),

                    name    = "Proportion",

                    hole    = .65,

                    sort = False,

                    marker = dict(colors = ['lime','darkblue','green', 'purple', 'orange']                    

                   ))





    layout = go.Layout(dict(title = "Distribution of " + title1

                           )

                      ) 

    fig  = go.Figure(data = trace1,layout = layout)

    fig.show()

    

# Donut plot proportion originating from each region

donut_plot(dfcoca1,'Origin_region','Shipment Origin Region')
# Boxplot of Gram_Price by Origin Region

regionpricefig = px.box(dfcoca1[dfcoca1.Origin_region != 'Unknown'],

                        x='Origin_region',

                        y='Gram_Price',

                        points='all',

                        hover_data=['Origin'],

                        title="Price per Gram by Origin Region",

                        labels= {'Gram_Price':"Price per Gram (USD)",

                        'Origin_region':'Origin Region'})



regionpricefig.show()
# Donut plot proportion originating from each region

donut_plot(dfcoca1,'Destination','Shipment Destination')
# Boxplot of Gram_Price by Origin Region

destinationpricefig = px.box(dfcoca1,

                        x='Destination',

                        y='Gram_Price',

                        points='all',

                        hover_data=['Quantity'],

                        title="Price per Gram by Destination",

                        labels= {'Gram_Price':"Price per Gram (USD)"})



destinationpricefig.show()
# Histogram of cocaine purity

dfpure = dfcoca.dropna(subset=['Purity','Gram_Price'])

sns.distplot(dfpure['Purity'], hist=True, rug=True)
dfpure.isnull().sum()

dfpure.Origin = dfpure.Origin.fillna('Unknown')

dfpure.Origin_region = dfpure.Origin_region.fillna('Unknown')
# Scatterplot of PPG by Purity

purityfig = px.scatter(dfpure[dfpure.Origin_region != 'Unknown'],

                 x="Purity",

                 y="Gram_Price",

                 color="Origin_region",

                 hover_data=['Origin'],

                 trendline="ols",

                 title= "Price per gram by Purity",

                 labels= {'Gram_Price':"Price per Gram (USD)",

                          'Purity':'Purity (%)',

                          'Origin_region':'Origin Region'}

                )

purityfig.show()
# Summary of model fitted to LatAm listings

results = px.get_trendline_results(purityfig)

results.px_fit_results[3].summary()