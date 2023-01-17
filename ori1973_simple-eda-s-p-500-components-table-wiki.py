# Import libraries

#------------------

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))



# Market-cap data & List of S&P 500 companies data from wikipedia
df = pd.read_csv("../input/sp_from_wiki.csv")

df.head()
# Number of Columns and Rows for this data-set 

df.shape
# Information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.

df.info()
# Let's see The column labels of the DataFrame

df.columns
# Drop some columns we dont need for this EDA

df.drop(['SEC filings', 'CIK', 'Date first added'], axis=1, inplace=True)

# Detect missing values for an array-like object

df.isnull().sum()
df.columns=(['SecurityName','Symbol', 'Sector', 'Sub_Industry',

       'Location', 'Founded'])

df.head()
# Delete duplicates

list=['FOX', 'UA', 'GOOGL', 'NWS', 'DISCK']

df = df[~df['Symbol'].isin (list)]

len(df)
df.Location.head(10)
# Where most of the companies located

df.Location.str.split(',').str[-1].value_counts().head(10)
# Normalized for percentage and show where most of the companies from.

(df.Location.str.split(',').str[-1].value_counts(normalize=True).head(60)*100).round(2).plot(kind='bar',

title='Location of S&P500 Companies',figsize=(24,8),fontsize=18);

plt.ylabel("Number of companies");
# Do you remember our Founded columns with all those Nan values, lets see out of curiosity, Who is the oldest company    

df.sort_values('Founded')[['SecurityName','Founded']].head(10)
# Normalized as a precentage 

(df.Sector.value_counts(normalize=True)*100).round(2)
# and visualize

df.Sector.value_counts().plot(kind='pie',title='S&P500 Sectors',autopct='%.2f%%',figsize=(14,14),shadow=True,fontsize=15,

                             explode=[0.05 for x in df.Sector.value_counts()])

plt.axis('off');
# Group by sub industry and show the first 10 rows

df.groupby(['Sub_Industry']).size().sort_values(ascending=False).head(10)
# Build a Group by index=Sector and columns= Sub_industry and counting how many companies in each sub_category & category

sub_sector= df.groupby(['Sector','Sub_Industry'])['Symbol'].count().unstack(level = -1,fill_value=0)

sub_sector
for i, x in enumerate(sub_sector.iterrows()):

    plt.figure()

    sub_sector.iloc[i,:].plot.pie(shadow=True,fontsize=12, figsize=(14,14),startangle=90)

    plt.title(sub_sector.index[i], fontsize = 20)

    plt.tight_layout()

    plt.axis('off');
MrktCap_df = pd.read_csv("../input/MrktCap_df.csv")

print(MrktCap_df.shape)

MrktCap_df.head()
# Merge by 'Symbol'

combo = pd.merge(df, MrktCap_df, on='Symbol')

combo.head()
combo['MktCap'] = pd.to_numeric(combo['MktCap'].str[:-1])

combo.head()
combo.MktCap.describe().round(2)
plt.style.use('ggplot')

combo.MktCap.plot(kind='hist', bins=50,xticks=range(0,1000,50), figsize=(20,7), xlim=0, 

fontsize=15, color='red',edgecolor='black',lw=1.2, title='Market Cap Histogram',density=True);
combo.groupby('Sector')['MktCap'].sum().sort_values( ascending=False).plot.pie(autopct='%.2f%%',figsize=(12,12),

shadow=True,fontsize=15, title=("SP500 Sector segmentation by Market Cap"));

plt.axis('off');
cc = combo.iloc[combo.groupby('Sector')['MktCap'].idxmax()].sort_values(by='MktCap',ascending=False)

cc[['Symbol' ,'MktCap']].plot(y='MktCap', kind='pie', labels=cc['SecurityName'], legend=None, figsize=(20,20), radius=2, fontsize=15)

plt.tight_layout();plt.axis('off');cc
combo['Percent'] = round((combo.MktCap / combo.MktCap.sum())*100,2)

cn = combo.nlargest(5,'Percent')

cn.set_index("SecurityName", inplace=True)

cn[['Symbol','Percent']].plot(kind='bar', figsize=(15,5), rot=0, fontsize=15, title=('S&P 500 – Weightings of the Largest Components (%)'))

plt.ylabel('Index Percent')

cn
table = combo.groupby('Sector')['MktCap'].sum().round(-2).sort_values(ascending=False)

plt.style.use('seaborn-dark-palette')

table.plot(kind='barh',figsize=(20,8),title='Total Market Cap by Sector (billions)', fontsize=20);

table
combo.groupby(df.Location.str.split(',').str[-1])['MktCap'].sum().sort_values(ascending=False).nlargest(15).plot.pie(autopct='%.2f%%',figsize=(15,15), fontsize=15,

                                                                                                  explode=[0.05 for x in range(15)], shadow=True, pctdistance=0.85);

plt.title('Market Cap by Locations S&P500 Companies', fontsize=25);

plt.ylabel("");

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle);

plt.tight_layout();