# Import Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
# Importing dataset

dataset = pd.read_csv('../input/indian-startup-funding/startup_funding.csv')
# Preview dataset

dataset.head()
# Dataset dimensions - (rows, columns)

dataset.shape
# Features data-type

dataset.info()
# Checking for Null values

(dataset.isnull().sum() / dataset.shape[0] * 100).sort_values(ascending = False).round(2).astype(str) + ' %'
# Replacing the commas in 'Amount in USD' feature

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x: str(x).replace(',', ''))
# Fixing the faulty values in 'Amount in USD' feature

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("undisclosed", "0"))

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("Undisclosed", "0"))

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("unknown", "0"))

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("14342000+", "0"))

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("\\\\xc2\\\\xa010000000", "0"))

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("\\\\xc2\\\\xa05000000", "0"))

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("\\\\xc2\\\\xa019350000", "0"))

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("\\\\xc2\\\\xa0600000", "0"))

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("\\\\xc2\\\\xa020000000", "0"))

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("\\\\xc2\\\\xa0N/A", "0"))

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("\\\\xc2\\\\xa016200000", "0"))

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("\\\\xc2\\\\xa0685000", "0"))

dataset['Amount in USD'] = dataset['Amount in USD'].apply(lambda x : str(x).replace("nan", "0"))
# Converting to numeric data-type

dataset['Amount in USD'] = pd.to_numeric(dataset['Amount in USD'])
# Checking for most frequent values in 'Amount in USD'

dataset['Amount in USD'].value_counts(normalize = True).head(10).mul(100).round(2).astype(str) + ' %'
# Replacing 0 in 'Amount in USD' with null values

dataset['Amount in USD'] = dataset['Amount in USD'].replace(0, np.nan)
# Replacing null values with mean

dataset['Amount in USD'].fillna(dataset['Amount in USD'].mean(), inplace = True)
# Fixing the faulty values in 'Date' column

dataset['Date dd/mm/yyyy'][dataset['Date dd/mm/yyyy'] == '12/05.2015'] = '12/05/2015'

dataset['Date dd/mm/yyyy'][dataset['Date dd/mm/yyyy'] == '13/04.2015'] = '13/04/2015'

dataset['Date dd/mm/yyyy'][dataset['Date dd/mm/yyyy'] == '15/01.2015'] = '15/01/2015'

dataset['Date dd/mm/yyyy'][dataset['Date dd/mm/yyyy'] == '22/01//2015'] = '22/01/2015'

dataset['Date dd/mm/yyyy'][dataset['Date dd/mm/yyyy'] == '05/072018'] = '05/07/2018'

dataset['Date dd/mm/yyyy'][dataset['Date dd/mm/yyyy'] == '01/07/015'] = '01/07/2015'

dataset['Date dd/mm/yyyy'][dataset['Date dd/mm/yyyy'] == '05/072018'] = '05/07/2018'

dataset['Date dd/mm/yyyy'][dataset['Date dd/mm/yyyy'] == '\\xc2\\xa010/7/2015'] = '10/07/2015'

dataset['Date dd/mm/yyyy'][dataset['Date dd/mm/yyyy'] == '\\\\xc2\\\\xa010/7/2015'] = '10/07/2015'
# Creating a feature 'Year Month' consisting of year and month

dataset['Year Month'] = (pd.to_datetime(dataset['Date dd/mm/yyyy']).dt.year*100) + (pd.to_datetime(dataset['Date dd/mm/yyyy']).dt.month)
# Dropping the 'Remarks' feature as it contains 86.24% null values

dataset.drop('Remarks', axis = 1, inplace = True)
# Replacing 'Bengaluru' with the more common name 'Bangalore' in the dataset

dataset['City  Location'][dataset['City  Location'] == 'Bengaluru'] = 'Bangalore'
# Replacing 'Undisclosed investors' with a common name 'Undisclosed Investors'

dataset['Investors Name'][dataset['Investors Name'] == 'Undisclosed investors'] = 'Undisclosed Investors'

dataset['Investors Name'][dataset['Investors Name'] == 'Undisclosed Investor'] = 'Undisclosed Investors'

dataset['Investors Name'][dataset['Investors Name'] == 'undisclosed investors'] = 'Undisclosed Investors'

dataset['Investors Name'][dataset['Investors Name'] == 'Undisclosed'] = 'Undisclosed Investors'
# Removing the space in 'Ola Cabs' as it gives two different words in WordCloud

dataset['Startup Name'][dataset['Startup Name'] == 'Ola Cabs'] = 'OlaCabs'
# Replacing with more common word

dataset['InvestmentnType'][dataset['InvestmentnType'] == 'Seed/ Angel Funding'] = 'Seed / Angel Funding'

dataset['InvestmentnType'][dataset['InvestmentnType'] == 'Seed\\\\nFunding'] = 'Seed Funding'

dataset['InvestmentnType'][dataset['InvestmentnType'] == 'Seed/ Angel Funding'] = 'Seed / Angel Funding'

dataset['InvestmentnType'][dataset['InvestmentnType'] == 'Seed/Angel Funding'] = 'Seed / Angel Funding'

dataset['InvestmentnType'][dataset['InvestmentnType'] == 'Angel / Seed Funding'] = 'Seed / Angel Funding'
# Selecting the most frequent values in 'Year Month'

months = dataset['Year Month'].value_counts().head(20)
print('Average number of fundings each month are',months.values.mean())
print('Minimum number of fundings made in a month are',months.values.min())
print('Maximum number of fundings in a month are',months.values.max())
# Creating a barplot for Number of fundings made each month

plt.figure(figsize = (20, 7))

sns.barplot(months.index, months.values, palette = 'colorblind')

plt.title('Number of Fundings each month', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30, 'fontweight' : 'bold'})

plt.xlabel('Months', fontdict = {'fontname' : 'Monospace', 'fontsize' : 20})

plt.ylabel('Number of Fundings', fontdict = {'fontname' : 'Monospace', 'fontsize' : 20})

plt.tick_params(axis = 'x', labelsize = 12)

plt.tick_params(axis = 'y', labelsize = 15)

plt.grid()

plt.show()
# Selecting top 10 cities 

cities = dataset['City  Location'].value_counts().head(10)
# Preview of frequencies of top 10 cities

cities.values
# Preview of names of top 10 countries

cities.index
# Creating a barplot for number of fundings made in each city

plt.figure(figsize = (20, 7))

sns.barplot(cities.values, cities.index, palette = 'Paired')

plt.title('Number of fundings in each city', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30, 'fontweight' : 'bold'})

plt.xlabel('Number of fundings', fontdict = {'fontname' : 'Monospace', 'fontsize' : 20})

plt.ylabel('Cities', fontdict = {'fontname' : 'Monospace', 'fontsize' : 20})

plt.tick_params(labelsize = 15)

plt.grid()

plt.show()
# Selecting the most frequent industries

industry = dataset['Industry Vertical'].value_counts().head(10)
# Preview of frequencies of top 10 industy types

industry.values
# Prevew the names of top 10 industry types

industry.index
# Creating a pie chart of top 10 industries

plt.figure(figsize = (20, 10))

plt.pie(industry.values, labels = industry.index, startangle = 30, explode = (0 , 0.20, 0, 0, 0, 0, 0, 0, 0, 0), 

        shadow = True, autopct = '%1.1f%%')

plt.axis('equal')

plt.title('Industry-wise distribution', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30, 'fontweight' : 'bold'})

plt.show()
# Selecting the most frequent subverticals

subvertical = dataset['SubVertical'].value_counts().head(10)
# Preview of frequencies of top 10 subverticals

subvertical.values
# Preview of names of top 10 subverticals

subvertical.index
# Creating a donut chart of top 10 Subverticals

plt.figure(figsize = (20, 10))

plt.pie(subvertical.values, labels = subvertical.index, startangle = 90, autopct = '%1.1f%%')

centre_circle = plt.Circle((0, 0), 0.70, fc = 'white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.title('Subvertical-wise distribution', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30, 'fontweight' : 'bold'})

plt.axis('equal')

plt.show()
# Selecting the most frequent investors 

investors = dataset['Investors Name'].value_counts().head(10)
# Preview of frequency of top 10 investors

investors.values
# Preview names of top 10 investors

investors.index
# Create a barplot of top 10 investors

plt.figure(figsize = (20, 7))

sns.barplot(investors.values, investors.index, palette = 'deep')

plt.title('Number of investments made by Top Investors', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30, 'fontweight' : 'bold'})

plt.xlabel('Number of Investments', fontdict = {'fontname' : 'Monospace', 'fontsize' : 20})

plt.ylabel('Investor names', fontdict = {'fontname' : 'Monospace', 'fontsize' : 20})

plt.tick_params(labelsize = 15)

plt.grid()

plt.show()
# Preview of top 10 most funded startups

dataset['Amount in USD'].sort_values(ascending = False).head(10)
# Preview of details of top 10 most funded startups

dataset.sort_values(by = 'Amount in USD', ascending = False).head(5)
# Calculating average funding received by a startup

dataset['Amount in USD'].mean() 
# Preview of least funded startups

dataset['Amount in USD'].sort_values().head(10)
# Preview of details of least funded startups

dataset.sort_values(by = 'Amount in USD').head(5)
# Selecting the startups funded the most number of times

most_funded = dataset['Startup Name'].value_counts().head(20)
# Preview frequencies

most_funded.values
# Preview names 

most_funded.index
# Creating a barplot of startups funded most number of times

plt.figure(figsize = (25, 5))

sns.barplot(most_funded.index, most_funded.values, palette = 'colorblind')

plt.title('Most number of times funded startups', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30, 'fontweight' : 'bold'})

plt.xlabel('Startup Name', fontdict = {'fontname' : 'Monospace', 'fontsize' : 20})

plt.ylabel('Number of times funded', fontdict = {'fontname' : 'Monospace', 'fontsize' : 20})

plt.tick_params(axis = 'x', labelsize = 10)

plt.tick_params(axis = 'y', labelsize = 15)

plt.grid()

plt.show()
from wordcloud import WordCloud, STOPWORDS
most_funded_1 = dataset['Startup Name'].value_counts().head(30)
names = most_funded_1.index
# Creating a wordcloud of startup names

plt.figure(figsize = (20, 7))

wordcloud = WordCloud(max_font_size = 25, width = 300, height = 100).generate(' '.join(names))

plt.title('Most number of times funded startups', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30, 'fontweight' : 'bold'})

plt.axis("off")

plt.imshow(wordcloud)

plt.show()
# Preview of types of investments sorted by frequency

dataset['InvestmentnType'].value_counts().head(10)
# Selecting 10 most common investment types

investment_type = dataset['InvestmentnType'].value_counts().head(10)
# Creating a Treemap of Investment types

import squarify

plt.figure(figsize = (10, 7))

squarify.plot(sizes = investment_type.values, label = investment_type.index, value = investment_type.values)

plt.title('Investment type distribution', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30, 'fontweight' : 'bold'})

plt.show()