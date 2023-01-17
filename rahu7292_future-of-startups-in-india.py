# Importing useful Libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Importing dataset



try:

    df_startup = pd.read_csv('input/startup_funding.csv')

    

except Exception as e:

    df_startup = pd.read_csv('../input/startup_funding.csv')
# Let's check top 5 entries.



df_startup.head()
# Basic information.



print(df_startup.info())
# Dealing with missing data.



missing_data = df_startup.isnull().sum().sort_values(ascending=False)

missing_data = (missing_data/len(df_startup))*100

print(missing_data)
# Removing unwanted column.



try:

    df_startup.drop('Remarks', axis=1, inplace=True)

except Exception:

    pass
# Converting AmountInUSD column into integer type as it in string type.

print(f"The Type of AmountInUSD column is\t {type(df_startup['AmountInUSD'][0])}")

print("We have to convert it into integer type.")



df_startup['AmountInUSD'] = df_startup['AmountInUSD'].apply(lambda x: float(str(x).replace(",","")))

df_startup['AmountInUSD'] = pd.to_numeric(df_startup['AmountInUSD'])



print(f"The type of AmountInUSD column after conversion is\t {type(df_startup['AmountInUSD'][0])}")
# Checking dataset after cleaning it.



df_startup.head()
print(f"The numebr of Unique Startups are\t {df_startup['StartupName'].nunique()} ")


df_startup['StartupName'].value_counts()[:10].plot(kind='bar', figsize=(10,5))

plt.show()
temp_df = df_startup.sort_values('AmountInUSD', ascending=False, )

temp_df = temp_df[['StartupName', 'AmountInUSD']][:10].set_index('StartupName', drop=True, )

temp_df.plot(kind='bar', figsize=(10,5))

plt.show()



temp_df.plot(kind='pie', subplots=True, figsize=(12,6), autopct='%.f%%')

plt.show()



temp_df.T
new_df = df_startup[df_startup['AmountInUSD'].notnull()]

new_df.sort_values('AmountInUSD', inplace=True)

new_df = new_df[['StartupName', 'AmountInUSD']][:10].set_index('StartupName', drop=True)

new_df.plot(kind='bar', figsize=(10,5))

plt.show()

new_df.T
print(f"The number of unique investors in Indian ecosystem between 2015 to 2017 are\t {df_startup['InvestorsName'].nunique()}")
investor = df_startup.groupby('InvestorsName')['AmountInUSD'].sum().reset_index()

investor.sort_values('AmountInUSD', inplace=True, ascending=False)

investor.reset_index()[:10]
df_startup['IndustryVertical'].value_counts()[:10].reset_index()
temp_df = df_startup[df_startup['IndustryVertical'].isin(['Technology','technology'])]

temp_df = temp_df['SubVertical'].value_counts()[:10].reset_index()

temp_df.columns = ['Sub Industry', 'Number of times']

temp_df.set_index('Sub Industry', drop=True, inplace=True)

temp_df.plot(kind='bar', figsize=(10,5))

plt.show()

temp_df.T
# converting ecommerce into Ecommerce

df_startup['IndustryVertical'] = df_startup['IndustryVertical'].apply(lambda x: 'ECommerce' if x=='eCommerce' else x)



new_df = df_startup.groupby('IndustryVertical')['AmountInUSD'].sum().reset_index()

new_df.sort_values('AmountInUSD', inplace=True, ascending=False)



new_df.set_index('IndustryVertical', inplace=True, drop=True)

new_df[:10].plot(kind='bar', figsize=(10,5))

plt.show()



new_df[:10].plot(kind='pie', subplots=True, figsize=(12,6), autopct='%.f%%')

plt.show()



new_df[:10].T
city = df_startup['CityLocation'].value_counts()[:10].reset_index()

city.columns = ['City', 'Number of Startups']

city.set_index('City', drop=True, inplace=True)

city.plot(kind='bar', figsize=(10,5), title='Top 10 Cities which have maximum startups')

plt.show()



city.plot(kind='pie', subplots=True, figsize=(12,6), autopct='%.f%%')

plt.show()

city.T

# Dealing with duplicates name.

l = ['Bangalore', 'Bangalore/ Bangkok', 'SFO / Bangalore','Seattle / Bangalore', 'Bangalore / SFO',

     'Bangalore / Palo Alto', 'Bangalore / San Mateo', 'Bangalore / USA',   ]

df_startup['CityLocation'] = df_startup['CityLocation'].apply(lambda x: 'Bangalore' if x in l else x )



city_df = df_startup.groupby('CityLocation')['AmountInUSD'].sum().reset_index()

city_df.sort_values('AmountInUSD', ascending=False, inplace=True)

city_df = city_df[:10]

city_df.reset_index(inplace=True, drop=True)

city_df.set_index('CityLocation', inplace=True, drop=True)

city_df.plot(kind='bar', figsize=(12,6), title='Top 10 cities which got maximum funding')

plt.show()



city_df.plot(kind='pie', figsize=(12,6), autopct='%.f%%', subplots=True)

plt.show()



city_df.T
# top_city is the cities which have maximum funding.

top_city = city_df.index

temp_df = df_startup[df_startup['CityLocation'].isin(top_city)]

temp_df = temp_df[['CityLocation', 'AmountInUSD']]



plt.figure(figsize=(12,6))

sns.swarmplot(data=temp_df, x=temp_df['CityLocation'], y=temp_df['AmountInUSD'])



plt.show()
# Converting some Dates into its proper format, as they were entered wrong.

def convert(x):

    if x=='12/05.2015':

        return '12/05/2015'

    elif x=='13/04.2015':

        return '13/04/2015'

    elif x=='15/01.2015':

        return '15/01/2015'

    elif x=='22/01//2015':

        return '22/01/2015'

    else:

        return x



df_startup['Date'] = df_startup['Date'].apply(convert)



# Need to convert string into datetime format object.

df_startup['year'] = (pd.to_datetime(df_startup['Date']).dt.year)

df_startup.head()



plt.figure(figsize=(12,6))

sns.boxenplot(data=df_startup, x='year', y='AmountInUSD')

plt.show()
# New column with date in year-month format.



df_startup['year_month'] = (pd.to_datetime(df_startup['Date']).dt.year*100) + (pd.to_datetime(df_startup['Date']).dt.month)



times = df_startup['year_month'].value_counts().reset_index()

times.set_index('index', drop=True, inplace=True)

times.index.name = 'Month-Year'

times.columns = ['Number of Startups']

times.sort_index(inplace=True)

times.plot(kind='bar', figsize=(13,7), title='Number of Startups over month')

plt.show()



# Let's see number of startups over Years

df_startup['year'].value_counts().plot(kind='bar', figsize=(13,7), title='Number of Startups over Years')

plt.show()

df_startup['year'].value_counts().plot(kind='pie', figsize=(11,7), title='Number of Startups over Years', subplots=True, autopct='%.f%%')

plt.show()
# Dealing with incorrect entries.

def convert(x):

    if x== 'SeedFunding':

        return 'Seed Funding'

    elif x== 'PrivateEquity':

        return 'Private Equity'

    elif x== 'Crowd funding':

        return 'Crowd Funding'

    else:

        return x



df_startup['InvestmentType'] = df_startup['InvestmentType'].apply(convert)



df_startup['InvestmentType'].value_counts().plot(kind='bar', figsize=(12,6), title='Type of Investment')

plt.show()



df_startup['InvestmentType'].value_counts().plot(kind='pie', figsize=(12,6), subplots=True, autopct='%.f%%')

plt.show()