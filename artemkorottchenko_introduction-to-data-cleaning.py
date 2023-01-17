# Import necessary libariries 



import numpy as np

import pandas as pd

import re



pd.options.mode.chained_assignment = None # Ignore certain warnings
# Import dataset



df = pd.read_csv('/kaggle/input/used-aircraft-pricing/aircraft_data.csv')
df.head()
df.shape
df.info()
# Luckily, there isn't any missing data here.



df['Make'].isnull().sum()
df['Make'].value_counts()[:30]
df['Make'].nunique()
print('There are a total of {}/{} uppercase rows in this column'.format((df['Make'].str.isupper().sum()), (len(df))))

df['Make'] = df['Make'].str.upper()

print('There are a total of {}/{} uppercase rows in this column'.format((df['Make'].str.isupper().sum()), (len(df))))
# No missing data.



df['Model'].isnull().sum()
df['Model'].value_counts()
# Applying upper method to the Model column. Some aircraft models are stricly numbers, which is why not all have been converted

# to uppercase. 



print('There are a total of {}/{} uppercase rows in this column'.format((df['Model'].str.isupper().sum()), (len(df))))

df['Model'] = df['Model'].str.upper()

print('There are a total of {}/{} uppercase rows in this column'.format((df['Model'].str.isupper().sum()), (len(df))))
# There are a total of 962 different models of aircraft in our dataset.



df['Model'].nunique()
df['Model'].value_counts()
df['National Origin'].isnull().sum()
# Filter for rows that have NaN values. 



df.loc[df['National Origin'].isnull()]
df.loc[137, ['National Origin']] = df.loc[137, ['National Origin']].replace(np.nan, 'United States')

df.loc[137]
df.loc[711, ['National Origin']] = df.loc[137, ['National Origin']].replace(np.nan, 'United States')

df.loc[711]
#Confirm that all 4 were dropped



print(len(df.loc[df['Make'] == 'HOMEBUILT']))

df = df.drop(df.loc[df['Make'] == 'HOMEBUILT'].index)

print(len(df.loc[df['Make'] == 'HOMEBUILT']))
# There is only 1 listing for each manufacturer. We can go ahead and drop both as not having them shouldn't 

# notably impact our dataset. 



print(len(df.loc[df['Make'] == 'MINX']))

print(len(df.loc[df['Make'] == 'BONSALL DC']))

df = df.drop(df.loc[df['Make'] == 'MINX'].index)

df = df.drop(df.loc[df['Make'] == 'BONSALL DC'].index)

print(len(df.loc[df['Make'] == 'MINX']))

print(len(df.loc[df['Make'] == 'BONSALL DC']))
# Confirm that the right amount of rows were dropped. 



print('length of dataset before dropping 6 rows: {}'.format(len(df)))

print('length of dataset after dropping 6 rows: {}'.format(len(df)))
# Similar to the other columns, apply the upper method to stay consistent.



print('There are a total of {}/{} uppercase rows in this column'.format((df['National Origin'].str.isupper().sum()), (len(df))))

df['National Origin'] = df['National Origin'].str.upper()

print('There are a total of {}/{} uppercase rows in this column'.format((df['National Origin'].str.isupper().sum()), (len(df))))
df['National Origin'].value_counts()
# Find the index value of the row that has the incorrect spelling of Switzerland



df.loc[df['National Origin'] == 'SWTIZERLAND']
# Update and confirm the results



df.loc[186, 'National Origin'] = df.loc[186, 'National Origin'].replace('SWTIZERLAND', 'SWITZERLAND')

df['National Origin'].value_counts()
# According to our dataset, most aircraft listings are made in the United States.



print('Aircraft found in the dataset are manufactured in {} different countries.'.format(df['National Origin'].nunique()))

print('')

print(df['National Origin'].value_counts())
df = df.rename(columns={'National Origin': 'Country of Origin'})
df.head()
# No missing data 



df['Category'].isnull().sum()
df['Category'].value_counts()
df['Category'] = np.where((df['Category'] == 'Single Piston'), 'Single Engine Piston', df['Category'])

df['Category'] = np.where((df['Category'] == 'Twin Piston'), 'Multi Engine Piston', df['Category'])

df['Category'] = np.where((df['Category'] == 'Turboprops'), 'Turboprop', df['Category'])
# Amend 'Gliders | Sailplanes' to 'Gliders/Sailplanes' to stay consistent with the 'Military/Classic/Vintage' name format.



df['Category'] = np.where((df['Category'] == 'Gliders | Sailplanes'), 'Gliders/Sailplanes', df['Category'])
# Ensure the changes have been applied



df['Category'].value_counts()
# Similar to the rest of the columns, apply the upper method to stay consistent.



print('There are a total of {}/{} uppercase rows in this column'.format((df['Category'].str.isupper().sum()), (len(df))))

df['Category'] = df['Category'].str.upper()

print('There are a total of {}/{} uppercase rows in this column'.format((df['Category'].str.isupper().sum()), (len(df))))
# Let's see what the dataframe looks like after making various changes to the Category, 

# Make, Model, and Country of Origin columns



df.head()
# No missing data in this column



df['Year'].isnull().sum()
df['Year'].value_counts()
# The data is for 94 different years

df['Year'].nunique()
df['Year'].unique()
print("There are {} rows that have 'Not listed' entered in the year column.".format(len(df[df['Year'] == 'Not Listed'])))

print('')

df[df['Year'] == 'Not Listed'].head()
print("There are {} rows that have '-' entered in the year column.".format(len(df[df['Year'] == '-'])))

print('')

df[df['Year'] == '-'].head()
# Drop the columns that don't have an actual year. Total rows to drop = 34 + 44 = 78 



print('Length of dataset prior to dropping rows with missing data: {}'.format(len(df)))

df = df.drop(df.loc[df['Year'] == 'Not Listed'].index)

df = df.drop(df.loc[df['Year'] == '-'].index)

print('Length of dataset after dropping rows with missing data: {}'.format(len(df)))
# Ensure the right amount of rows were dropped



2524-2446
# Looks good.



df['Year'].unique()
# Last step is to convert the 'Year' column to type = integer



df['Year'] = df['Year'].astype(np.int64)

df['Year'].dtype
df.head()
df['Total Hours'].isnull().sum()
df['Total Hours'].value_counts(dropna=False)
len(df.loc[df['Total Hours'].isnull()])
print("Total number of NaN: {} and '-': {} BEFORE converting to 0.".format((len(df.loc[df['Total Hours'].isnull()])), len(df.loc[df['Total Hours'] == '-'])))



df['Total Hours'] = np.where((df['Total Hours'] == '-'), 0, df['Total Hours'])

df['Total Hours'] = np.where((df['Total Hours'].isnull()), 0, df['Total Hours'])

df['Total Hours'] = np.where((df['Total Hours'] == '0'), 0, df['Total Hours'])



print("Total number of NaN: {} and '-': {} AFTER converting to 0.".format((len(df.loc[df['Total Hours'].isnull()])), len(df.loc[df['Total Hours'] == '-'])))
# The data is very messy. There are letters, commas, colons, and periods within the data. Let's clean it up.



df['Total Hours'].value_counts()[-20:]
# Let's filter for rows that aren't pure digits. 



df_messy_hours = df.loc[~df['Total Hours'].astype(str).str.isdigit()]

print(len(df_messy_hours))

df_messy_hours
# Search for rows that have 'h' or 'H' within them.



contains_h = df_messy_hours[df_messy_hours['Total Hours'].str.contains('h')]

print(len(contains_h))

contains_h.head()
contains_H = df_messy_hours[df_messy_hours['Total Hours'].str.contains('H')]

print(len(contains_H))

contains_H.tail()
# Remove all letters from these rows so that only numbers remain



contains_h['Total Hours'] = contains_h['Total Hours'].astype(str).str.replace('[^0-9]', '')

contains_H['Total Hours'] = contains_H['Total Hours'].astype(str).str.replace('[^0-9]', '')
# Drop row 1928 since it already exists in the 'contains_h' subset



contains_H.drop(1928, inplace=True)

len(contains_H)
contains_H
# Amending total hours to correct number.



contains_H.loc[2325, 'Total Hours'] = contains_H.loc[2325, 'Total Hours'].replace('122200', 'NaN')

contains_H.loc[2372, 'Total Hours'] = contains_H.loc[2372, 'Total Hours'].replace('8207', '821')

contains_H.loc[2397, 'Total Hours'] = contains_H.loc[2397, 'Total Hours'].replace('1883145', 'NaN')
# Update our df with the correct values from above. 



df.update(contains_h)

df.update(contains_H)
df
df.loc[~df['Total Hours'].astype(str).str.isdigit()]
# Filter rows that aren't pure digits again



messy_hours2 = df.loc[~df['Total Hours'].astype(str).str.isdigit()]
no_letters = messy_hours2[pd.to_numeric(messy_hours2['Total Hours'], errors='coerce').notnull()]

print(len(no_letters))

no_letters[30:]
no_letters.loc[1929, 'Total Hours'] = no_letters.loc[1929, 'Total Hours'].replace('1.978', '1978')

no_letters.loc[2023, 'Total Hours'] = no_letters.loc[2023, 'Total Hours'].replace('5.198', '5198')

no_letters.loc[2093, 'Total Hours'] = no_letters.loc[2093, 'Total Hours'].replace('5.74', '574')

no_letters.loc[2115, 'Total Hours'] = no_letters.loc[2115, 'Total Hours'].replace('5.497', '5497')

no_letters.loc[2127, 'Total Hours'] = no_letters.loc[2127, 'Total Hours'].replace('5.615', '5615')

no_letters.loc[2230, 'Total Hours'] = no_letters.loc[2230, 'Total Hours'].replace('1.06', '106')

no_letters.loc[2331, 'Total Hours'] = no_letters.loc[2331, 'Total Hours'].replace('10.72', '1072')
# Convert the numbers to integers



no_letters['Total Hours'] = no_letters['Total Hours'].astype(float).round().astype(int)
# Update the dataframe with the updated rows from the 'no_letters' df. 



df.update(no_letters)
# We currently don't have any values that are null in the 'Total Hours' column



df['Total Hours'].isnull().sum()
# Convert to integers



df['Total Hours'] = pd.to_numeric(df['Total Hours'], errors='coerce')
# After the integer conversion, 109 rows will need to be dropped.



print("There are {} rows with NaN that need to be dropped". format(df['Total Hours'].isnull().sum()))

df.dropna(subset=['Total Hours'], inplace=True)

print("There are {} rows with NaN remaining in the 'Total Hours' column". format(df['Total Hours'].isnull().sum()))
df.head()
df['Condition'].value_counts(dropna=False)
# Let's filter for aircraft with 0 Total Hours, manufactured between 2018-2020 and don't have a listed condition. 

# These should all be listed as New according to the assumptions above. 



print(len(df.loc[(df['Total Hours'] == 0) & (df['Year'] >= 2018) & (df['Condition'].isnull())]))

df['Condition'] = np.where((df['Total Hours'] == 0) & (df['Year'] >= 2018) & (df['Condition'].isnull()) , 'New', df['Condition'])

print(len(df.loc[(df['Total Hours'] == 0) & (df['Year'] >= 2018) & (df['Condition'].isnull())]))
# 12 rows were updated, 600 remain. 



df['Condition'].isnull().sum()
# Filter for aircraft that are listed as having Total Hours greater than 0 and the condition listed as NaN.

# Change these all to Used



print(len(df.loc[(df['Total Hours'] != 0) & (df['Condition'].isnull())]))

df['Condition'] = np.where((df['Total Hours'] != 0) & (df['Condition'].isnull()) , 'Used', df['Condition'])

print(len(df.loc[(df['Total Hours'] != 0) & (df['Condition'].isnull())]))
# 575 rows were updated, 25 remain. 



df['Condition'].isnull().sum()
# Let's look at the remaining listings



df[df['Condition'].isnull()]
print('Length of dataset prior to dropping NaN values from the Condition column: {}'.format(len(df)))

df.dropna(subset=['Condition'], inplace=True)

print('Length of dataset after dropping NaN values from the Condition column: {}'.format(len(df)))
df['Condition'].value_counts(dropna=False)
# Drop used aircraft with 0 total hours:



len(df[(df['Total Hours'] == 0) & (df['Condition'] == 'Used')])
print('Length of dataset prior to dropping NaN values from the Condition column: {}'.format(len(df)))

df = df.drop(df[(df['Total Hours'] == 0) & (df['Condition'] == 'Used')].index)

print('Length of dataset after dropping NaN values from the Condition column: {}'.format(len(df)))
# Ensure the correct amount of rows were dropped.



2312-2241
# Project aircraft, similar to homebuilt aircraft mentioned above can vary widely and there isn't sufficient data

# to carry out a meaningful analysis. But for the sake of curiousity I'll leave this for now. 



df[df['Condition'] == 'Project'] 
# Looks good.



df['Condition'].value_counts()
# One more time - apply the upper method on the entire column.



print('There are a total of {}/{} uppercase rows in this column'.format((df['Condition'].str.isupper().sum()), (len(df))))

df['Condition'] = df['Condition'].str.upper()

print('There are a total of {}/{} uppercase rows in this column'.format((df['Condition'].str.isupper().sum()), (len(df))))
# The data is a little messy.



df['Price'].value_counts().tail(20)
# No missing data in the Price column. Let's check the currency column.



df['Price'].isnull().sum()
# 415 NaN values in the Currency column



df['Currency'].isnull().sum()
# There are 5 different currencies within our dataset.



df['Currency'].value_counts(dropna=False)
df[df['Currency'].isnull()].head()
df[df['Currency'] == 'EUR'].tail()
df[df['Currency'] == 'GBP'].tail()
# Create a variable for EUR



price_eur = df[df['Price'].str.contains(r'â‚¬', flags=re.IGNORECASE, regex=True, na=False)]

price_eur
# Update the Currency column for the listing below



price_eur[price_eur['Currency'].isnull()]
df.loc[2361, ['Currency']] = df.loc[2361, ['Currency']].replace(np.nan, 'EUR')

df.loc[2361, 'Currency']
# We can now go ahead and remove the unwanted characters within the Price column for Euros



price_eur['Price'] = price_eur['Price'].str.replace('Price:', '')

price_eur['Price'] = price_eur['Price'].str.replace('â‚¬', '')

print(len(price_eur))

price_eur
df.update(price_eur)
# Let's repeat the steps above for GBP now. 



price_gbp = df[df['Price'].str.contains(r'£', flags=re.IGNORECASE, regex=True, na=False)]

print(len(price_gbp))

price_gbp
# No missing values for Currency = GBP



price_gbp[price_gbp['Currency'].isnull()]
# Remove all unwaned characters within the GBP rows in the Price column



price_gbp['Price'] = price_gbp['Price'].str.replace('Price: ', '')

price_gbp['Price'] = price_gbp['Price'].str.replace('Â£', '')

print(len(price_gbp))

price_gbp
df.update(price_gbp)
# Lastly, follow the steps above for USD



price_usd = df[df['Price'].str.contains(r'USD', flags=re.IGNORECASE, regex=True, na=False)]

print(len(price_usd))

price_usd
# Replace the NaN rows in the Currency column with 'USD'



price_usd['Currency'] = price_usd['Currency'].replace(np.nan, 'USD')

price_usd.head()
# Remove all unwanted characters



price_usd['Price'] = price_usd['Price'].str.replace('Price: USD', '')

price_usd['Price'] = price_usd['Price'].str.replace('$', '')

price_usd.head()
# Update the main df



df.update(price_usd)
# Only 1 NaN remains. 

# Drop row with index 2260 since it doesn't have a price, nor currency.



print(df['Currency'].isnull().sum())

df[df['Currency'].isnull()]
print('Length of dataset before dropping row at index 2260: {}'.format(len(df)))

df.drop(2260, inplace=True)

print('Length of dataset before dropping row at index 2260: {}'.format(len(df)))
df['Currency'].isnull().sum()
df['Currency'].value_counts()
# Let's quickly take a look at the other currencies

# CAD looks good



df[df['Currency'] == 'CAD']
# Let's quickly take a look at the other currencies

# CHF looks good



df[df['Currency'] == 'CHF']
df['Price'].value_counts()
# Lastly, drop unwanted characters such as the '$' symbol from the entire Price column.



df['Price'] = df['Price'].str.replace('$', '')

df['Price'] = df['Price'].str.replace(' ', '')

df['Price'] = df['Price'].str.replace(',', '')



df['Price'].value_counts()
# Price column looks good now, we can move on. 



df.head()
df.head()
# 11 missing rows in this column



df['Location'].isnull().sum()
df[df['Location'].isnull()]
# Drop the 11 rows with NaN values in the Location column.



print('Length of dataset before dropping NaN values from the Location column: {}'. format(len(df)))

df.dropna(subset=['Location'], inplace=True)

print('Length of dataset before dropping NaN values from the Location column: {}'. format(len(df)))
df
# Remove blank spaces from Location column



df.Location = df.Location.str.replace(' ', '')
# I manually kept adding countries to this list as I kept looking through the Location column. 



country_list = ['UnitedKingdom','Monaco', 'United Kingdom', 'USA', 'Canada', 'Luxembourg', 'Germany', 'Austria',

                    'Monaco', 'Poland', 'Belgium', 'Russian Federation', 'Netherlands', 'Sweden',

                    'Norway', 'Switzerland', 'France', 'Spain','Denmark', 'Lithuania', 'Turkey', 'Italy', 

                'Iceland', 'SouthAfrica', 'UnitedStates', 'CzechRepublic', 'NewZealand', 'Brazil', 'Australia', 

                'Bulgaria', 'CostaRica', 'RussianFederation', 'Chile', 'Nigeria', 'Pakistan', 'Indonesia', 

                'Venezuela', 'Malaysia', 'Congo', 'NewGuinea', 'UnitedArabEmirates', 'Singapore', 'CAN', 'POL',

                'DEU', 'FRA', 'ITA', 'ZAF', 'AUS', 'ARG', 'SRB', 'CZE', 'NLD', 'MEX', 'ESP', 'AUS', 'URY', 'KEN', 'CHE']



pattern = '|'.join(country_list)
# Create a function to search through the Location column and extract country names. A Country column is created with the 

# individual Country names



def pattern_search(search_str:str, search_list:str):



    search_obj = re.search(search_list, search_str)

    if search_obj :

        return_str = search_str[search_obj.start(): search_obj.end()]

    else:

        return_str = np.nan

    return return_str



df['Country'] = df['Location'].astype(str).apply(lambda x: pattern_search(search_str=x, search_list=pattern))

df
df['Country'].isnull().sum()
# In the interest of time I'm going to completely drop these rows.



df.dropna(subset=['Country'], inplace=True)

df['Country'].isnull().sum()
df.head()
# Some of the countries are repeated. I'll have to manually update these. 



df['Country'].value_counts()
# Manual changes to ensure that countries aren't double counted and have a consistent format.



df['Country'] = np.where((df['Country'] == 'FRA'), 'France', df['Country'])

df['Country'] = np.where((df['Country'] == 'MEX'), 'Mexico', df['Country'])

df['Country'] = np.where((df['Country'] == 'URY'), 'Uruguay', df['Country'])

df['Country'] = np.where((df['Country'] == 'KEN'), 'Kenya', df['Country'])

df['Country'] = np.where((df['Country'] == 'ITA'), 'Italy', df['Country'])

df['Country'] = np.where((df['Country'] == 'ESP'), 'Spain', df['Country'])

df['Country'] = np.where((df['Country'] == 'NLD'), 'Netherlands', df['Country'])

df['Country'] = np.where((df['Country'] == 'CZE'), 'Czech Republic', df['Country'])

df['Country'] = np.where((df['Country'] == 'SRB'), 'Serbia', df['Country'])

df['Country'] = np.where((df['Country'] == 'ARG'), 'Argentina', df['Country'])

df['Country'] = np.where((df['Country'] == 'CzechRepublic'), 'Czech Republic', df['Country'])

df['Country'] = np.where((df['Country'] == 'CostaRica'), 'Costa Rica', df['Country'])

df['Country'] = np.where((df['Country'] == 'UnitedArabEmirates'), 'United Arab Emirates', df['Country'])

df['Country'] = np.where((df['Country'] == 'RussianFederation'), 'Russia', df['Country'])

df['Country'] = np.where((df['Country'] == 'POL'), 'Poland', df['Country'])

df['Country'] = np.where((df['Country'] == 'DEU'), 'Germany', df['Country'])

df['Country'] = np.where((df['Country'] == 'AUS'), 'Australia', df['Country'])

df['Country'] = np.where((df['Country'] == 'SouthAfrica'), 'South Africa', df['Country'])

df['Country'] = np.where((df['Country'] == 'ZAF'), 'South Africa', df['Country'])

df['Country'] = np.where((df['Country'] == 'UnitedKingdom'), 'United Kingdom', df['Country'])

df['Country'] = np.where((df['Country'] == 'CHE'), 'Switzerland', df['Country'])

df['Country'] = np.where((df['Country'] == 'CAN'), 'Canada', df['Country'])

df['Country'] = np.where((df['Country'] == 'NewGuinea'), 'New Guinea', df['Country'])

df['Country'] = np.where((df['Country'] == 'United States'), 'USA', df['Country'])
df['Country'].value_counts()
# Double check the data. 



df[df['Country'] == 'Canada']
# Copy/Paste from https://gist.github.com/JeffPaine/3083347



states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 

          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 

          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 

          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 

          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]



pattern = '|'.join(states)
df['US - State'] = df['Location'].astype(str).apply(lambda x: pattern_search(search_str=x, search_list=pattern))
df.head()
# Manual changes to correct the Country column



df['Country'] = np.where((df['Country'] == 'Canada') & (df['Location'] == 'NorthAmerica+Canada,Mexico'), 'Mexico', df['Country'])

df['Country'] = np.where((df['Country'] == 'Canada') & (df['Location'] == 'NorthAmerica+Canada,UnitedStates'), 'USA', df['Country'])

df['Country'] = np.where((df['Country'] == 'Canada') & (df['Location'] == 'NorthAmerica+Canada,Canada'), 'Canada', df['Country'])

df['Country'] = np.where((df['Country'] == 'Canada') & (df['US - State'] != 'CA') & (df['US - State'].notnull()), 'USA', df['Country'])

df['US - State'] = np.where((df['Country'] == 'Canada') & (df['US - State'].notnull()), np.nan, df['US - State'])

df['Country'] = np.where((df['Country'] == 'Canada') & (df['Location'] == 'NorthAmerica+Canada,UnitedStates-CA'), 'USA', df['Country'])

df['US - State'] = np.where((df['Country'] == 'USA') & (df['Location'] == 'NorthAmerica+Canada,UnitedStates-CA'), 'CA', df['US - State'])

df['US - State'] = np.where((df['Country'] != 'USA') & (df['US - State'].notnull()), np.nan, df['US - State'])
# While cross checking each country using the code below (I appplied the filter below for all countries, one at a time), 

# I noticed that Uruguay was incorrectly entered as the Country for some of the listings. 



df[(df['Country'] == 'Uruguay')]
# Amendments to correct Uruguay:



df['Country'] = np.where((df['Country'] == 'Uruguay') & (df['Location'] == 'HAWKESBURY,\n\tON\n\tCAN'), 'Canada', df['Country']) 

df['Country'] = np.where((df['Country'] == 'Uruguay') & (df['Location'] == 'HAWKESBURY,\n\tQC\n\tCAN'), 'Canada', df['Country']) 

df['Country'] = np.where((df['Country'] == 'Uruguay') & (df['Location'] == 'HAWKESBURY\n\t\n\tUSA'), 'USA', df['Country'])

df[(df['Country'] == 'Uruguay')]
df['Country'].value_counts()[:10]
# Australia also has several errors. See cell below for amendments. 



df[(df['Country'] == 'Australia')]
# Australia amendments



df['Country'] = np.where((df['Country'] == 'Australia') & (df['Location'] == 'Australia&NZ,NewZealand'), 'New Zealand', df['Country']) 

df['Country'] = np.where((df['Country'] == 'Australia') & (df['Location'] == 'AUSTIN,\n\tTX\n\tUSA'), 'USA', df['Country']) 
df['Country'].isnull().sum()
df['Country'].value_counts()
df['US - State'].isnull().sum()
df['US - State'].value_counts()
# Last step - Applying upper method



print('There are a total of {}/{} uppercase rows in this column'.format((df['Country'].str.isupper().sum()), (len(df))))

df['Country'] = df['Country'].str.upper()

print('There are a total of {}/{} uppercase rows in this column'.format((df['Country'].str.isupper().sum()), (len(df))))
df.head()
df.info()
# Drop - Location column



location_df = df['Location']

df.drop(['Location'], axis=1, inplace=True)

df.head()
# Drop - Engine 1 Hours, Engine 2 Hours, Prop 1 Hours, Prop 2 Hours, Total Seats, Flight Rules



unused_columns = df[['Engine 1 Hours', 'Engine 2 Hours', 'Prop 1 Hours', 'Prop 2 Hours', 'Total Seats', 'Flight Rules']]

df.drop(['Engine 1 Hours', 'Engine 2 Hours', 'Prop 1 Hours', 'Prop 2 Hours', 'Total Seats', 'Flight Rules'], axis=1, inplace=True)
# Drop S/N and REG columns



sn_reg = df[['S/N', 'REG']]

df.drop(['S/N', 'REG'], axis=1, inplace=True)
# Rename and Rearrange the columns



df = df.rename(columns={'Country': 'Location - Country', 'US - State': 'Location - US State'})



df.head()
# Rearrange the columns



rearrange_columns = df.columns.tolist()

rearrange_columns = [

 'Condition',

 'Category',

 'Year',

 'Make',

 'Model',

 'Country of Origin',

 'Total Hours',

 'Location - Country',

 'Location - US State',

 'Price',

 'Currency', 

 ]
df = df[rearrange_columns]

df.head()
# Convert columns to correct data types



df['Total Hours'] = df['Total Hours'].astype(np.int64)

df['Price'] = df['Price'].astype(np.int64)

df['Year'] = pd.to_datetime(df['Year'], format='%Y').dt.year

df.info()
df.head()
# Convert all Prices to USD



df.loc[df['Currency'] == 'EUR', 'Price'] = df.loc[df['Currency'] == 'EUR', 'Price']*(1/0.8888)

df.loc[df['Currency'] == 'GBP', 'Price'] = df.loc[df['Currency'] == 'GBP', 'Price']*(1/0.8023)

df.loc[df['Currency'] == 'CAD', 'Price'] = df.loc[df['Currency'] == 'CAD', 'Price']*(1/1.3591)

df.loc[df['Currency'] == 'CHF', 'Price'] = df.loc[df['Currency'] == 'CHF', 'Price']*(1/0.9458)

df['Price'] = df['Price'].astype(np.int64)
# We can now drop the Currency column since all of prices are in USD



df.drop(['Currency'], axis=1, inplace=True)
df.head()
df.to_csv('clean_aircraft_data.csv')