# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Read the dataset for district wise rainfall of India

india_df = pd.read_csv('/kaggle/input/rainfall-in-india/rainfall in india 1901-2015.csv') 

india_df.info()
# Let's take care of the null values if any

# We will front fill the null values to maintain the trend



india_df['JAN'].fillna(method = 'ffill', inplace = True)

india_df['FEB'].fillna(method = 'ffill', inplace = True)

india_df['MAR'].fillna(method = 'ffill', inplace = True)

india_df['APR'].fillna(method = 'ffill', inplace = True)

india_df['MAY'].fillna(method = 'ffill', inplace = True)

india_df['JUN'].fillna(method = 'ffill', inplace = True)

india_df['JUL'].fillna(method = 'ffill', inplace = True)

india_df['AUG'].fillna(method = 'ffill', inplace = True)

india_df['SEP'].fillna(method = 'ffill', inplace = True)

india_df['OCT'].fillna(method = 'ffill', inplace = True)

india_df['NOV'].fillna(method = 'ffill', inplace = True)

india_df['DEC'].fillna(method = 'ffill', inplace = True)

india_df['ANNUAL'].fillna(method = 'ffill', inplace = True)

india_df['Jan-Feb'].fillna(method = 'ffill', inplace = True)

india_df['Mar-May'].fillna(method = 'ffill', inplace = True)

india_df['Jun-Sep'].fillna(method = 'ffill', inplace = True)

india_df['Oct-Dec'].fillna(method = 'ffill', inplace = True)
# Subdivisions

subdivisions_list = ['MADHYA MAHARASHTRA','KONKAN & GOA','COASTAL KARNATAKA','SOUTH INTERIOR KARNATAKA', 'KERALA']



# Now let's carve out the region of interest

western_ghats_df = india_df.loc[india_df['SUBDIVISION'].isin(subdivisions_list)]

rest_of_india_df = india_df.loc[~india_df['SUBDIVISION'].isin(subdivisions_list)]

display(western_ghats_df)



# Madhya Maharashtra subdivision

madhya_maharashtra_df = western_ghats_df.loc[western_ghats_df['SUBDIVISION'] == 'MADHYA MAHARASHTRA']



# Konkan & Goa subdivision

konkan_goa_df = western_ghats_df.loc[western_ghats_df['SUBDIVISION'] == 'KONKAN & GOA']



# Costal Karnataka subdivision

coastal_karnataka_df = western_ghats_df.loc[western_ghats_df['SUBDIVISION'] == 'COASTAL KARNATAKA']



# South Interior Karnataka subdivision

south_interior_karnataka_df = western_ghats_df.loc[western_ghats_df['SUBDIVISION'] == 'SOUTH INTERIOR KARNATAKA']



# Kerala subdivision

kerala_df = western_ghats_df.loc[western_ghats_df['SUBDIVISION'] == 'KERALA']
# Annual subdivision wise rainfall pattern in Western Ghats



plt.figure(figsize=(10,10))

plt.plot(madhya_maharashtra_df['YEAR'], madhya_maharashtra_df['ANNUAL'].ewm(span=10, adjust=False).mean())

plt.plot(konkan_goa_df['YEAR'], konkan_goa_df['ANNUAL'].ewm(span=10, adjust=False).mean())

plt.plot(coastal_karnataka_df['YEAR'], coastal_karnataka_df['ANNUAL'].ewm(span=10, adjust=False).mean())

plt.plot(south_interior_karnataka_df['YEAR'], south_interior_karnataka_df['ANNUAL'].ewm(span=10, adjust=False).mean())

plt.plot(kerala_df['YEAR'], kerala_df['ANNUAL'].ewm(span=10, adjust=False).mean())

plt.ylabel("Annual Rainfall (mm)")

plt.xlabel("Year")

plt.legend(['Madhya Maharashtra', 'Konkan & Goa', 'Coastal Karnataka', 'South Interior Karnataka', 'Kerala'], loc='upper left')

plt.show()
# Rainfall pattern in June-September in Western Ghats



plt.figure(figsize=(10,10))

plt.plot(madhya_maharashtra_df['YEAR'], madhya_maharashtra_df['Jun-Sep'].ewm(span=10, adjust=False).mean())

plt.plot(konkan_goa_df['YEAR'], konkan_goa_df['Jun-Sep'].ewm(span=10, adjust=False).mean())

plt.plot(coastal_karnataka_df['YEAR'], coastal_karnataka_df['Jun-Sep'].ewm(span=10, adjust=False).mean())

plt.plot(south_interior_karnataka_df['YEAR'], south_interior_karnataka_df['Jun-Sep'].ewm(span=10, adjust=False).mean())

plt.plot(kerala_df['YEAR'], kerala_df['Jun-Sep'].ewm(span=10, adjust=False).mean())

plt.ylabel("Rainfall in June-Sep (mm)")

plt.xlabel("Year")

plt.legend(['Madhya Maharashtra', 'Konkan & Goa', 'Coastal Karnataka', 'South Interior Karnataka', 'Kerala'], loc='upper left')

plt.show()
# Rainfall pattern in October-December in Western Ghats



plt.figure(figsize=(10,10))

plt.plot(madhya_maharashtra_df['YEAR'], madhya_maharashtra_df['Oct-Dec'].ewm(span=10, adjust=False).mean())

plt.plot(konkan_goa_df['YEAR'], konkan_goa_df['Oct-Dec'].ewm(span=10, adjust=False).mean())

plt.plot(coastal_karnataka_df['YEAR'], coastal_karnataka_df['Oct-Dec'].ewm(span=10, adjust=False).mean())

plt.plot(south_interior_karnataka_df['YEAR'], south_interior_karnataka_df['Oct-Dec'].ewm(span=10, adjust=False).mean())

plt.plot(kerala_df['YEAR'], kerala_df['Oct-Dec'].ewm(span=10, adjust=False).mean())

plt.ylabel("Rainfall in Oct-Dec (mm)")

plt.xlabel("Year")

plt.legend(['Madhya Maharashtra', 'Konkan & Goa', 'Coastal Karnataka', 'South Interior Karnataka', 'Kerala'], loc='upper left')

plt.show()
# Rainfall pattern in March-May in Western Ghats



plt.figure(figsize=(10,10))

plt.plot(madhya_maharashtra_df['YEAR'], madhya_maharashtra_df['Mar-May'].ewm(span=10, adjust=False).mean())

plt.plot(konkan_goa_df['YEAR'], konkan_goa_df['Mar-May'].ewm(span=10, adjust=False).mean())

plt.plot(coastal_karnataka_df['YEAR'], coastal_karnataka_df['Mar-May'].ewm(span=10, adjust=False).mean())

plt.plot(south_interior_karnataka_df['YEAR'], south_interior_karnataka_df['Mar-May'].ewm(span=10, adjust=False).mean())

plt.plot(kerala_df['YEAR'], kerala_df['Mar-May'].ewm(span=10, adjust=False).mean())

plt.ylabel("Rainfall in March-May (mm)")

plt.xlabel("Year")

plt.legend(['Madhya Maharashtra', 'Konkan & Goa', 'Coastal Karnataka', 'South Interior Karnataka', 'Kerala'], loc='upper left')

plt.show()
# Analyse the seasonal variation in annual rainfall for the subdivsions

sns.set_style('whitegrid')

plt.figure(figsize=(10, 10))

plt.xticks(rotation='vertical')

sns.boxplot(x='SUBDIVISION', y='ANNUAL', data=western_ghats_df)

plt.ylabel("Annual Rainfall in Western Ghats (mm)")

plt.xlabel("Subdivision")

plt.show()
# Annual rainfall pattern in Western Ghats



western_ghats_yearly_df = western_ghats_df.groupby('YEAR').mean().reset_index()



plt.figure(figsize=(10,10))

plt.plot(western_ghats_yearly_df['YEAR'], western_ghats_yearly_df['ANNUAL'].ewm(span=20, adjust=False).mean())

plt.ylabel("Annual Rainfall in Western Ghats (mm)")

plt.xlabel("Year")

plt.show()
# Seasonality in rainfall in Western Ghats

western_ghats_df.groupby('YEAR').sum().drop(['ANNUAL','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec'], axis=1).T.plot(figsize=(10,10), legend=False)

plt.ylabel("Rainfall (mm)")

plt.xlabel("Year")

plt.show()
# Compare rainfall in western ghats with rest of india

rest_of_india_yearly_df = rest_of_india_df.groupby('YEAR').mean().reset_index()



plt.figure(figsize=(10,10))

plt.plot(western_ghats_yearly_df['YEAR'], western_ghats_yearly_df['ANNUAL'].ewm(span=20, adjust=False).mean())

plt.plot(rest_of_india_yearly_df['YEAR'], rest_of_india_yearly_df['ANNUAL'].ewm(span=20, adjust=False).mean())

plt.ylabel("Annual Rainfall (mm)")

plt.xlabel("Year")

plt.legend(['Western Ghats', 'Rest of India'], loc='upper left')

plt.show()