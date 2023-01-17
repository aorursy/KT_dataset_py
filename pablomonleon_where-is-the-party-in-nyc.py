import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import matplotlib as mpl

from matplotlib import pyplot as plt  

%matplotlib inline

plt.style.use(['fivethirtyeight'])

mpl.rcParams['lines.linewidth'] = 3


req_df = pd.read_csv(

    '../input/311_Service_Requests_from_2010_to_Present.csv', header=0,

    sep=',', parse_dates=['Created Date', 'Closed Date'],

    dayfirst=True, index_col='Created Date')
req_df.head(3)

req_df.shape
req_df['Complaint Type'].unique()

req_df['Complaint Type'].value_counts().plot(kind='bar', figsize=(10,6))

req_df['Borough'].value_counts().plot(kind='pie', title='Boroughs')

req_df[['Longitude', 'Latitude']].plot(kind='scatter',

    x='Longitude', y='Latitude', figsize=(10,10)).axis('equal')
f, ax = plt.subplots()

req_df[req_df['Borough'] == 'MANHATTAN'][['Longitude', 'Latitude']].plot(kind='scatter', x='Longitude', y='Latitude', ax=ax, figsize=(10,10)).axis('equal')

req_df[req_df['Borough'] == 'BROOKLYN'][['Longitude', 'Latitude']].plot(kind='scatter', ax=ax, x='Longitude', y='Latitude', color='r', figsize=(10,10)).axis('equal')

req_df[req_df['Borough'] == 'QUEENS'][['Longitude', 'Latitude']].plot(kind='scatter', ax=ax, x='Longitude', y='Latitude', color='g', figsize=(10,10)).axis('equal')

req_df[req_df['Borough'] == 'BRONX'][['Longitude', 'Latitude']].plot(kind='scatter', ax=ax, x='Longitude', y='Latitude', color='y', figsize=(10,10)).axis('equal')

req_df[req_df['Borough'] == 'STATEN ISLAND'][['Longitude', 'Latitude']].plot(kind='scatter', ax=ax, x='Longitude', y='Latitude', color='m', figsize=(10,10)).axis('equal')
req_df[['Longitude', 'Latitude']].plot(kind='hexbin',

    x='Longitude', y='Latitude', mincnt=1, gridsize=80, colormap='jet', figsize=(10,6)).axis('equal')
req_df[req_df['Complaint Type'] == 'Noise - Commercial']['Descriptor'].value_counts()

req_df[req_df['Complaint Type'] == 'Noise - Street/Sidewalk']['Descriptor'].value_counts()

req_df[req_df['Complaint Type'] == 'Noise - House of Worship']['Descriptor'].value_counts()

req_df['Descriptor'].value_counts()

req_df[req_df['Descriptor'] == 'Loud Music/Party'].plot(

    kind='hexbin', x='Longitude', y='Latitude', gridsize=80,

    colormap='jet', mincnt=1, figsize=(10,6)).axis('equal')
req_df[(req_df.index.hour >= 0) & (req_df.index.hour < 6)

       & ((req_df['Descriptor'] == 'Loud Music/Party')|(req_df['Descriptor'] == 'Loud Talking'))

      ].plot(kind='hexbin', x='Longitude', y='Latitude',

                                       gridsize=80, colormap='jet', mincnt=1, figsize=(10,6)).axis('equal')