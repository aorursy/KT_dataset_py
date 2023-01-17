# Importing the necessary libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

from plotly.offline import download_plotlyjs, iplot, init_notebook_mode

import plotly.graph_objs as go

import plotly.plotly as py

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()



%matplotlib inline

plt.style.use('ggplot')
# Let's see the number of dataset and their name we have

try:

    dataset = os.listdir('input')

    for var in dataset:

        print(var)

except Exception as e :

    dataset = os.listdir('../input')

    for var in dataset:

        print(var)
# loading datasets



try:

    df_kiva_loans = pd.read_csv('input/kiva_loans.csv')

    df_kiva_mpi = pd.read_csv('input/kiva_mpi_region_locations.csv')

    df_loan_region = pd.read_csv('input/loan_themes_by_region.csv')

    df_loan_ids = pd.read_csv('input/loan_theme_ids.csv')



except Exception as e:

    df_kiva_loans = pd.read_csv('../input/kiva_loans.csv')

    df_kiva_mpi = pd.read_csv('../input/kiva_mpi_region_locations.csv')

    df_loan_region = pd.read_csv('../input/loan_themes_by_region.csv')

    df_loan_ids = pd.read_csv('../input/loan_theme_ids.csv')
# let's see the kiva_loan dataset.

df_kiva_loans.head()
# let's see the kiva_mpi_region_location dataset.

df_kiva_mpi.head()
# let's see the kiva_loan_theme_id dataset.

df_loan_ids.head()
# let's see the kiva_loan_theme_by_region dataset.



df_loan_region.head()


df_kiva_loans.describe()
df_kiva_loans.describe(include=[np.object])


null = df_kiva_loans.isnull().sum()

null_per = (null/len(df_kiva_loans))*100

null_per = pd.DataFrame(data={'Columns':null.index,'Number of Null values':null.values, 'Percentage of null value':null_per.values})

null_per


df_kiva_loans['country'].value_counts()[:10].plot(kind='bar', figsize=(11,6))

plt.title('Top 10 Countires that got maximum number of times Loan')

plt.xlabel('Country')

plt.ylabel('Number of Times')

plt.show()
from wordcloud import WordCloud



names = df_kiva_loans['country'][pd.notnull(df_kiva_loans['country'])]

wc = WordCloud(width=600, height=400, max_font_size=50).generate(' '.join(names))

plt.figure(figsize=(16,8))

plt.title('Wordcloud for Countries', fontsize=35)

plt.imshow(wc)

plt.axis('off')

plt.show()
# Using groupby function here.

country = df_kiva_loans.groupby('country')['funded_amount'].sum().reset_index()

country.sort_values('funded_amount',ascending=False, inplace=True)

country = country[:10]

country.set_index('country', drop=True, inplace=True,)



country.plot(kind='bar', figsize=(12,6), title='Top 10 countries that got maximum amount of total Loan')



plt.show()
# Using groupby function here.



sector = df_kiva_loans.groupby('sector')['funded_amount'].sum().reset_index()

sector.sort_values('funded_amount',ascending=False, inplace=True)

sector = sector[:10]

sector.set_index('sector', drop=True, inplace=True,)





sector.plot(kind='bar',title='Top 10 Sectors that got maximum amount of total loan.', figsize=(12,6))

plt.show()

sector.plot(kind='pie',title='Top 10 Sectors that got maximum amount of total loan.',subplots=True, 

            figsize=(12,6), autopct='%.f%%') 

plt.show()
# Using groupby function here.



activity = df_kiva_loans.groupby('activity')['funded_amount'].sum().reset_index()

activity.sort_values('funded_amount',ascending=False, inplace=True)

activity = activity[:10]



# iplot for interactive plots.

activity.iplot(kind='bar',x='activity', y='funded_amount',

             title='Top 10 activity that got maximum amount of total loan.', xTitle='activity', yTitle='Amount' )

plt.show()

activity.iplot(kind='pie',title='Top 10 activity that got maximum amount of total loan.',

             labels='activity', values='funded_amount', pull=0.2, hole=0.2) 

plt.show()
df_kiva_loans['funded_amount'].plot(kind='hist', bins=50, figsize=(12,6))

plt.xlabel('Funded Amount')

plt.title('Funded Amount Distribution')

plt.show()
temp_df = df_kiva_loans[df_kiva_loans['funded_amount']<10000]

temp_df['funded_amount'].plot(kind='hist', figsize=(12,6), bins=55)

plt.xlabel('Funded Amount')

plt.title('Loan Amount Distribution')

plt.show()
new_df = df_kiva_loans['repayment_interval'].value_counts().reset_index()

new_df.set_index('index', inplace=True, drop=True)

new_df.plot(kind='pie', subplots=True, title='Repayment Interval', autopct='%.f%%', figsize=(12,6))

plt.show()
lender = df_kiva_loans[df_kiva_loans['lender_count']<200]

lender['lender_count'].plot(kind='box', figsize=(12,6), title='Lender count Distribution')

plt.show()
gender = df_kiva_loans['borrower_genders'].value_counts()[:2].reset_index()

gender.iplot(kind='pie', labels='index', values='borrower_genders', title='Gender Distribution of Borrower')

plt.show()
months = df_kiva_loans['term_in_months'].value_counts()[:10].reset_index()

 

months.columns = ['Terms in Months','Frequency']

months.iplot(kind='bar', x='Terms in Months', y= 'Frequency', title='Terms of Months', 

             xTitle='Terms of Month', yTitle='Number of times')
high_mpi = df_kiva_mpi.sort_values('MPI', ascending=False)[:10]

high_mpi.iplot(kind='bar', x='LocationName', y='MPI', title='Top 10 Countries with highest MPI', 

              xTitle='Country', yTitle='MPI')
low_mpi = df_kiva_mpi.sort_values('MPI', )[:10]

low_mpi.iplot(kind='bar', x='LocationName', y='MPI', title='Top 10 Countries with Low MPI', 

              xTitle='Country', yTitle='MPI')
data = [{'lat': df_kiva_mpi['lat'] ,

  'lon': df_kiva_mpi['lon'] ,       

  'marker': {'color': df_kiva_mpi['MPI'] ,

   'line': {'color': 'rgb(40,40,40)', 'width': 0.5},

   'size': 5,

   'sizemode': 'diameter',

    'colorbar': dict(

            title = 'MPI', 

            thickness = 10,           

            outlinecolor = "rgba(68, 68, 68, 0)",            

            ticklen = 3,                       

            dtick = 0.1      )        },

  'text': df_kiva_mpi['LocationName'].astype(str) + '  ->  ' + df_kiva_mpi['MPI'].astype(str) + '  MPI' ,

  'type': 'scattergeo',

  

      }]





layout = go.Layout(

    title = 'MPI',

    showlegend = True,

    geo = dict(

            scope='world',

            projection=dict( type = 'natural earth'),

            showland = True,

            landcolor = 'rgb(217, 217, 217)',

            subunitwidth=1,

            countrywidth=1,

            subunitcolor="rgb(255, 255, 255)",

            countrycolor="rgb(255, 255, 255)"

        ),)



fig =  go.Figure(layout=layout, data=data)

iplot( fig, validate=False)
df_kiva_loans['date'] = pd.to_datetime(df_kiva_loans['date'])

temp_df = df_kiva_loans[['date','funded_amount']]

temp_df.set_index('date', drop=True, inplace=True)

temp_df.plot(figsize=(15,6))

plt.show()
neighbour = ['India' ,'Pakistan', 'China', 'Nepal', 'Bangladesh', 'Bhutan', 'Myanmar (Burma)', 'Afghanistan', 'Sri Lanka', ]

df_india = df_kiva_loans[df_kiva_loans['country'].isin(neighbour)]

print(df_india['country'].unique())

print("The data of Sri Lanka and Bangladesh is not there in kiva's dataset.")
plt.figure(figsize=(15,8))

sns.boxplot(data=df_india, x='country', y='funded_amount')

plt.xlabel('Country', fontsize=25)

plt.show()
temp_df = df_india[df_india['country']=='India']

temp_df = temp_df['sector'].value_counts().reset_index()[:10]

temp_df.iplot(kind='pie', labels='index', values='sector', hole=0.2, pull=0.2, title='Top 10 Sectors')

plt.show()
activity = df_india[df_india['country']=='India']

activity = activity['activity'].value_counts().reset_index()[:10]

activity.iplot(kind='pie', labels='index', values='activity', hole=0.2, pull=0.2, title='Top1 10 Activity')

plt.show()
cities = df_india[df_india['country']=='India']

cities = cities['region'].value_counts().reset_index()[:10]

cities.iplot(kind='pie', labels='index', values='region', hole=0.2, pull=0.2, title='Top 10 City in India')

plt.show()
top_cities = cities['index']

top_cities = top_cities.values

top_cities

new_df = df_loan_region[df_loan_region['region'].isin(top_cities)]

new_df['region'].unique()

lon = new_df['lon'].unique()

lon = np.append(lon,88.8215)

lat = new_df['lat'].unique()

lat = np.append(lat,26.5738)

names = new_df['region'].unique()

data = [{'lat': lat ,

  'lon': lon , 

  'locationmode':'country names' ,      

  'marker': {

   'line': {'color': 'rgb(40,40,40)', 'width': 0.5},

   'size': 5,

   'sizemode': 'diameter',

            },

  'text': names ,

  'type': 'scattergeo',

  

      }]





layout = go.Layout(

    title = 'Top 10 City in India',

    showlegend = True,

    geo = dict(

            scope='asia' ,

            projection=dict( type = 'natural earth'),

            showland = True,

            landcolor = 'rgb(217, 217, 217)',

            subunitwidth=1,

            countrywidth=1,

            subunitcolor="rgb(255, 255, 255)",

            countrycolor="rgb(255, 255, 255)"

        ),)



fig =  go.Figure(layout=layout, data=data)

iplot( fig, validate=False)
temp_df = df_india[df_india['country']=='India']

temp_df = temp_df[temp_df['borrower_genders'].isin(['female', 'male']) ]

temp_df = temp_df['borrower_genders'].value_counts().reset_index()

temp_df.iplot(kind='pie', labels='index', values='borrower_genders', 

              hole=0.2, pull=0.2, title='Gender Distribution of India')

plt.show()
count = df_india['country'].value_counts().reset_index()

count.iplot(kind='pie', labels='index', values='country', pull=0.2, hole=0.2,

            title="Frequency of Loan to India and it's neighbour")

plt.show()
rep_pay = df_india[df_india['country']=='India']

rep_pay = rep_pay['repayment_interval'].value_counts().reset_index()

rep_pay.iplot(kind='pie', labels='index', values='repayment_interval', pull=0.2, hole=0.2, 

             title='Repayment Interval of Loan of India')
region = df_india['region'].value_counts().reset_index()[:10]

region.iplot(kind='bar', x='index', y='region', title='Top 10 Region',

             xTitle='Region', yTitle='Number of times they got Loan')

india = df_india[df_india['country']=='India']

region = india['region'].unique()

temp_df = df_loan_region[df_loan_region['region'].isin(region)]



# Getting latitude, longitude and names of regions

lat = temp_df['lat'].unique()

lon = temp_df['lon'].unique()

names = temp_df['region'].unique()



# Ploting India's map

data = [{'lat': lat ,

  'lon': lon , 

  'locationmode':'country names' ,      

  'marker': {

   'line': {'color': 'rgb(40,40,40)', 'width': 0.5},

   'size': 5,

   'sizemode': 'diameter',

            },

  'text': names ,

  'type': 'scattergeo',

  

      }]





layout = go.Layout(

    title = 'Regions of India',

    showlegend = True,

    geo = dict(

            scope='asia' ,

            projection=dict( type = 'natural earth'),

            showland = True,

            landcolor = 'rgb(217, 217, 217)',

            subunitwidth=1,

            countrywidth=1,

            subunitcolor="rgb(255, 255, 255)",

            countrycolor="rgb(255, 255, 255)"

        ),)



fig =  go.Figure(layout=layout, data=data)

iplot( fig, validate=False)