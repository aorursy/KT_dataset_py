import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

sns.set(style='darkgrid',context='notebook')

%matplotlib inline

from pandas_profiling import ProfileReport



from plotly.subplots import make_subplots

import plotly.graph_objects as go



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings('ignore')
city_day=pd.read_csv('/kaggle/input/air-quality-data-in-india/city_day.csv')

print(city_day.shape)

city_day.head()
city_day.info()
city_day.describe()
def missing_values_table(df):

        # Total missing values

        missing_val = df.isnull().sum()

        

        # Percentage of missing values

        missing_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        missing_val_table = pd.concat([missing_val, missing_val_percent], axis=1)

        

        # Rename the columns

        missing_val_table_ren_columns = missing_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing values in descending order, 

        # ignoring the colums with no missing values.

        missing_val_table_ren_columns = missing_val_table_ren_columns[

            missing_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(2)

        

        # Print some summary information

        print ("The DataFrame has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(missing_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        plt.subplots(figsize=(12,8))

        sns.barplot(y=missing_val_table_ren_columns['% of Total Values'],x=missing_val_table_ren_columns.index)

        plt.show()

        # Return the dataframe with missing information

        return missing_val_table_ren_columns

    

missing_values_table(city_day)
city_day1=city_day.copy()
city_day['BTX']=city_day['Benzene'] + city_day['Toluene'] + city_day['Xylene']

city_day.drop(['Benzene','Toluene','Xylene'],axis=1,inplace=True)

city_day[city_day['City']=='Gurugram'].tail(7)
city_day['Date'] = pd.to_datetime(city_day['Date'])

print('Data is available for the period {} to {} '.format(city_day['Date'].min(),city_day['Date'].max()))
cities_all=city_day['City'].value_counts()

print('We have data for the following cities:')

print(list(cities_all.index))
def city_wise_pollution(pollutant):

    i=city_day[[pollutant,'City']].groupby('City').mean().sort_values(by=pollutant,ascending=False).reset_index()

    return i[:10].style.background_gradient(cmap='PuBu')
from IPython.display import display_html

def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.render()

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
pm25=city_wise_pollution('PM2.5')

pm10=city_wise_pollution('PM10')

btx=city_wise_pollution('BTX')

so2=city_wise_pollution('SO2')

no2=city_wise_pollution('NO2')

co=city_wise_pollution('CO')

nh3=city_wise_pollution('NH3')

AQI=city_wise_pollution('AQI')



pollutants=['PM2.5','PM10','SO2','BTX','CO','NH3','NO2']



display_side_by_side(pm25,pm10,btx,so2,no2,co,nh3,AQI)
def barplot(df,pollutant):

    bar=df[[pollutant,'City']].groupby(['City']).mean().sort_values(by=pollutant,ascending=False).reset_index()

    ax,fig=plt.subplots(figsize=(16,6))

    sns.barplot(x='City',y=pollutant,data=bar,palette='viridis')

    plt.xlabel('City',fontsize=16)

    plt.ylabel(pollutant,fontsize=16)

    plt.xticks(rotation=45,horizontalalignment='center',fontsize=12)
barplot(city_day,'PM2.5')
barplot(city_day,'PM10')
barplot(city_day,'SO2')
barplot(city_day,'NO2')
barplot(city_day,'CO')
barplot(city_day,'NH3')
barplot(city_day,'AQI')
city_day.set_index('Date',inplace=True)  # only run this line once, second time it will give out error

pollutants=['PM2.5','PM10','SO2','BTX','CO','NH3','NO2']



axes=city_day[pollutants].plot(marker='.',figsize=(16,20),subplots=True,alpha=0.3)

for axes in axes:

    axes.set_xlabel('Years',fontsize=16)

    axes.set_ylabel('Concentration',fontsize=12)

plt.show()
def seasonality(df,val):

    df['year']=[d.year for d in df.Date]   

    df['months']=[d.strftime('%m') for d in df.Date]

    

    fig,axes=plt.subplots(1,2,figsize=(16,8))

    

    sns.boxplot(x='year',y=val, data=df,ax=axes[0])

    sns.lineplot(x='months',y=val,data=df.loc[~df.year.isin([2020]),:]) # not including 2020 as data is available

    axes[0].set_title('Yearly Boxplot',fontsize=14)                     # onlt till May

    axes[1].set_title('Monthly Lineplot',fontsize=14)

    plt.show()
city_day2=city_day.copy()  # df with date as index

city_day.reset_index(inplace=True)  # only run this line once, second time it will give out error

seasonality(city_day,'PM10')
seasonality(city_day,'PM2.5')
seasonality(city_day,'SO2')
seasonality(city_day,'NO2')
seasonality(city_day,'BTX')
seasonality(city_day,'O3')
seasonality(city_day,'AQI')
cities=['Ahmedabad','Bengaluru','Chennai','Delhi','Kolkata']



city_day_cities=city_day[city_day['Date'] >= '2019-01-01']

AQI_table=city_day_cities[city_day_cities['City'].isin(cities)]

AQI_table=AQI_table[['Date','City','AQI','AQI_Bucket']]

AQI_table
AQI_table_pivot=AQI_table.pivot(index='Date',columns='City',values='AQI')

AQI_table_pivot.head(7)
AQI_table_pivot.fillna(method='bfill',inplace=True)

AQI_table_pivot.describe()
fig=make_subplots(rows=5,cols=1,subplot_titles=('Ahmedabad','Bengaluru','Chennai','Delhi','Kolkata'))



fig.add_trace(go.Bar(x=AQI_table_pivot.index,y=AQI_table_pivot['Ahmedabad']

                     ,marker=dict(color=AQI_table_pivot['Ahmedabad']

                                  ,coloraxis="coloraxis")),1,1)



fig.add_trace(go.Bar(x=AQI_table_pivot.index,y=AQI_table_pivot['Bengaluru']

                     ,marker=dict(color=AQI_table_pivot['Bengaluru']

                                  ,coloraxis="coloraxis")),2,1)



fig.add_trace(go.Bar(x=AQI_table_pivot.index,y=AQI_table_pivot['Chennai']

                     ,marker=dict(color=AQI_table_pivot['Chennai']

                                  ,coloraxis="coloraxis")),3,1)



fig.add_trace(go.Bar(x=AQI_table_pivot.index,y=AQI_table_pivot['Delhi']

                     ,marker=dict(color=AQI_table_pivot['Delhi']

                                  ,coloraxis="coloraxis")),4,1)



fig.add_trace(go.Bar(x=AQI_table_pivot.index,y=AQI_table_pivot['Kolkata']

                     ,marker=dict(color=AQI_table_pivot['Kolkata']

                                  ,coloraxis="coloraxis")),5,1)



fig.update_layout(coloraxis=dict(colorscale='Temps'),showlegend=False,title_text="AQI Levels")



fig.update_layout( width=1000,height=1200,shapes=[dict(type= 'line',yref= 'paper'

                                                       ,y0= 0,y1= 1,xref= 'x',x0= '2020-03-25',x1= '2020-03-25')])



fig.show()
cities=['Ahmedabad','Bengaluru','Chennai','Delhi','Kolkata']



pollutants_2019=city_day[(city_day['Date'] >= '2019-01-01') & (city_day['Date'] <= '2019-05-01')]

pollutants_2019.fillna(method='bfill',inplace=True)

pollutants_2019.set_index('Date',inplace=True)



pollutants_2020=city_day[(city_day['Date'] >= '2020-01-01') & (city_day['Date'] <= '2020-05-01')]

pollutants_2020.fillna(method='bfill',inplace=True)

pollutants_2020.set_index('Date',inplace=True)



pollutants_2019=pollutants_2019[pollutants_2019['City'].isin(cities)][['City','SO2','CO','PM2.5','NO']]

pollutants_2020=pollutants_2020[pollutants_2020['City'].isin(cities)][['City','SO2','CO','PM2.5','NO']]

def comparison(city):

    # Figure for 2019

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=pollutants_2019.index,y=pollutants_2019[pollutants_2019['City']==city]['SO2']

                             ,line=dict(dash='dash',color='blue'),name='SO2'))

    

    fig.add_trace(go.Scatter(x=pollutants_2019.index,y=pollutants_2019[pollutants_2019['City']==city]['CO']

                             ,line=dict(dash='solid',color='green'),name='CO'))

                  

    fig.add_trace(go.Scatter(x=pollutants_2019.index,y=pollutants_2019[pollutants_2019['City']==city]['PM2.5']

                             ,line=dict(dash='solid',color='rosybrown'),name='PM2.5'))

                  

    fig.add_trace(go.Scatter(x=pollutants_2019.index,y=pollutants_2019[pollutants_2019['City']==city]['NO']

                             ,line=dict(dash='dashdot',color='slategrey'),name='NO'))

                  

    fig.update_layout(title_text=city+' 2019 ',width=800,height=500)

    fig.show()

    

    # Figure for 2020

    

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=pollutants_2020.index,y=pollutants_2020[pollutants_2020['City']==city]['SO2']

                             ,line=dict(dash='dash',color='blue'),name='SO2'))

    

    fig.add_trace(go.Scatter(x=pollutants_2020.index,y=pollutants_2020[pollutants_2020['City']==city]['CO']

                             ,line=dict(dash='solid',color='green'),name='CO'))

                  

    fig.add_trace(go.Scatter(x=pollutants_2020.index,y=pollutants_2020[pollutants_2020['City']==city]['PM2.5']

                             ,line=dict(dash='solid',color='rosybrown'),name='PM2.5'))

                  

    fig.add_trace(go.Scatter(x=pollutants_2020.index,y=pollutants_2020[pollutants_2020['City']==city]['NO']

                             ,line=dict(dash='dashdot',color='slategrey'),name='NO'))

                  

    fig.update_layout(title_text=city+' 2020 ',width=800,height=500)

    fig.show()
comparison('Delhi')
comparison('Ahmedabad')
comparison('Chennai')
comparison('Kolkata')
comparison('Bengaluru')