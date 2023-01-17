import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Load the data

df=pd.read_csv('/kaggle/input/aus-real-estate-sales-march-2019-to-april-2020/aus-property-sales-sep2018-april2020.csv')

df['date_sold'] = pd.to_datetime(df['date_sold'])

df.tail()
import matplotlib.image as image

from matplotlib.font_manager import FontProperties

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

import matplotlib.dates as mdates

from datetime import datetime

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import matplotlib as mpl

import matplotlib.patches as mpatches

from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

register_matplotlib_converters()

%matplotlib inline 

im = image.imread('/kaggle/input/logohtagmin/HtAG-logo.png')

fp = FontProperties(fname=r"/kaggle/input/faicons/Font Awesome 5 Free-Regular-400.otf") 

plt.imshow(im)
df = df[np.abs(df.price - df.price.mean()) <= (9.0 * df.price.std())] # Clean the outliers

plt.figure(figsize=(19,7))

plt.plot_date(df['date_sold'], df['price'], xdate=True, markersize=1)
import plotly.express as px

df_map=df[df['property_type']=='townhouse']

df_map=df_map[np.abs(df_map.price - df_map.price.mean()) <= (3.0 * df_map.price.std())] # filter within 3 STD for beter colour scale

fig = px.density_mapbox(df_map, lat='lat', lon='lon', z='price', radius=10,

                        center=dict(lat=df_map.lat.mean(), lon=df_map.lon.mean()), zoom=4,color_continuous_scale='Viridis',

                        mapbox_style="carto-positron", title="Townhouse sale density and price Sep 2018 - April 2020")



fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

fig.show()
def plot_setup(colors, im, df, city='AUS'):

    fig,ax= plt.subplots()

    fig.set_size_inches(21, 7)



    fig.suptitle(city+' Property Sales September 2018 - April 2020', fontsize=20)

    ax.set_xlabel('Date Sold')

    ax.set_ylabel('Price')

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    imagebox = OffsetImage(im, zoom=0.7)

    ab = AnnotationBbox(imagebox, (df["date_sold"].mean(), df["price"].max()/1.1), frameon = False)

    ax.add_artist(ab)



    patchList = []

    for key in colors:

            data_key = mpatches.Patch(color=colors[key], label=key)

            patchList.append(data_key)

    plt.legend(handles=patchList, loc='upper right')

    return fig, ax
# plot setup



colors = {'house':'green', 'unit':'red', 'townhouse':'blue'}

fig, ax=plot_setup(colors,im,df)

#plot the data

ax.scatter(x=df["date_sold"].values,

           y=df["price"].values, 

           c=df["property_type"].apply(lambda x: colors[x]),

           alpha=0.3, marker='o', s=5)





# setting x axis bounds

ax.set_xlim((df["date_sold"].min(), df["date_sold"].max()))

plt.show()
df['city_name']=df['city_name'].str.title()

cities=df['city_name'].unique()

cities=['Sydney', 'Melbourne','Brisbane', 'Adelaide', 'Perth',  'Canberra']

for city in cities:



    

    df_plot=df[df['city_name']==city]

    df_plot = df_plot[np.abs(df_plot.price - df_plot.price.mean()) <= (5.0 * df_plot.price.std())] #limit to 5.0 STD for better scale

    fig,ax=plot_setup(colors,im, df_plot, city)

    ax.scatter(x=df_plot["date_sold"].values,

           y=df_plot["price"].values, 

           c=df_plot["property_type"].apply(lambda x: colors[x]),

           alpha=0.3, marker='o', s=5)



    # setting x axis bounds

    ax.set_xlim((df_plot["date_sold"].min(), df_plot["date_sold"].max()))

    plt.show()

   

    
from scipy.ndimage.filters import gaussian_filter1d



property_types=df.property_type.unique()



for property_type in property_types:

    fig, ax = plt.subplots(figsize=(21,7))

    fig.suptitle(property_type.title()+' Weekly Median Price March 2019 - April 2020', fontsize=20)

    ax.set_ylabel('Median Price')

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    

    cities_=list(cities)  

    for city in cities:

        df_weekly=df[['date_sold', 'city_name', 'property_type', 'price']][df['property_type']==property_type].copy().set_index('date_sold').sort_index()

        

        #Remove series for cities with few weekly sales for the property type. 700 sales will give us roughly 12-15 weekly sales average for 2019-2020.

        if len(df_weekly[(df_weekly['city_name']==city) & (df_weekly.index>pd.to_datetime('2020-01-04'))])<300: 

            cities_.remove(city)

            continue



        df_temp=df_weekly[df_weekly['city_name']==city].resample('W').median().interpolate(method='linear', limit_direction='both')

        df_temp['price'] = gaussian_filter1d(df_temp['price'], sigma=1) #smooth the line

        #Cut off the front and last week as the data may not be complete for these periods

        df_temp=df_temp[2:]

        df_temp=df_temp[:-2]

        df_temp.plot(ax=ax, linewidth=3, grid=True)

    ax.legend(cities_, loc='upper left')

    ax.set_xlabel('Week')

    imagebox = OffsetImage(im, zoom=0.5)

    ab = AnnotationBbox(imagebox, (df_temp.index.mean(), ax.get_ylim()[1]/1.05), frameon = False)

    ax.add_artist(ab)

    
import seaborn as sns

freqs={'M': 'Monthly','Q':'Quaterly'}

for freq in freqs:

    for property_type in property_types:

        fig, ax = plt.subplots(figsize=(21,7))

        fig.suptitle(property_type.title()+' '+freqs[freq]+' Median Price September 2018 - May 2020', fontsize=20)

        ax.set_ylabel('Median Price')

        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))



        cities_=list(cities)  

        for city in cities:

            #if city !='Brisbane':

            #    cities_.remove(city)

            #    continue

            df_monthly=df[['date_sold', 'city_name', 'property_type', 'price']][df['property_type']==property_type].copy().set_index('date_sold')



            #Remove series for cities with few weekly sales for the property type. 700 sales will give us roughly 12-15 weekly sales average for 2019-2020.

            if len(df_monthly[(df_monthly['city_name']==city) & (df_monthly.index>pd.to_datetime('2020-01-04'))])<300: 

                #len(df_weekly[(df_weekly.index>pd.to_datetime('2020-01-04')) & (df_weekly['city_name']==city)])

                cities_.remove(city)

                continue



            df_temp=df_monthly[df_monthly['city_name']==city].resample(freq).median().interpolate(method='linear', limit_direction='both')

            df_temp['price'] = gaussian_filter1d(df_temp['price'], sigma=1) #smooth the line

            #df_temp=df_temp[:-1]

            df_temp.plot(ax=ax, linewidth=3, grid=True)

        ax.legend(cities_, loc='upper left')

        ax.set_xlabel('Quarter')

        imagebox = OffsetImage(im, zoom=0.5)

        ab = AnnotationBbox(imagebox, (df_temp.index[0]+ (df_temp.index[-1]-df_temp.index[0])/2, ax.get_ylim()[0]*1.15), frameon = False)

        ax.add_artist(ab)
import seaborn as sns

for freq in freqs:

    for property_type in property_types:

        for city in cities:

            fig, ax = plt.subplots(figsize=(21,7))

            ax.set_ylabel('Median Price')

            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

            fig.suptitle(city+' '+property_type.title()+'s '+freqs[freq]+' Median Price October 2018 - April 2020', fontsize=20)

            

            df_monthly=df[['date_sold', 'city_name', 'property_type', 'price']][df['property_type']==property_type].copy().sort_values(by='date_sold')

            df_monthly = df_monthly[np.abs(df_monthly.price - df_monthly.price.mean()) <= (7.0 * df_monthly.price.std())] #limit to 7.0 STD for better scale

            df_monthly['date_sold_month'] = df_monthly['date_sold'].dt.to_period(freq)

            ax=sns.boxplot(x= 'date_sold_month', y = 'price', data=df_monthly[df_monthly['city_name']==city])

            #ax=sns.swarmplot(x='date_sold_month', y='price', data=df_monthly[df_monthly['city_name']==city], color=".25")
import seaborn as sns



for city in cities:

    fig, ax = plt.subplots(figsize=(21,7))

    ax.set_ylabel('Median Price')

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    fig.suptitle(city+ ' Median Price First Two Quarters 2020', fontsize=20)

    

    df_monthly=df[['date_sold', 'city_name', 'property_type', 'price']][df['city_name']==city].copy().sort_values(by='date_sold')

    df_monthly = df_monthly[np.abs(df_monthly.price - df_monthly.price.mean()) <= (3.0 * df_monthly.price.std())] #limit to 7.0 STD for better scale

    df_monthly['Quarter'] = df_monthly['date_sold'].dt.to_period('Q')

    df_monthly=df_monthly[(df_monthly['Quarter']=='2020Q1') | (df_monthly['Quarter']=='2020Q2')]

    ax=sns.boxplot(x= 'Quarter', y = 'price', data=df_monthly , hue='property_type')

    

    imagebox = OffsetImage(im, zoom=0.5)

    ab = AnnotationBbox(imagebox, ((ax.get_xlim()[1]-ax.get_xlim()[0])/4, ax.get_ylim()[1]/1.1), frameon = False)

    ax.add_artist(ab)

    #ax=sns.swarmplot(x='date_sold_month', y='price', data=df_monthly[df_monthly['city_name']==city], color=".25")
import seaborn as sns



for city in cities:

    fig, ax = plt.subplots(figsize=(21,7))

    ax.set_ylabel('Median Price')

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    fig.suptitle(city+ ' Median Price April 2019 & 2020', fontsize=20)

    

    df_monthly=df[['date_sold', 'city_name', 'property_type', 'price']][df['city_name']==city].copy().sort_values(by='date_sold')

    df_monthly = df_monthly[np.abs(df_monthly.price - df_monthly.price.mean()) <= (3.0 * df_monthly.price.std())] #limit to 7.0 STD for better scale

    df_monthly['Month'] = df_monthly['date_sold'].dt.to_period('M')

    df_monthly=df_monthly[(df_monthly['Month']=='2020-04') | (df_monthly['Month']=='2019-04')]

    ax=sns.boxplot(x= 'Month', y = 'price', data=df_monthly , hue='property_type')

    

    imagebox = OffsetImage(im, zoom=0.5)

    ab = AnnotationBbox(imagebox, ((ax.get_xlim()[1]-ax.get_xlim()[0])/4, ax.get_ylim()[1]/1.1), frameon = False)

    ax.add_artist(ab)

    #ax=sns.swarmplot(x='date_sold_month', y='price', data=df_monthly[df_monthly['city_name']==city], color=".25")
for property_type in property_types:

    for city in cities:

        fig, ax = plt.subplots(figsize=(21,7))

        ax.set_ylabel('Median Price')

        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

        fig.suptitle(city+' '+property_type.title()+'s Median Price First Two Quarters 2020', fontsize=20)

        

        df_monthly=df[['date_sold', 'city_name', 'property_type', 'price']][df['property_type']==property_type].copy().sort_values(by='date_sold')

        df_monthly = df_monthly[np.abs(df_monthly.price - df_monthly.price.mean()) <= (5.0 * df_monthly.price.std())] #limit to 7.0 STD for better scale

        df_monthly['Quarter'] = df_monthly['date_sold'].dt.to_period('Q')

        df_monthly=df_monthly[(df_monthly['Quarter']=='2020Q1') | (df_monthly['Quarter']=='2020Q2')]

        ax=sns.boxplot(x= 'Quarter', y = 'price', data=df_monthly[df_monthly['city_name']==city])

        imagebox = OffsetImage(im, zoom=0.5)

        ab = AnnotationBbox(imagebox, ((ax.get_xlim()[1]-ax.get_xlim()[0])/4, ax.get_ylim()[1]/1.1), frameon = False)

        ax.add_artist(ab)

        #ax=sns.swarmplot(x='Quarter', y='price', data=df_monthly[df_monthly['city_name']==city], color=".25")
for property_type in property_types:

    for city in cities:

        fig, ax = plt.subplots(figsize=(21,7))

        ax.set_ylabel('Median Price')

        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

        fig.suptitle(city+' '+property_type.title()+'s April 2019 & 2020', fontsize=20)

        

        df_monthly=df[['date_sold', 'city_name', 'property_type', 'price']][df['property_type']==property_type].copy().sort_values(by='date_sold')

        df_monthly = df_monthly[np.abs(df_monthly.price - df_monthly.price.mean()) <= (7.0 * df_monthly.price.std())] #limit to 7.0 STD for better scale

        df_monthly['Month'] = df_monthly['date_sold'].dt.to_period('M')

        df_monthly=df_monthly[(df_monthly['Month']=='2019-04') | (df_monthly['Month']=='2020-04')]

        ax=sns.boxplot(x= 'Month', y = 'price', data=df_monthly[df_monthly['city_name']==city])

        ax=sns.swarmplot(x='Month', y='price', data=df_monthly[df_monthly['city_name']==city], color=".25")

        imagebox = OffsetImage(im, zoom=0.5)

        ab = AnnotationBbox(imagebox, ((ax.get_xlim()[1]-ax.get_xlim()[0])/4, ax.get_ylim()[1]/1.1), frameon = False)

        ax.add_artist(ab)
for property_type in property_types:

    for city in cities:

        fig, ax = plt.subplots(figsize=(21,7))

        ax.set_ylabel('Median Price')

        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

        fig.suptitle(city+' '+property_type.title()+'s March and April 2020', fontsize=20)

        

        df_monthly=df[['date_sold', 'city_name', 'property_type', 'price']][df['property_type']==property_type].copy().sort_values(by='date_sold')

        df_monthly = df_monthly[np.abs(df_monthly.price - df_monthly.price.mean()) <= (5.0 * df_monthly.price.std())] #limit to 7.0 STD for better scale

        df_monthly['Month'] = df_monthly['date_sold'].dt.to_period('M')

        df_monthly=df_monthly[(df_monthly['Month']=='2020-03') | (df_monthly['Month']=='2020-04')]

        ax=sns.boxplot(x= 'Month', y = 'price', data=df_monthly[df_monthly['city_name']==city])

        

        imagebox = OffsetImage(im, zoom=0.5)

        ab = AnnotationBbox(imagebox, ((ax.get_xlim()[1]-ax.get_xlim()[0])/4, ax.get_ylim()[1]/1.1), frameon = False)

        ax.add_artist(ab)

        #ax=sns.swarmplot(x='Quarter', y='price', data=df_monthly[df_monthly['city_name']==city], color=".25")
for property_type in property_types:

    for city in cities:

        fig, ax = plt.subplots(figsize=(21,7))

        ax.set_ylabel('Median Price')

        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

        fig.suptitle(city+' '+property_type.title()+'s March 2020 & April 2020', fontsize=20)

        

        df_monthly=df[['date_sold', 'city_name', 'property_type', 'price']][df['property_type']==property_type].copy().sort_values(by='date_sold')

        df_monthly = df_monthly[np.abs(df_monthly.price - df_monthly.price.mean()) <= (7.0 * df_monthly.price.std())] #limit to 7.0 STD for better scale

        df_monthly['Month'] = df_monthly['date_sold'].dt.to_period('M')

        df_monthly=df_monthly[(df_monthly['Month']=='2020-03') | (df_monthly['Month']=='2020-04')]

        ax=sns.boxplot(x= 'Month', y = 'price', data=df_monthly[df_monthly['city_name']==city])

        ax=sns.swarmplot(x='Month', y='price', data=df_monthly[df_monthly['city_name']==city], color=".25")

        imagebox = OffsetImage(im, zoom=0.5)

        ab = AnnotationBbox(imagebox, ((ax.get_xlim()[1]-ax.get_xlim()[0])/4, ax.get_ylim()[1]/1.1), frameon = False)

        ax.add_artist(ab)
for freq in freqs:

    for property_type in property_types:

        fig, ax = plt.subplots(figsize=(21,7))

        fig.suptitle(property_type.title()+' '+freqs[freq]+' Median Price % Change September 2018 - April 2020', fontsize=20)

        ax.set_ylabel('Median Price Change %')

        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))



        cities_=list(cities)  

        for city in cities:

            df_monthly=df[['date_sold', 'city_name', 'property_type', 'price']][df['property_type']==property_type].copy().set_index('date_sold')



            #Remove series for cities with few sales for the month of april

            if len(df_monthly[(df_monthly['city_name']==city) & (df_monthly.index>pd.to_datetime('2020-01-04'))])<500: 

                cities_.remove(city)

                continue



            df_temp=df_monthly[df_monthly['city_name']==city].resample(freq).median().interpolate(method='linear', limit_direction='both').pct_change(periods=1,fill_method='ffill')*100

            if freq=='M':

                df_temp['price'] = gaussian_filter1d(df_temp['price'], sigma=1) #smooth the line

            df_temp=df_temp.dropna()

            #df_temp=df_temp[:-1]

            df_temp.plot(ax=ax, linewidth=3, grid=True)

        ax.legend(cities_, loc='upper left')

        ax.set_xlabel('Quarter')

        imagebox = OffsetImage(im, zoom=0.5)

        ab = AnnotationBbox(imagebox, (df_temp.index[-2], ax.get_ylim()[1]/1.2), frameon = False)

        ax.add_artist(ab)

        plt.axhline(0, color='black', linewidth=5)

        ml = MultipleLocator(1)

        ax.xaxis.set_minor_locator(ml)

        ax.xaxis.grid(which="minor", color='k',  linewidth=0.1)

style = dict(size=10, color='#F1F1F1', weight='bold')



for property_type in property_types:

    

    fig, ax = plt.subplots(figsize=(21,7))

    fig.suptitle('Weekly '+property_type.title()+' Sales By City, September 2018 - April 2020', fontsize=20)

    

    df_weekly_count=df[df['property_type']==property_type]

    df_weekly_count=df_weekly_count[['date_sold','city_name', 'property_type']].set_index('date_sold').groupby(['city_name']).resample('W').count().sort_values(by='property_type')



    df_plot=df_weekly_count[['property_type']].unstack(0)

    df_plot.index = [ts.strftime('%d-%m-%Y') for ts in df_plot.index]

    df_plot.columns=df_plot.columns.droplevel()

    

    df_plot.columns.name='City'

    

    #cols=[ 'Sydney', 'Melbourne',  'Brisbane', 'Adelaide', 'Perth', 'Canberra']



    df_plot=df_plot[cities]

    

    df_plot[:-1][1:].plot(kind='bar', stacked=True, width=1, ax=ax, grid=False) #,colormap='tab10'

    plt.gcf().autofmt_xdate()

    

    for label in ax.xaxis.get_ticklabels()[::2]:

        label.set_visible(False)



        

    ax.text(16, ax.get_ylim()[1]/10, 'Christmas 2018', ha='center', **style)

    ax.text(69, ax.get_ylim()[1]/10, 'Christmas 2019', ha='center', **style)

    ax.text(32, ax.get_ylim()[1]/10, 'Easter 2019', ha='center', **style)

    ax.text(81, ax.get_ylim()[1]/10, 'Easter 2020', ha='center', **style)

    ax.annotate('COVID-19 Restrictions', xy=(80, ax.get_ylim()[1]/3), xytext=(55, ax.get_ylim()[1]/1.5), arrowprops=dict(facecolor='#F4F4F4', shrink=0.05), size=18, bbox=dict(fc='#F4F4F4', ec='none', pad=3),)

   

    

    ax.set_xlabel('Week')

    ax.set_ylabel('Sales')

    ax.yaxis.set_major_locator(MultipleLocator(200))

    imagebox = OffsetImage(im, zoom=0.7)

    ab = AnnotationBbox(imagebox, (ax.get_xlim()[1]/1.1, ax.get_ylim()[1]/1.1), frameon = False)

    ax.add_artist(ab)

    ax.legend(loc='upper left')



cities=[ 'Brisbane', 'Adelaide', 'Perth', 'Canberra']



for property_type in property_types:

    

    fig, ax = plt.subplots(figsize=(21,7))

    fig.suptitle('Weekly '+property_type.title()+' Sales By City, September 2018 - April 2020', fontsize=20)

    

    df_weekly_count=df[df['property_type']==property_type]

    df_weekly_count=df_weekly_count[['date_sold','city_name', 'property_type']].set_index('date_sold').groupby(['city_name']).resample('W').count().sort_values(by='property_type')



    df_plot=df_weekly_count[['property_type']].unstack(0)

    df_plot.index = [ts.strftime('%d-%m-%Y') for ts in df_plot.index]

    df_plot.columns=df_plot.columns.droplevel()

    

    df_plot.columns.name='City'

    





    df_plot=df_plot[cities]

    

    df_plot[:-1][1:].plot(kind='bar', stacked=True, width=1, ax=ax, grid=False) #,colormap='tab10'

    plt.gcf().autofmt_xdate()

    

    for label in ax.xaxis.get_ticklabels()[::2]:

        label.set_visible(False)

        

    ax.text(16, ax.get_ylim()[1]/10, 'Christmas 2018', ha='center', **style)

    ax.text(69, ax.get_ylim()[1]/10, 'Christmas 2019', ha='center', **style)

    ax.text(32, ax.get_ylim()[1]/10, 'Easter 2019', ha='center', **style)

    ax.text(81, ax.get_ylim()[1]/10, 'Easter 2020', ha='center', **style)

    ax.annotate('COVID-19 Restrictions', xy=(80, ax.get_ylim()[1]/3), xytext=(55, ax.get_ylim()[1]/1.5), arrowprops=dict(facecolor='#F4F4F4', shrink=0.05), size=18, bbox=dict(fc='#F4F4F4', ec='none', pad=3),)

    ax.set_xlabel('Week')

    ax.set_ylabel('Sales')

    ax.yaxis.set_major_locator(MultipleLocator(200))

    imagebox = OffsetImage(im, zoom=0.7)

    ab = AnnotationBbox(imagebox, (ax.get_xlim()[1]/1.1, ax.get_ylim()[1]/1.1), frameon = False)

    ax.add_artist(ab)

    ax.legend(loc='upper left')
cities=['Sydney', 'Melbourne','Brisbane', 'Adelaide', 'Perth',  'Canberra']





for property_type in property_types:

    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21,7), sharey=True)

    fig.suptitle('Weekly '+property_type.title()+' Sales By City, First 4 months 2019 and 2020 ', fontsize=20)

    

    df_weekly_count=df[df['property_type']==property_type]

    df_weekly_count=df_weekly_count[['date_sold','city_name', 'property_type']].set_index('date_sold').groupby(['city_name']).resample('W').count().sort_values(by='property_type')



    df_plot=df_weekly_count[['property_type']].unstack(0)

    

    

    df_plot_2019=df_plot[(df_plot.index>'2019-01-01') & (df_plot.index<'2019-05-15')]

    df_plot_2020=df_plot[df_plot.index>'2020-01-01']

    

    df_plot_2019.index = [ts.strftime('%d-%m-%Y') for ts in df_plot_2019.index]

    df_plot_2019.columns=df_plot_2019.columns.droplevel()

    df_plot_2019.columns.name='City'

    

    df_plot_2020.index = [ts.strftime('%d-%m-%Y') for ts in df_plot_2020.index]

    df_plot_2020.columns=df_plot_2020.columns.droplevel()

    df_plot_2020.columns.name='City'

    





    df_plot_2019=df_plot_2019[cities]

    df_plot_2020=df_plot_2020[cities]

    

    df_plot_2019[1:].plot(kind='bar', stacked=True, width=1, ax=ax1, grid=False) #,colormap='tab10'

    df_plot_2020[1:].plot(kind='bar', stacked=True, width=1, ax=ax2, grid=False) #,colormap='tab10'

    

    ax1.text(13, ax1.get_ylim()[1]/12, 'Easter 2019', ha='left', **style)

    ax2.text(12, ax2.get_ylim()[1]/12, 'Easter 2020', ha='left', **style)

    ax2.annotate('COVID-19 Restrictions', xy=(10, ax2.get_ylim()[1]/3), xytext=(3, ax2.get_ylim()[1]/1.5), arrowprops=dict(facecolor='#F4F4F4', shrink=0.05), size=18, bbox=dict(fc='#F4F4F4', ec='none', pad=3),)

    ax2.yaxis.set_tick_params(labelbottom=True)

    plt.gcf().autofmt_xdate()

    imagebox = OffsetImage(im, zoom=0.5)

    ab = AnnotationBbox(imagebox, (1, ax1.get_ylim()[1]/1.1), frameon = False)

    ax1.add_artist(ab)

    ax1.set_title('2019')

    ax1.legend(loc='upper right')

    ab = AnnotationBbox(imagebox, (1, ax2.get_ylim()[1]/1.1), frameon = False)

    ax2.add_artist(ab)

    ax2.set_title('2020')

    ax2.legend(loc='upper right')

    
