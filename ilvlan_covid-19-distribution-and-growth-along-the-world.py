import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import plotly.graph_objects as go

import seaborn as sns
confirmed_cases = pd.read_csv('/kaggle/input/covid19-01222020-to-02272020/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')



deaths_cases = pd.read_csv('/kaggle/input/covid19-01222020-to-02272020/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')



recovered_cases = pd.read_csv('/kaggle/input/covid19-01222020-to-02272020/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')



confirmed_cases.head()
confirmed_cases.info()
confirmed_by_country = confirmed_cases.groupby('Country/Region')

deaths_by_country = deaths_cases.groupby('Country/Region')

recovered_by_country = recovered_cases.groupby('Country/Region')





aggr_confirmed_country = confirmed_by_country.apply(lambda df:df.iloc[:,4:].sum()).iloc[:,-1]

aggr_deaths_country = deaths_by_country.apply(lambda df:df.iloc[:,4:].sum()).iloc[:,-1]

aggr_recovered_country = recovered_by_country.apply(lambda df:df.iloc[:,4:].sum()).iloc[:,-1]



aggr_confirmed_country = aggr_confirmed_country.rename(index = 

                                 {'Mainland China':'China','North Macedonia':'Macedonia',

                                  'South Korea':'Korea, Republic of','UK':'United Kingdom',

                                  'Macau':'Maca','US':'United States of America'})



aggr_deaths_country = aggr_deaths_country.rename(index = 

                                 {'Mainland China':'China','North Macedonia':'Macedonia',

                                  'South Korea':'Korea, Republic of','UK':'United Kingdom',

                                  'Macau':'Maca','US':'United States of America'})



aggr_recovered_country = aggr_recovered_country.rename(index = 

                                 {'Mainland China':'China','North Macedonia':'Macedonia',

                                  'South Korea':'Korea, Republic of','UK':'United Kingdom',

                                  'Macau':'Maca','US':'United States of America'})
from io import StringIO

import requests

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}

url = 'https://pkgstore.datahub.io/JohnSnowLabs/country-and-continent-codes-list/country-and-continent-codes-list-csv_csv/data/b7876b7f496677669644f3d1069d3121/country-and-continent-codes-list-csv_csv.csv'

s=requests.get(url, headers= headers).text

country_info = pd.read_csv(StringIO(s))



def get_tree_letter_code(s):

    country = country_info[country_info['Country_Name'].str.contains(s)]

    if country.empty:

        return ''

    else:

        return country['Three_Letter_Country_Code'].values[0]



aggr_country_df = pd.DataFrame()

aggr_country_df['country'] = aggr_confirmed_country.index

aggr_country_df['confirmed'] = aggr_confirmed_country.values

aggr_country_df['deaths'] = aggr_deaths_country.values

aggr_country_df['recovered'] = aggr_recovered_country.values

aggr_country_df['Three_Letter_Country_Code'] = aggr_country_df['country'].apply(get_tree_letter_code)

aggr_country_df.head()
def draw_world_map(zindex,cscale,title,color_title):

    fig = go.Figure(data=go.Choropleth(

        locations = aggr_country_df['Three_Letter_Country_Code'],

        z = zindex,

        text = aggr_country_df['country'],

        colorscale = cscale,

        autocolorscale=False,

        reversescale=False,

        marker_line_color='darkgray',

        marker_line_width=0.5,

        colorbar_title = color_title,

    ))



    fig.update_layout(

        title_text=title,

        geo=dict(

            showframe=False,

            showcoastlines=False,

            projection_type='equirectangular'

        )

    )

    fig.show()

draw_world_map(aggr_country_df['confirmed'],'Reds','Confirmed covid19','number of confirmed')


aggr_country_df['log-confirmed'] = np.log(aggr_country_df['confirmed']+ 1e-7 )

aggr_country_df['deaths-ratio'] = aggr_country_df['deaths']/aggr_country_df['confirmed']

aggr_country_df['recovered-ratio'] = aggr_country_df['recovered']/aggr_country_df['confirmed']

aggr_country_df.head()

draw_world_map(aggr_country_df['log-confirmed'],'Blues','logarithmic distribution of confirmed cases','log of num of confirmed')
draw_world_map(aggr_country_df['deaths-ratio'],'Reds','distribution of deaths-ratio cases','ratio')
draw_world_map(aggr_country_df['recovered-ratio'],'Greens',' distribution of recovered-ratio cases','ration')


time_series_confirmed = confirmed_by_country.apply(lambda df:df.iloc[:,4:].sum())

time_series_deaths = deaths_by_country.apply(lambda df:df.iloc[:,4:].sum())

time_series_recovered = recovered_by_country.apply(lambda df:df.iloc[:,4:].sum())



time_series_deaths_ratio = time_series_deaths.divide((time_series_confirmed + 1e-10))

time_series_recovered_ratio = time_series_recovered.divide((time_series_confirmed + 1e-10))


def draw_growth_plot(country):

    

    test_df = pd.DataFrame()

    t1 = np.trim_zeros(time_series_confirmed.loc[country].values)

    t2 = time_series_deaths.loc[country].values[-t1.shape[0]:]

    t3 = time_series_recovered.loc[country].values[-t1.shape[0]:]



    test_df['day'] = range(len(t1))

    test_df['conf'] = t1

    test_df['death'] =t2

    test_df['rec'] = t3



    test_df = pd.melt(test_df, id_vars=['day'],value_vars=['conf', 'death','rec'])

    test_df

    fig, ax1 = plt.subplots(figsize=(10, 10))

    sns.barplot(x='day', y='value', hue='variable', data=test_df,ax = ax1)

draw_growth_plot('Iran')
draw_growth_plot('Mainland China')



draw_growth_plot('South Korea')
draw_growth_plot('US')
cmap = sns.diverging_palette(220, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(10,10)) 

sns.heatmap(time_series_confirmed.transpose().corr(),cmap = cmap,linewidths=.5,vmax=1,ax=ax)
cmap = sns.diverging_palette(220, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(10,10)) 

sns.heatmap(time_series_deaths_ratio.transpose().corr(),cmap = cmap,linewidths=.5,ax=ax)
cmap = sns.diverging_palette(220, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(10,10)) 

sns.heatmap(time_series_recovered_ratio.transpose().corr(),cmap = cmap,linewidths=.5,ax=ax)
# time_series_confirmed 



def draw_comparison_growth(arr,lower_bound_treshold,title):

    temp = arr.transpose()

    temp = temp.set_index([pd.Index([i for i in range(len(temp.index))])])

    plt.figure(figsize=(10,10))

    aggr_count = temp.iloc[-1,:]

    for cnt in temp.columns:

        if cnt == 'Mainland China' or cnt =='Others' or aggr_count[cnt] < lower_bound_treshold :

            pass

        else:

            plt.plot(temp[cnt])

    plt.legend()

    plt.title(title)

    

draw_comparison_growth(time_series_confirmed,50,'Growth of confirmed cases which at least has 400 confirmed case')
draw_comparison_growth(time_series_deaths,5,'Growth of death cases which at least has 5 death case') 
# time_series_recovered

draw_comparison_growth(time_series_recovered,5,'Growth of recovered cases which at least has 5 recovered case') 