import datetime as dt

dt_string = dt.datetime.now().strftime("%d/%m/%Y")

print(f"Kernel last updated: {dt_string}")
import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

import folium

from folium.plugins import HeatMap, HeatMapWithTime

%matplotlib inline
print(os.listdir('/kaggle/input'))

DATA_FOLDER = "/kaggle/input/coronavirus-2019ncov"

print(os.listdir(DATA_FOLDER))

GEO_DATA = "/kaggle/input/china-regions-map"

print(os.listdir(GEO_DATA))

WD_GEO_DATA = '/kaggle/input/python-folio-country-boundaries'

print(os.listdir(WD_GEO_DATA))
data_df = pd.read_csv(os.path.join(DATA_FOLDER, "covid-19-all.csv"))

cn_geo_data = os.path.join(GEO_DATA, "china.json")

wd_geo_data = os.path.join(WD_GEO_DATA, "world-countries.json")
print(f"Rows: {data_df.shape[0]}, Columns: {data_df.shape[1]}")
data_df.head()
data_df.tail()
for column in data_df.columns:

    print(f"{column}:{data_df[column].dtype}")
print(f"Date - unique values: {data_df['Date'].nunique()} ({min(data_df['Date'])} - {max(data_df['Date'])})")
data_df['Date'] = pd.to_datetime(data_df['Date'])
for column in data_df.columns:

    print(f"{column}:{data_df[column].dtype}")
print(f"Date - unique values: {data_df['Date'].nunique()} ({min(data_df['Date'])} - {max(data_df['Date'])})")
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(data_df)
print(f"Countries/Regions:{data_df['Country/Region'].nunique()}")

print(f"Province/State:{data_df['Province/State'].nunique()}")
ch_map = folium.Map(location=[35, 100], zoom_start=4)



folium.GeoJson(

    cn_geo_data,

    name='geojson'

).add_to(ch_map)



folium.LayerControl().add_to(ch_map)



ch_map
wd_map = folium.Map(location=[0,0], zoom_start=2)



folium.GeoJson(

    wd_geo_data,

    name='geojson'

).add_to(wd_map)



folium.LayerControl().add_to(wd_map)



wd_map
data_cn = data_df.loc[data_df['Country/Region']=="China"]

data_cn = data_cn.sort_values(by = ['Province/State','Date'], ascending=False)
filtered_data_last = data_cn.drop_duplicates(subset = ['Province/State'],keep='first')
def plot_count(feature, value, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    df = df.sort_values([value], ascending=False).reset_index(drop=True)

    g = sns.barplot(df[feature][0:30], df[value][0:30], palette='Set3')

    g.set_title("Number of {} - first 30 by number".format(title))

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

    plt.show()    
plot_count('Province/State', 'Confirmed', 'Confirmed cases (last updated)', filtered_data_last, size=4)
def plot_time_variation(df, y='Confirmed', hue='Province/State', size=1, is_log=False):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    g = sns.lineplot(x="Date", y=y, hue=hue, data=df)

    plt.xticks(rotation=90)

    plt.title(f'{y} cases grouped by {hue}')

    if(is_log):

        ax.set(yscale="log")

    ax.grid(color='black', linestyle='dotted', linewidth=0.75)

    plt.show()  
plot_time_variation(data_cn, size=4, is_log=True)
plot_time_variation(data_cn, y='Recovered', size=4, is_log=True)
def plot_time_variation_all(df, title='Mainland China', size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))

    g = sns.lineplot(x="Date", y='Confirmed', data=df, color='blue', label='Confirmed')

    g = sns.lineplot(x="Date", y='Recovered', data=df, color='green', label='Recovered')

    g = sns.lineplot(x="Date", y='Deaths', data=df, color = 'red', label = 'Deaths')

    plt.xlabel('Date')

    plt.ylabel(f'Total {title} cases')

    plt.xticks(rotation=90)

    plt.title(f'Total {title} cases')

    ax.grid(color='black', linestyle='dotted', linewidth=0.75)

    plt.show()  

data_cn = data_df.loc[data_df['Country/Region']=="China"]

data_cn = data_cn.sort_values(by = ['Province/State','Date'], ascending=False)

data_cn_agg = data_cn.groupby(['Date']).sum().reset_index()

plot_time_variation_all(data_cn_agg, size=3)
filtered_data_last = filtered_data_last.reset_index()

plot_time_variation(data_cn.loc[~(data_cn['Province/State']=='Hubei')],y='Recovered', size=4, is_log=True)
m = folium.Map(location=[30, 100], zoom_start=4)



folium.Choropleth(

    geo_data=cn_geo_data,

    name='Confirmed cases - regions',

    key_on='feature.properties.name',

    fill_color='YlGn',

    fill_opacity=0.05,

    line_opacity=0.3,

).add_to(m)



radius_min = 2

radius_max = 40

weight = 1

fill_opacity = 0.2



_color_conf = 'red'

group0 = folium.FeatureGroup(name='<span style=\\"color: #EFEFE8FF;\\">Confirmed cases</span>')

for i in range(len(filtered_data_last)):

    lat = filtered_data_last.loc[i, 'Latitude']

    lon = filtered_data_last.loc[i, 'Longitude']

    province = filtered_data_last.loc[i, 'Province/State']

    recovered = filtered_data_last.loc[i, 'Recovered']

    death = filtered_data_last.loc[i, 'Deaths']



    _radius_conf = np.sqrt(filtered_data_last.loc[i, 'Confirmed'])

    if _radius_conf < radius_min:

        _radius_conf = radius_min



    if _radius_conf > radius_max:

        _radius_conf = radius_max



    _popup_conf = str(province) + '\n(Confirmed='+str(filtered_data_last.loc[i, 'Confirmed']) + '\nDeaths=' + str(death) + '\nRecovered=' + str(recovered) + ')'

    folium.CircleMarker(location = [lat,lon], 

                        radius = _radius_conf, 

                        popup = _popup_conf, 

                        color = _color_conf, 

                        fill_opacity = fill_opacity,

                        weight = weight, 

                        fill = True, 

                        fillColor = _color_conf).add_to(group0)



group0.add_to(m)

folium.LayerControl().add_to(m)

m
def plot_time_variation_mortality(df, title='Mainland China', size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))

    g = sns.lineplot(x="Date", y='Mortality (D/C)', data=df, color='blue', label='Mortality (Deaths / Confirmed)')

    g = sns.lineplot(x="Date", y='Mortality (D/R)', data=df, color='green', label='Mortality (Death / Recovered)')

    plt.xlabel('Date')

    ax.set_yscale('log')

    plt.ylabel(f'Mortality {title} [%]')

    plt.xticks(rotation=90)

    plt.title(f'Mortality percent {title}\nCalculated as Deaths/Confirmed cases and as Death / Recovered cases')

    ax.grid(color='black', linestyle='dashed', linewidth=1)

    plt.show()  
data_cn_agg['Mortality (D/C)'] = data_cn_agg['Deaths'] / data_cn_agg['Confirmed'] * 100

data_cn_agg['Mortality (D/R)'] = data_cn_agg['Deaths'] / data_cn_agg['Recovered'] * 100

plot_time_variation_mortality(data_cn_agg, size = 5)
data_wd = data_df.copy()

data_wd = pd.DataFrame(data_wd.groupby(['Country/Region', 'Date'])['Confirmed', 'Recovered', 'Deaths'].sum()).reset_index()

data_wd.columns = ['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths' ]

data_wd = data_wd.sort_values(by = ['Country','Date'], ascending=False)
data_ct = data_wd.sort_values(by = ['Country','Date'], ascending=False)

filtered_data_ct_last = data_wd.drop_duplicates(subset = ['Country'], keep='first')

data_ct_agg = data_ct.groupby(['Date']).sum().reset_index()
plot_count('Country', 'Confirmed', 'Confirmed cases - all World', filtered_data_ct_last, size=4)
plot_count('Country', 'Recovered', 'Recovered - all World', filtered_data_ct_last, size=4)
plot_count('Country', 'Deaths', 'Deaths - all World', filtered_data_ct_last, size=4)
plot_time_variation_all(data_ct_agg, 'All World', size=4)
data_select_agg = data_ct.groupby(['Country', 'Date']).sum().reset_index()
data_select_agg['Active'] = data_select_agg['Confirmed'] - data_select_agg['Deaths'] - data_select_agg['Recovered']
def plot_time_variation_countries(df, countries, case_type='Confirmed', size=3, is_log=False):

    f, ax = plt.subplots(1,1, figsize=(4*size,4*size))

    for country in countries:

        df_ = df[(df['Country']==country) & (df['Date'] > '2020-02-01')] 

        g = sns.lineplot(x="Date", y=case_type, data=df_,  label=country)  

        ax.text(max(df_['Date']), (df_.loc[df_['Date']==max(df_['Date']), case_type]), str(country))

    plt.xlabel('Date')

    plt.ylabel(f'Total  {case_type} cases')

    plt.title(f'Total {case_type} cases')

    plt.xticks(rotation=90)

    if(is_log):

        ax.set(yscale="log")

    ax.grid(color='black', linestyle='dotted', linewidth=0.75)

    plt.show()  
countries = ['China', 'Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France', 'Netherlands', 'Japan', 'Romania']

plot_time_variation_countries(data_select_agg, countries, size=4)
countries = ['China', 'Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France', 'Netherlands', 'Japan', 'Romania']

plot_time_variation_countries(data_select_agg, countries,case_type = 'Deaths', size=4)
countries = ['China', 'Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France',\

             'Netherlands', 'Austria', 'Japan', 'Romania']

plot_time_variation_countries(data_select_agg, countries, case_type = 'Active', size=4)
countries = ['China','Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France',\

             'Netherlands', 'Austria', 'Japan', 'Romania']

plot_time_variation_countries(data_select_agg, countries, size=4, is_log=True)
countries = ['China','Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France',

             'Netherlands', 'Austria', 'Japan', 'Romania']

plot_time_variation_countries(data_select_agg, countries,case_type = 'Recovered', size=4, is_log=True)
countries = ['China','Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France',\

             'Netherlands', 'Austria', 'Japan', 'Romania']

plot_time_variation_countries(data_select_agg, countries,case_type = 'Deaths', size=4, is_log=True)
countries = ['China','Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France',\

             'Netherlands', 'Austria', 'Japan', 'Romania']

plot_time_variation_countries(data_select_agg, countries,case_type = 'Active', size=4, is_log=True)
countries = ['China', 'Iran',  'Switzerland', 'South Korea', 'Austria']

plot_time_variation_countries(data_select_agg, countries,case_type = 'Active', size=4, is_log=True)
descent_active_case_countries = []

for country in data_select_agg.Country.unique():

    df = data_select_agg.loc[data_select_agg.Country == country]

    max_active = max(df['Active'])

    last_active = df.loc[df['Date']==max(df['Date']), 'Active'].values[0]

    if((last_active < 0.67 * max_active) & (max_active > 500)):

        print(f"country: {country} max active cases: {max_active} last day active cases: {last_active}")

        descent_active_case_countries.append(country)

print(descent_active_case_countries)
plot_time_variation_countries(data_select_agg, descent_active_case_countries,case_type = 'Active', size=4, is_log=True)
def plot_time_variation_countries_group_of_features(df, countries, features,title):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(5,4,figsize=(18, 24))



    for country in countries:

        plt.subplot(5,4,i + 1)

        df_ = df[(df['Country']==country) & (df['Date'] > '2020-02-01')] 

        df_['date'] = df_['Date'].apply(lambda x: x.timetuple().tm_yday)

        for feature in features:

            g = sns.lineplot(x="date", y=feature, data=df_,  label=feature)

        plt.title(f'{country}') 

        plt.xlabel('')

        i = i + 1

    fig.suptitle(title)

    plt.show()  
countries = ['Austria', 'Azerbaijan', 'China', 'Croatia', 

             'Denmark', 'Germany', 'Iceland', 'Iran', 

             'Malaysia', 'New Zealand',  'South Korea', 'Switzerland', 

             'Italy', 'Spain', 'Portugal', 'France',

             'Japan', 'Serbia', 'Greece', 'Romania']

features = ['Recovered', 'Active']

plot_time_variation_countries_group_of_features(data_select_agg, countries, features, 'Recovered (cumulative) vs. Active - cases vs. day of year')
data_ps = data_df.sort_values(by = ['Province/State','Date'], ascending=False)

filtered_data_ps = data_ps.drop_duplicates(subset = ['Province/State'],keep='first').reset_index()



data_cr = data_df.sort_values(by = ['Country/Region','Date'], ascending=False)

filtered_data_cr = data_cr.drop_duplicates(subset = ['Country/Region'],keep='first').reset_index()



filtered_data_cr = filtered_data_cr.loc[~filtered_data_cr.Latitude.isna()]

filtered_data_cr = filtered_data_cr.loc[~filtered_data_cr.Longitude.isna()]

filtered_data = pd.concat([filtered_data_cr, filtered_data_ps], axis=0).reset_index()
m = folium.Map(location=[0,0], zoom_start=2)

max_val = max(filtered_data['Confirmed'])

HeatMap(data=filtered_data[['Latitude', 'Longitude', 'Confirmed']],\

        radius=15, max_zoom=12).add_to(m)

m
m = folium.Map(location=[0,0], zoom_start=2)

HeatMap(data=filtered_data[['Latitude', 'Longitude', 'Recovered']].fillna(0),\

        radius=15, max_zoom=12).add_to(m)

m
m = folium.Map(location=[0,0], zoom_start=2)

HeatMap(data=filtered_data[['Latitude', 'Longitude', 'Deaths']].fillna(0),\

        radius=17, max_zoom=12).add_to(m)

m
data_all_wd = pd.DataFrame(data_df.groupby(['Country/Region', 'Date'])['Confirmed',  'Recovered', 'Deaths'].sum()).reset_index()

data_all_wd.columns = ['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths' ]

data_all_wd = data_all_wd.sort_values(by = ['Country','Date'], ascending=False)

filtered_all_wd_data_last = data_all_wd.drop_duplicates(subset = ['Country'],keep='first')

filtered_all_wd_data_last.loc[filtered_all_wd_data_last['Country']=='Mainland China', 'Country'] = 'China'
import plotly.express as px

mindate = min(data_df['Date'])

maxdate = max(data_df['Date'])

print(f"Date min/max: {mindate}, {maxdate}")

data_wd_df = data_df.copy()

data_wd_df['Days'] = data_wd_df['Date'].apply(lambda x: (x - dt.datetime.strptime('2020-01-21', "%Y-%m-%d")).days)

data_wd_df.loc[data_wd_df.Confirmed.isna(), 'Confirmed'] = 0

data_wd_df['Size'] = np.round(5 * np.sqrt(data_wd_df['Confirmed']),0)

max_confirmed = max(data_wd_df['Confirmed'])

min_confirmed = min(data_wd_df['Confirmed'])

hover_text = []

for index, row in data_wd_df.iterrows():

    hover_text.append(('Date: {}<br>'+

                       'Country/Region: {}<br>'+

                       'Province/State: {}<br>'+

                      'Confirmed: {}<br>'+

                      'Recovered: {}<br>'+

                      'Deaths: {}<br>').format(row['Date'], 

                                            row['Country/Region'],

                                            row['Province/State'],

                                            row['Confirmed'],

                                            row['Recovered'],

                                            row['Deaths']))

data_wd_df['hover_text'] = hover_text

fig = px.scatter_geo(data_wd_df, lon='Longitude', lat='Latitude', color="Confirmed",

                     hover_name="hover_text", size="Size",

                     animation_frame="Days",

                     projection="natural earth",

                     range_color =[min_confirmed,max_confirmed],

                    width=700,height=525)

fig.show()
data_ct_agg = data_all_wd.groupby(['Date']).sum().reset_index()



data_ct_agg['Mortality (D/C)'] = data_ct_agg['Deaths'] / data_ct_agg['Confirmed'] * 100

data_ct_agg['Mortality (D/R)'] = data_ct_agg['Deaths'] / data_ct_agg['Recovered'] * 100

plot_time_variation_mortality(data_ct_agg, title = ' - All world', size = 3)
data_all_wd = pd.DataFrame(data_df.groupby(['Country/Region', 'Date'])['Confirmed',  'Recovered', 'Deaths'].sum()).reset_index()

data_all_wd.columns = ['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths' ]

data_all_wd = data_all_wd.sort_values(by = ['Country','Date'], ascending=False)

data_italy = data_all_wd[data_all_wd['Country']=='Italy']

data_it_agg = data_italy.groupby(['Date']).sum().reset_index()



data_it_agg['Mortality (D/C)'] = data_it_agg['Deaths'] / data_it_agg['Confirmed'] * 100

data_it_agg['Mortality (D/R)'] = data_it_agg['Deaths'] / data_it_agg['Recovered'] * 100



plot_time_variation_mortality(data_it_agg, title = ' - Italy', size = 3)
data_iran = data_all_wd[data_all_wd['Country']=='Iran']

data_ir_agg = data_iran.groupby(['Date']).sum().reset_index()



data_ir_agg['Mortality (D/C)'] = data_ir_agg['Deaths'] / data_ir_agg['Confirmed'] * 100

data_ir_agg['Mortality (D/R)'] = data_ir_agg['Deaths'] / data_ir_agg['Recovered'] * 100



plot_time_variation_mortality(data_ir_agg, title = ' - Iran', size = 3)
data_sk = data_all_wd[data_all_wd['Country']=='South Korea']

data_sk_agg = data_sk.groupby(['Date']).sum().reset_index()



data_sk_agg['Mortality (D/C)'] = data_sk_agg['Deaths'] / data_sk_agg['Confirmed'] * 100

data_sk_agg['Mortality (D/R)'] = data_sk_agg['Deaths'] / data_sk_agg['Recovered'] * 100



plot_time_variation_mortality(data_sk_agg, title = ' - South Korea', size = 3)
data_sp = data_all_wd[data_all_wd['Country']=='Spain']

data_sp_agg = data_sp.groupby(['Date']).sum().reset_index()



data_sp_agg['Mortality (D/C)'] = data_sp_agg['Deaths'] / data_sp_agg['Confirmed'] * 100

data_sp_agg['Mortality (D/R)'] = data_sp_agg['Deaths'] / data_sp_agg['Recovered'] * 100



plot_time_variation_mortality(data_sp_agg, title = ' - Spain', size = 3)
data_de = data_all_wd[data_all_wd['Country']=='Germany']

data_de_agg = data_de.groupby(['Date']).sum().reset_index()



data_de_agg['Mortality (D/C)'] = data_de_agg['Deaths'] / data_de_agg['Confirmed'] * 100

data_de_agg['Mortality (D/R)'] = data_de_agg['Deaths'] / data_de_agg['Recovered'] * 100



plot_time_variation_mortality(data_ir_agg, title = ' - Germany', size = 3)
data_us = data_all_wd[data_all_wd['Country']=='US']

data_us_agg = data_us.groupby(['Date']).sum().reset_index()



data_us_agg['Mortality (D/C)'] = data_us_agg['Deaths'] / data_us_agg['Confirmed'] * 100

data_us_agg['Mortality (D/R)'] = data_us_agg['Deaths'] / data_us_agg['Recovered'] * 100



plot_time_variation_mortality(data_us_agg, title = ' - US', size = 3)
data_fr = data_all_wd[data_all_wd['Country']=='France']

data_fr_agg = data_fr.groupby(['Date']).sum().reset_index()



data_fr_agg['Mortality (D/C)'] = data_fr_agg['Deaths'] / data_fr_agg['Confirmed'] * 100

data_fr_agg['Mortality (D/R)'] = data_fr_agg['Deaths'] / data_fr_agg['Recovered'] * 100



plot_time_variation_mortality(data_fr_agg, title = ' - France', size = 3)
data_uk = data_all_wd[data_all_wd['Country']=='United Kingdom']

data_uk_agg = data_uk.groupby(['Date']).sum().reset_index()



data_uk_agg['Mortality (D/C)'] = data_uk_agg['Deaths'] / data_uk_agg['Confirmed'] * 100

data_uk_agg['Mortality (D/R)'] = data_uk_agg['Deaths'] / data_uk_agg['Recovered'] * 100



plot_time_variation_mortality(data_uk_agg, title = ' - UK', size = 3)
data_ro = data_all_wd[data_all_wd['Country']=='Romania']

data_ro_agg = data_ro.groupby(['Date']).sum().reset_index()



data_ro_agg['Mortality (D/C)'] = data_ro_agg['Deaths'] / data_ro_agg['Confirmed'] * 100

data_ro_agg['Mortality (D/R)'] = data_ro_agg['Deaths'] / data_ro_agg['Recovered'] * 100



plot_time_variation_mortality(data_ro_agg, title = ' - Romania', size = 3)
def plot_time_variation_mortality_countries(df, countries):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(4,4,figsize=(18, 16))



    for country in countries:

        plt.subplot(4,4,i + 1)

        df_ = df.loc[(df['Country']==country) & (df['Date'] > '2020-02-01')] 

        df_['date'] = df_['Date'].apply(lambda x: x.timetuple().tm_yday)

        df_['Mortality (D/C)'] = df_['Deaths'] / df_['Confirmed'] * 100

        df_['Mortality (D/R)'] = df_['Deaths'] / df_['Recovered'] * 100

        g = sns.lineplot(x="date", y='Mortality (D/C)', data=df_,  label='M (D/C)')

        g = sns.lineplot(x="date", y='Mortality (D/R)', data=df_,  label='M (D/R)')

        plt.title(f'{country}') 

        ax[i//4, i%4].set_yscale('log')

        plt.xlabel('')

        i = i + 1

    fig.suptitle('Mortality Deaths/Recovered (D/R) & Deaths/Confirmed (D/C)')

    plt.show()  
countries = ['Austria', 'Azerbaijan', 'China', 'Croatia', 

             'Denmark', 'Germany', 'Iceland', 'Iran', 

             'Malaysia', 'New Zealand',  'South Korea', 'Switzerland', 

             'Japan', 'Italy', 'US', 'Romania']

plot_time_variation_mortality_countries(data_select_agg, countries)
def plot_time_variation_mortality_countries(df, countries, title='Mainland China', size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4*size))

    colors = ['red', 'blue', 'green', 'magenta', 'lightgreen', 'black']

    for i, country in enumerate(countries):

        dc_df = df.loc[df.Country == country].copy()

        dc_df_agg = dc_df.groupby(['Date']).sum().reset_index()

        dc_df_agg['Mortality (D/C)'] = dc_df_agg['Deaths'] / dc_df_agg['Confirmed'] * 100

        dc_df_agg['Mortality (D/R)'] = dc_df_agg['Deaths'] / dc_df_agg['Recovered'] * 100

        g = sns.lineplot(x="Date", y='Mortality (D/C)', data=dc_df_agg, linestyle='-', color=colors[i], label=f'Mortality (Deaths/Confirmed) - {country}')

        ax.text(max(dc_df_agg['Date']), (dc_df_agg['Mortality (D/C)'].tail(1)), str(country))

        g = sns.scatterplot(x="Date", y='Mortality (D/R)', data=dc_df_agg, linestyle='-.', color=colors[i],label=f'Mortality (Deaths/Recovered) - {country}')

        ax.text(max(dc_df_agg['Date']), (dc_df_agg['Mortality (D/R)'].tail(1)), str(country))

    plt.xlabel('Date')

    ax.set_yscale('log')

    plt.ylabel(f'Mortality {title} [%]')

    plt.xticks(rotation=90)

    plt.title(f'Mortality percent {title}\nCalculated as Deaths/Confirmed cases and as Death / Recovered cases')

    ax.grid(color='black', linestyle='dashed', linewidth=1)

    plt.show()  
countries = ['US', 'Italy', 'China', 'South Korea', 'Japan', 'Romania']

plot_time_variation_mortality_countries(data_all_wd, countries, '- selection of World Countries',5)