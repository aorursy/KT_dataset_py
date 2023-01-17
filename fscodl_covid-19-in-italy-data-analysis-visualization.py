# install calmap

! pip install calmap
# essential libraries

import json

import random

from urllib.request import urlopen

import requests

import lxml.html as lh



# storing and analysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

import calmap

import folium



# offline plotly visualization

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

init_notebook_mode(connected=True) 



# color pallette

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active cases - yellow

hos = '#d2691e' # hospitalized cases - brown



# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')



# gathering the geojson for Italian Regions

with urlopen('https://gist.githubusercontent.com/datajournalism-it/48e29e7c87dca7eb1d29/raw/2636aeef92ba0770a073424853f37690064eb0ea/regioni.geojson') as response:

    regions = json.load(response)



# gathering the geojson for Italian Provinces

with urlopen('https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_provinces.geojson') as response:

    provinces = json.load(response)
# importing datasets

full_table = pd.read_csv('../input/covid19-in-italy/covid19_italy_region.csv', 

                         names = ['SNo','Date', 'Country', 'RegionCode', 'Region', 'Lat', 'Long', 'HospitalizedNonICU', 'HospitalizedICU', 'Hospitalized', 'DomesticQuarantine', 'ConfirmedCurrent', 'ConfirmedNew', 'Recovered', 'Deaths', 'Confirmed', 'Swabs'], 

                         header = 0,

                         index_col = False)

full_table.replace("Emilia Romagna", "Emilia-Romagna", inplace = True)

full_table.head()
# dataframe info

# full_table.info()
# checking for missing value

# full_table.isna().sum()
#Scraper to create the dataframe with the population by region

url='https://www.tuttitalia.it/regioni/popolazione/'

page = requests.get(url)

doc = lh.fromstring(page.content)

tr_elements = doc.xpath('//tr')

[len(T) for T in tr_elements]



col=[]

i=0

for t in tr_elements[0]:

    i+=1

    name=t.text_content()

    col.append((name,[]))

    



for j in range(1,len(tr_elements)):

    T=tr_elements[j]

    

    if len(T)!=7:

        break

    

    i=0

    

    for t in T.iterchildren():

        data=t.text_content() 

        if i>0:

            try:

                data=int(data)

            except:

                pass

        col[i][1].append(data)

        i+=1

        

Dict = {title:column for (title,column) in col}

pop_reg = pd.DataFrame(Dict)

pop_reg = pop_reg.iloc[:,1:3]

pop_reg.columns = ['Region','Population']



for i in range(0, len(pop_reg['Population'])):

    pop_reg['Population'][i] = float(pop_reg['Population'][i].translate({ord('.'): None}))

pop_reg['Population'] = pop_reg['Population'].astype(float)
# importing datasets

full_table_prov = pd.read_csv('../input/covid19-in-italy/covid19_italy_province.csv', 

                         names = ['SNo','Date', 'Country', 'RegionCode', 'Region','ProvinceCode','Province','ProvinceAbbreviation', 'Lat', 'Long', 'Confirmed'], 

                         header = 0,

                         index_col = False)

full_table_prov.head()
# cases 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']
# latest

full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()



# latest condensed

full_latest_grouped = full_latest.groupby('Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()



# latest condensed | Regional visualization adjustment (Merging Trento and Bolzano into Trentino-Alto Adige)



#name = {

#    "P.A. Bolzano", "Trentino-Alto Adige",

#    "P.A. Trento", "Trentino-Alto Adige" }

full_latest_grouped2 = full_latest.copy()

full_latest_grouped2.replace("P.A. Bolzano", "Trentino-Alto Adige", inplace = True)

full_latest_grouped2.replace("P.A. Trento", "Trentino-Alto Adige", inplace = True)

full_latest_grouped2 = full_latest_grouped2.groupby(['Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()



#latest condensed with data about swabs (tests), quarantine and hospitalization

full_latest_grouped_moreinfo = full_latest.groupby('Region')['Confirmed', 'Deaths', 'Recovered', 'Active','Swabs','DomesticQuarantine','Hospitalized','HospitalizedNonICU', 'HospitalizedICU'].sum().reset_index()



#Regional visualization adjustment (Merging Trento and Bolzano into Trentino-Alto Adige)

flgm2 = full_latest.copy()

flgm2.replace("P.A. Bolzano", "Trentino-Alto Adige", inplace = True)

flgm2.replace("P.A. Trento", "Trentino-Alto Adige", inplace = True)

flgm2 = flgm2.groupby('Region')['Confirmed', 'Deaths', 'Recovered', 'Active','Swabs','DomesticQuarantine','Hospitalized','HospitalizedNonICU', 'HospitalizedICU'].sum().reset_index()



#full_latest_grouped2 = full_table

#full_latest_grouped2.replace("P.A. Bolzano", "Trentino-Alto Adige", inplace = True)

#full_latest_grouped2.replace("P.A. Trento", "Trentino-Alto Adige", inplace = True)

#full_latest_grouped2 = full_latest_grouped2.groupby(['Region','Date'], as_index=False)['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

#full_latest_grouped2 = full_latest_grouped2[full_latest_grouped2['Date']==max(full_latest_grouped2['Date'])].drop(columns='Date').reset_index(drop = True)

# latest

full_latest_prov = full_table_prov[full_table_prov['Date'] == max(full_table_prov['Date'])].reset_index()



# latest condensed

full_latest_grouped_prov = full_latest_prov.groupby('Province')['Confirmed'].sum().reset_index()
temp = full_table.groupby(['Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()

temp.style.background_gradient(cmap='Reds')
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)

temp.style.background_gradient(cmap='Pastel1')
tm = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])

fig = px.treemap(tm, path=["variable"], values="value", height=400, width=600,

                 color_discrete_sequence=[act, rec, dth])

fig.show()
temp_f = full_latest_grouped.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.style.background_gradient(cmap='Reds')
temp_flg = temp_f[temp_f['Deaths']>0][['Region', 'Deaths']]

temp_flg.sort_values('Deaths', ascending=False).reset_index(drop=True).style.background_gradient(cmap='Reds')
temp = temp_f[temp_f['Recovered']==0][['Region', 'Confirmed', 'Deaths', 'Recovered']]

temp.reset_index(drop=True).style.background_gradient(cmap='Reds')
temp = full_latest_grouped[full_latest_grouped['Confirmed']==

                          full_latest_grouped['Deaths']]

temp = temp[['Region', 'Confirmed', 'Deaths']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Reds')
temp = full_latest_grouped[full_latest_grouped['Confirmed']==

                          full_latest_grouped['Recovered']]

temp = temp[['Region', 'Confirmed', 'Recovered']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Greens')
temp = full_latest_grouped[full_latest_grouped['Confirmed']==

                          full_latest_grouped['Deaths']+

                          full_latest_grouped['Recovered']]

temp = temp[['Region', 'Confirmed', 'Deaths', 'Recovered']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Greens')
temp_f = full_latest_grouped_moreinfo.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)



temp_f.style.background_gradient(cmap='Reds')
temp_f = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active','Swabs','DomesticQuarantine','Hospitalized','HospitalizedNonICU', 'HospitalizedICU'].sum().reset_index()

temp_f = temp_f[temp_f['Date']==max(temp_f['Date'])].reset_index(drop=True)

temp_f.style.background_gradient(cmap='Pastel1')
# Italy



m = folium.Map(location=[41.8719, 12.5674], tiles='cartodbpositron',

               min_zoom=5, max_zoom=10, zoom_start=5)



for i in range(0, len(full_latest)):

    folium.Circle(

        location=[full_latest.iloc[i]['Lat'], full_latest.iloc[i]['Long']],

        color='crimson', 

        fill = True,

        fill_color='crimson',

        tooltip =   '<li><bold>Country : '+str(full_latest.iloc[i]['Country'])+

                    '<li><bold>Region : '+str(full_latest.iloc[i]['Region'])+

                    '<li><bold>Confirmed : '+str(full_latest.iloc[i]['Confirmed'])+

                    '<li><bold>Deaths : '+str(full_latest.iloc[i]['Deaths'])+

                    '<li><bold>Recovered : '+str(full_latest.iloc[i]['Recovered']),

        radius=int(full_latest.iloc[i]['Confirmed'])**1).add_to(m)

m
#Making sure the properties from the geojson include the region name



print(full_latest_grouped["Region"][0])

print(regions["features"][3]["properties"])
#Confirmed

fig = go.Figure(go.Choroplethmapbox(geojson=regions, locations=full_latest_grouped2['Region'],

                                    featureidkey="properties.NOME_REG",

                                    z=full_latest_grouped2['Confirmed'], colorscale='matter', zmin=0, zmax=max(full_latest_grouped2['Confirmed']),

                                    marker_opacity=0.8, marker_line_width=0.1))

fig.update_layout(mapbox_style="carto-positron",

                  mapbox_zoom=4, mapbox_center = {"lat": 41.8719, "lon": 12.5674})

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

fig.update_traces(showscale=True)

fig.update_layout(title='Confirmed Cases by Region')

fig.show()
# Deaths



fig = go.Figure(go.Choroplethmapbox(geojson=regions, locations=full_latest_grouped2['Region'],

                                    featureidkey="properties.NOME_REG",

                                    z=full_latest_grouped2['Deaths'], colorscale='amp', zmin=0, zmax=max(full_latest_grouped2['Deaths']),

                                    marker_opacity=0.8, marker_line_width=0.1))

fig.update_layout(mapbox_style="carto-positron",

                  mapbox_zoom=4, mapbox_center = {"lat": 41.8719, "lon": 12.5674})

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

fig.update_traces(showscale=True)

fig.update_layout(title='Deaths by Region')

fig.show()
formated_gdf = full_table.groupby(['Date', 'Region'])['Lat','Long','Confirmed', 'Deaths', 'Recovered'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])

formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.5)



fig = px.scatter_mapbox(formated_gdf, lat="Lat", lon="Long",

                     color="Confirmed", size='size', hover_name="Region", hover_data=['Confirmed','Deaths'],

                     color_continuous_scale='matter',

                     range_color= [0, max(formated_gdf['Confirmed'])+2],

                     animation_frame="Date", 

                     title='Spread over time')

fig.update(layout_coloraxis_showscale=True)

fig.update_layout(mapbox_style="carto-positron",

                  mapbox_zoom=4, mapbox_center = {"lat": 41.8719, "lon": 12.5674})

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

fig.show()
#Confirmed

temp = full_latest_prov.groupby(['Province', 'ProvinceCode'])['Confirmed'].sum().reset_index()



fig = go.Figure(go.Choroplethmapbox(geojson=provinces, locations=temp['ProvinceCode'],

                                    featureidkey="properties.prov_istat_code_num",

                                    z=temp['Confirmed'], colorscale='matter', zmin=0, zmax=max(temp['Confirmed']),

                                    text = temp['Province'],

                                    hoverinfo = 'text+z',

                                    marker_opacity=0.8, marker_line_width=0.1))

fig.update_layout(mapbox_style="carto-positron",

                  mapbox_zoom=4, mapbox_center = {"lat": 41.8719, "lon": 12.5674})

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

fig.update_traces(showscale=True)

fig.update_layout(title='Confirmed Cases by Province')

fig.show()
temp = full_table.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case',

             title='Cases over time', color_discrete_sequence = [rec, dth, act])

fig.show()
temp = full_table.groupby('Date').sum().reset_index()



# adding two more columns

temp['No. of Deaths to 100 Confirmed Cases'] = round(temp['Deaths']/temp['Confirmed'], 3)*100

temp['No. of Recovered to 100 Confirmed Cases'] = round(temp['Recovered']/temp['Confirmed'], 3)*100

temp['No. of Hospitalized to 100 Confirmed Cases'] = round(temp['Hospitalized']/temp['Confirmed'], 3)*100



# temp['No. of Recovered to 1 Death Case'] = round(temp['Recovered']/temp['Deaths'], 3)



temp = temp.melt(id_vars='Date', value_vars=['No. of Deaths to 100 Confirmed Cases', 'No. of Recovered to 100 Confirmed Cases', 'No. of Hospitalized to 100 Confirmed Cases'], 

                 var_name='Ratio', value_name='Value')



fig = px.line(temp, x="Date", y="Value", color='Ratio', log_y=True, 

              title='Recovery, Mortality and Hospitalization Rate Over The Time', color_discrete_sequence=[dth, rec, hos],

              height=800)

fig.update_layout(legend_orientation='h', legend_title='')

fig.show()
reg_spread = full_table[full_table['Confirmed']!=0].groupby('Date')['Region'].unique().apply(len)

reg_spread = pd.DataFrame(reg_spread).reset_index()



fig = px.line(reg_spread, x='Date', y='Region',

              title='Number of Italian Regions to which COVID-19 spread over the time',

             color_discrete_sequence=[cnf,dth, rec])

#fig.update_traces(textposition='top center')

#fig.update_layout(uniformtext_minsize=5, uniformtext_mode='hide')

fig.show()
cl = full_latest.groupby('Region')['Confirmed', 'Deaths', 'Recovered'].sum()

cl = cl.reset_index().sort_values(by='Confirmed', ascending=False).reset_index(drop=True)

# cl.head().style.background_gradient(cmap='rainbow')



ncl = cl.copy()

ncl['Active'] = ncl['Confirmed'] - ncl['Deaths'] - ncl['Recovered']

ncl = ncl.melt(id_vars="Region", value_vars=['Active', 'Recovered', 'Deaths'])



fig = px.bar(ncl.sort_values(['variable', 'value']), 

             y="Region", x="value", color='variable', orientation='h', height=800,

             title='Number and state of Cases by Region', color_discrete_sequence=[act, dth, rec])

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.update_traces(opacity=0.6)

fig.show()
flg = full_latest_grouped_moreinfo

#flg.head()
fig = px.bar(flg.sort_values('Confirmed', ascending=False).head(5).sort_values('Confirmed', ascending=True), 

             x="Confirmed", y="Region", title='Confirmed Cases', text='Confirmed', orientation='h', 

             width=700, height=700, range_x = [0, max(flg['Confirmed'])+10000])

fig.update_traces(marker_color=cnf, opacity=0.6, textposition='outside')

fig.show()
fig = px.bar(flg.sort_values('Deaths', ascending=False).head(5).sort_values('Deaths', ascending=True), 

             x="Deaths", y="Region", title='Deaths', text='Deaths', orientation='h', 

             width=700, height=700, range_x = [0, max(flg['Deaths'])+5000])

fig.update_traces(marker_color=dth, opacity=0.6, textposition='outside')

fig.show()
fig = px.bar(flg.sort_values('Recovered', ascending=False).head(5).sort_values('Recovered', ascending=True), 

             x="Recovered", y="Region", title='Recovered', text='Recovered', orientation='h', 

             width=700, height=700, range_x = [0, max(flg['Recovered'])+10000])

fig.update_traces(marker_color=rec, opacity=0.6, textposition='outside')

fig.show()
fig = px.bar(flg.sort_values('Active', ascending=False).head(5).sort_values('Active', ascending=True), 

             x="Active", y="Region", title='Currently Active', text='Active', orientation='h', 

             width=700, height=700, range_x = [0, max(flg['Active'])+10000])

fig.update_traces(marker_color=act, opacity=0.6, textposition='outside')

fig.show()
# (Only regions with more than 500 case are considered)



flg['Mortality Rate'] = round((flg['Deaths']/flg['Confirmed'])*100, 2)

temp = flg[flg['Confirmed']>500]

temp = temp.sort_values('Mortality Rate', ascending=False)



fig = px.bar(temp.sort_values('Mortality Rate', ascending=False).head(5).sort_values('Mortality Rate', ascending=True), 

             x="Mortality Rate", y="Region", text='Mortality Rate', orientation='h', 

             width=700, height=600, range_x = [0, 20], title='Mortality Rate (No. of Deaths Per 100 Confirmed Case)')

fig.update_traces(marker_color=dth, opacity=0.6, textposition='outside')

fig.show()
fig = px.bar(flg.sort_values('Hospitalized', ascending=False).head(5).sort_values('Hospitalized', ascending=True), 

             x="Hospitalized", y="Region", title='Hospitalized', text='Hospitalized', orientation='h', 

             width=700, height=700, range_x = [0, max(flg['Hospitalized'])+2500])

fig.update_traces(marker_color=hos, opacity=0.6, textposition='outside')

fig.show()
flg['Hospitalization Rate'] = round((flg['Hospitalized']/flg['Confirmed'])*100, 2)

temp = flg[flg['Confirmed']>100]

temp = temp.sort_values('Mortality Rate', ascending=False)



fig = px.bar(temp.sort_values('Hospitalization Rate', ascending=False).head(5).sort_values('Hospitalization Rate', ascending=True), 

             x="Hospitalization Rate", y="Region", text='Hospitalization Rate', orientation='h', 

             width=700, height=600, range_x = [0, 100], title='Hospitalization Rate (No. of Hospitalized Per 100 Confirmed Case)')

fig.update_traces(marker_color=hos, opacity=0.6, textposition='outside')

fig.show()
fig = px.bar(flg.sort_values('DomesticQuarantine', ascending=False).head(5).sort_values('DomesticQuarantine', ascending=True), 

             x="DomesticQuarantine", y="Region", title='Domestic Quarantine', text='DomesticQuarantine', orientation='h', 

             width=700, height=700, range_x = [0, max(flg['DomesticQuarantine'])+5000])

fig.update_traces(marker_color=act, opacity=0.6, textposition='outside')

fig.show()
fig = px.bar(flg.sort_values('Swabs', ascending=False).head(5).sort_values('Swabs', ascending=True), 

             x="Swabs", y="Region", title='Swabs (tests)', text='Swabs', orientation='h', 

             width=700, height=700, range_x = [0, max(flg['Swabs'])+80000])

fig.update_traces(marker_color='purple', opacity=0.6, textposition='outside')

fig.show()
# merge dataframes

temp = pd.merge(full_latest_grouped2, pop_reg, how='left', right_on='Region', left_on='Region')

# print(temp[temp['Country Name'].isna()])

temp = temp[['Region', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Population']]

#temp.columns = ['Region', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Population']

    

# calculate Confirmed/Population

temp['Confirmed Per Million Inhabitants'] = round(temp['Confirmed']/temp['Population']*1000000, 2)



fig = px.bar(temp.head(20).sort_values('Confirmed Per Million Inhabitants', ascending=True), 

             x='Confirmed Per Million Inhabitants', y='Region', orientation='h', 

             width=1000, height=700, text='Confirmed Per Million Inhabitants', title='Confirmed cases Per Million Inhabitants',

             range_x = [0, max(temp['Confirmed Per Million Inhabitants'])+2500])

fig.update_traces(textposition='outside', marker_color=dth, opacity=0.7)

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
# merge dataframes (flgm2 is full_latest_grouped2 but with Trento and Bolzano merged into Trentino Alto Adige)

temp = pd.merge(flgm2, pop_reg, how='left', right_on='Region', left_on='Region')

# print(temp[temp['Country Name'].isna()])

temp = temp[['Region', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Population','HospitalizedICU','HospitalizedNonICU','Hospitalized']]

#temp.columns = ['Region', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Population']

    

# calculate Hospitalized/Population

temp['Hospitalized not in ICU Per Million Inhabitants'] = round(temp['HospitalizedNonICU']/temp['Population']*1000000, 2)

temp['Hospitalized in ICU Per Million Inhabitants'] = round(temp['HospitalizedICU']/temp['Population']*1000000, 2)

# countries with population greater that 1 million only

#temp = temp[temp['Population']>1000000].sort_values('Confirmed Per Million People', ascending=False).reset_index(drop=True)

# temp.head()





# temp['No. of Recovered to 1 Death Case'] = round(temp['Recovered']/temp['Deaths'], 3)

temp = temp.melt(id_vars='Region', value_vars=['Hospitalized not in ICU Per Million Inhabitants', 'Hospitalized in ICU Per Million Inhabitants'], 

                 var_name='Hospitalized cases per Million Inhabitants', value_name='Value')



fig = px.bar(temp.sort_values('Value', ascending=True),

             x="Value", y="Region", color='Hospitalized cases per Million Inhabitants', orientation='h', 

             title='Hospitalized Cases Per Million Inhabitants',

             color_discrete_sequence=['saddlebrown', 'sandybrown'],

             height=1000,

             text='Value',

             range_x = [0, max(temp['Value'])+500]

             )

fig.update_traces(textposition='outside', opacity=0.7)

fig.update_layout(barmode='stack')

fig.update_layout(uniformtext_minsize=11, uniformtext_mode='hide')

fig.update_layout(legend_orientation="h", legend_title='')

fig.show()

temp = full_table.groupby('Date')['ConfirmedNew'].sum().reset_index()

temp['Date'] = pd.to_datetime(temp['Date'])

temp['Date'] = temp['Date'].dt.strftime('%d %b')



fig = px.bar(temp, x="ConfirmedNew", y="Date", orientation='h', height=800, 

             text = 'ConfirmedNew',

             title='N. of New Confirmed cases in Italy for each day',

             range_x = [0, max(temp['ConfirmedNew'])+1000])

fig.update_layout(xaxis_title='Newly Confirmed Cases')

fig.update_traces(marker_color=act, opacity=0.6, textposition='outside')

fig.show()
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

#temp['Date'] = pd.to_datetime(temp['Date'])

#temp['Date'] = temp['Date'].dt.strftime('%d %b')

temp = temp.reset_index().sort_values(by='Confirmed', ascending=True).reset_index(drop=True)



ntemp = temp.copy()

ntemp['Active'] = ntemp['Confirmed'] - ntemp['Deaths'] - ntemp['Recovered']

ntemp = ntemp.melt(id_vars="Date", value_vars=['Active', 'Recovered', 'Deaths'])

ntemp['Date'] = pd.to_datetime(ntemp['Date'])

ntemp['Date'] = ntemp['Date'].dt.strftime('%d %b')



fig = px.bar(ntemp.sort_values(['variable', 'value']), 

             y="Date", x="value", color='variable', orientation='h', height=1200,

             title='Total N. of Active, Deceased and Recovered cases in Italy', color_discrete_sequence=[act, dth, rec])

fig.update_yaxes(categoryorder = "total ascending")

fig.update_layout(xaxis_title='Value')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.update_traces(opacity=0.6)

fig.show()
temp = full_table.groupby(['Region', 'Date'])['Confirmed', 'Deaths', 'Recovered'].sum()

temp = temp.reset_index()

temp['Date'] = pd.to_datetime(temp['Date'])

temp['Date'] = temp['Date'].dt.strftime('%d %b')



fig = px.bar(temp, x="Confirmed", y="Date", color='Region', orientation='h', height=1200,

             title='Total N. of Confirmed cases')

fig.show()
temp = full_table.groupby(['Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()



mask = temp['Region'] != temp['Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



temp['Date'] = pd.to_datetime(temp['Date'])

temp['Date'] = temp['Date'].dt.strftime('%d %b')



fig = px.bar(temp, x="Confirmed", y="Date", color='Region', orientation='h', height = 1200,

             title='New Confirmed cases every day')

fig.show()
temp = full_table.groupby(['Region', 'Date'])['Confirmed', 'Deaths', 'Recovered'].sum()

temp = temp.reset_index()

temp['Date'] = pd.to_datetime(temp['Date'])

temp['Date'] = temp['Date'].dt.strftime('%d %b')



fig = px.bar(temp, x="Deaths", y="Date", color='Region', orientation='h', height=1200,

             title='Total N. of Deaths')

fig.show()
temp = full_table.groupby(['Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()



mask = temp['Region'] != temp['Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



temp['Date'] = pd.to_datetime(temp['Date'])

temp['Date'] = temp['Date'].dt.strftime('%d %b')



fig = px.bar(temp, x="Deaths", y="Date", color='Region', orientation='h', height=1200,

             title='New Deaths every day')

fig.show()
temp = full_table.groupby(['Date', 'Region'])['Confirmed'].sum().reset_index()

temp['Date'] = pd.to_datetime(temp['Date'])

temp['Date'] = temp['Date'].dt.strftime('%m/%d/%Y')

temp = temp.sort_values(by='Date')



fig = px.bar(temp, y='Region', x='Confirmed', color='Region', orientation='h',  

             title='Confirmed cases over time', animation_frame='Date', height=1000, 

             range_x=[0, max(temp['Confirmed']+5000)],

             text='Confirmed')

fig.update_traces(textposition='outside')

fig.update_layout(yaxis={'categoryorder':'total ascending'})

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
temp = full_table.groupby(['Date', 'Region'])['Confirmed'].sum().reset_index()

temp['Date'] = pd.to_datetime(temp['Date'])

temp['Date'] = temp['Date'].dt.strftime('%d %b')

px.line(temp, x="Date", y="Confirmed", color='Region', title='Cases Spread', height=600)
temp = full_latest_grouped

fig = px.scatter(temp, 

                 x='Confirmed', y='Deaths', color='Region',

                 text='Region', log_x=True, log_y=True, title='Deaths vs Confirmed')

fig.update_traces(textposition='top center')

fig.show()
fig = px.treemap(full_latest.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

                 path=["Region"], values="Confirmed", height=700,

                 title='Number of Confirmed Cases',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()



fig = px.treemap(full_latest.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 

                 path=["Region"], values="Deaths", height=700,

                 title='Number of Deaths reported',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()
# first date

# ----------

first_date = full_table[full_table['Confirmed']>0]

# converting Date to datetime

first_date['Date'] = pd.to_datetime(first_date['Date'])

first_date = first_date.groupby('Region')['Date'].agg(['min']).reset_index()

# first_date.head()



from datetime import timedelta  



# last date

# ---------

last_date = full_table

# converting Date to datetime

last_date['Date'] = pd.to_datetime(last_date['Date'])

last_date = full_table.groupby(['Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

last_date = last_date.sum().diff().reset_index()



mask = last_date['Region'] != last_date['Region'].shift(1)

last_date.loc[mask, 'Confirmed'] = np.nan

last_date.loc[mask, 'Deaths'] = np.nan

last_date.loc[mask, 'Recovered'] = np.nan



last_date = last_date[last_date['Confirmed']>0]

last_date = last_date.groupby('Region')['Date'].agg(['max']).reset_index()

# last_date.head()



# first_last

# ----------

first_last = pd.concat([first_date, last_date[['max']]], axis=1)



# added 1 more day, which will show the next day as the day on which last case appeared

first_last['max'] = first_last['max'] + timedelta(days=1)



# no. of days

first_last['Days'] = first_last['max'] - first_last['min']



# task column as country

first_last['Task'] = first_last['Region']



# rename columns

first_last.columns = ['Region', 'Start', 'Finish', 'Days', 'Task']



# sort by no. of days

first_last = first_last.sort_values('Days')

# first_last.head()



# visualization

# --------------



# produce random colors

clr = ["#"+''.join([random.choice('0123456789ABC') for j in range(6)]) for i in range(len(first_last))]



#plot

fig = ff.create_gantt(first_last, index_col='Region', colors=clr, show_colorbar=False, 

                      bar_width=0.2, showgrid_x=True, showgrid_y=True, height=500, 

                      title=('Gantt Chart'))

fig.show()
temp = full_table.groupby(['Date', 'Region'])['Confirmed'].sum()

temp = temp.reset_index().sort_values(by=['Date', 'Region'])



plt.style.use('seaborn')

g = sns.FacetGrid(temp, col="Region", hue="Region", 

                  sharey=False, col_wrap=4)

g = g.map(plt.plot, "Date", "Confirmed")

g.set_xticklabels(rotation=90)

plt.show()
temp = full_table.copy()



temp['LnConfirmed'] = np.log(temp['Confirmed'])

temp = temp.groupby(['Date', 'Region'])['LnConfirmed'].sum()

temp = temp.reset_index().sort_values(by=['Date', 'Region'])





plt.style.use('seaborn')

g = sns.FacetGrid(temp, col="Region", hue="Region", 

                  sharey=False, col_wrap=4)

g = g.map(plt.plot, "Date", "LnConfirmed")

g.set_xticklabels(rotation=90)

plt.show()
temp = full_table.groupby(['Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()



mask = temp['Region'] != temp['Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



plt.style.use('seaborn')

g = sns.FacetGrid(temp, col="Region", hue="Region", 

                  sharey=False, col_wrap=4)

g = g.map(sns.lineplot, "Date", "Confirmed")

g.set_xticklabels(rotation=90)

plt.show()
full_table['Date'] = pd.to_datetime(full_table['Date'])
temp = full_table.groupby('Date')['Confirmed'].sum()

temp = temp.diff()



plt.figure(figsize=(20, 5))

ax = calmap.yearplot(temp, fillcolor='white', cmap='Oranges', linewidth=0.5)
temp = full_table.groupby('Date')['Deaths'].sum()

temp = temp.diff()



plt.figure(figsize=(20, 5))

ax = calmap.yearplot(temp, fillcolor='white', cmap='Reds', linewidth=0.5)
spread = full_table[full_table['Confirmed']!=0].groupby('Date')

spread = spread['Region'].unique().apply(len).diff()



plt.figure(figsize=(20, 5))

ax = calmap.yearplot(spread, fillcolor='white', cmap='Greens', linewidth=0.5)