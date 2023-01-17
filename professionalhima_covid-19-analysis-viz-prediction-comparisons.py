# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns

import datetime



from plotnine import *





import plotly.express as px

import folium



# color pallette



c = '#393e46'

d = '#ff2e63'

r = '#30e3ca'

i = '#f8b400'

cdr = [c, d, r] # grey - red - blue

idr = [i, d, r] # yellow - red - blue
!ls ../input/corona-virus-report
# importing datasets

full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 

                         parse_dates=['Date'])

full_table.head()
# dataframe info

full_table.info()
# checking for missing value

full_table.isna().sum()
#full_table[full_table['Country/Region']=='Others']['Province/State'].unique()

full_table['Country/Region'] = np.where(full_table['Country/Region']=='Others',full_table['Province/State'],full_table['Country/Region'])

full_table['Country/Region'].unique()
# still infected = confirmed - deaths - recovered

full_table['Infected'] = full_table['Confirmed'] - full_table['Deaths'] -full_table['Recovered']



# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values with NA

full_table[['Province/State']] = full_table[['Province/State']].fillna('NA')

full_table[['Confirmed', 'Deaths', 'Recovered', 'Infected']] = full_table[['Confirmed', 'Deaths', 'Recovered', 'Infected']].fillna(0)
# complete dataset 

# complete = full_table.copy()



# cases in the Diamond Princess cruise ship

ship = full_table[full_table['Country/Region']=='Diamond Princess cruise ship']



# full table

full_table = full_table

china = full_table[full_table['Country/Region']=='China']

row = full_table[full_table['Country/Region']!='China']



# latest

full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()

china_latest = full_latest[full_latest['Country/Region']=='China']

row_latest = full_latest[full_latest['Country/Region']!='China']



# condensed

full_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Infected'].sum().reset_index()

china_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Infected'].sum().reset_index()

row_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Infected'].sum().reset_index()
#No. of countries affected

full_grouped['Country/Region'].nunique()
#Total Cases

full_grouped[['Confirmed', 'Deaths', 'Recovered', 'Infected']].sum().sum()
full_grouped[['Confirmed', 'Deaths', 'Recovered', 'Infected']].sum()
#Donut chart

#colors

colors = ['#66b3ff','#99ff99','#ffcc99']

labels = ['Deaths', 'Recovered', 'Pending Recovery']

fig1, ax1 = plt.subplots()

sizes = full_grouped[['Deaths', 'Recovered', 'Infected']].sum()

ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)

#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.show()
pd.options.display.float_format = '{:,.2f}'.format

# highest confirmed cases

full_grouped.sort_values('Confirmed', ascending= False)[:10].style.background_gradient(cmap='seismic')
#confirmed cases distribution

#full_table.sort_values('Deaths', ascending=False)

max(full_latest['Date'])
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Infected'].sum()

temp = temp.reset_index()

temp = temp.sort_values('Date', ascending=False).reset_index(drop=True)
full_table[full_table['Date'] == max(full_table['Date'])]['Deaths'].sum()
import matplotlib.dates as mdates

from matplotlib.dates import DateFormatter

fig,ax = plt.subplots(figsize=(16, 8))

style = dict(size=20, color='red')

plt.plot('Date','Confirmed',data=temp,color='cyan',linewidth=5,linestyle='--')

plt.plot('Date','Deaths',data=temp,color='red',linewidth=5,linestyle='-.')

plt.plot('Date','Recovered',data=temp,color='green',linewidth=5,linestyle='-')

plt.plot('Date','Infected',data=temp,color='maroon',linewidth=5,linestyle=':')

ax.text('2020-03-08', 9000 ,full_table[full_table['Date'] == max(full_table['Date'])]['Deaths'].sum(),**style)

plt.legend(['Confirmed', 'Deaths', 'Recovered', 'Infected'], loc='best', fontsize=15)

plt.xticks(temp['Date'],rotation=90, size=15)

plt.xlabel('Date', size=20)

plt.ylabel('# of Cases', size=20)

date_form = DateFormatter("%m-%d")

ax.xaxis.set_major_formatter(date_form)

#plt.xlim(min(temp['Date']),max(temp['Date'])+ datetime.timedelta(days=5))

plt.tight_layout()

plt.show()
temp_f = full_grouped[['Country/Region', 'Confirmed', 'Deaths', 'Recovered','Infected']]

temp_f = temp_f.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f
India = temp_f[temp_f['Country/Region']=="India"]

India.head()
USA = temp_f[temp_f['Country/Region']=="US"]

USA.head()
import calendar

import datetime

d = max(full_latest['Date'])

plt.figure(figsize=(16,8))

ax=sns.barplot(y='Country/Region',x='Confirmed',data = temp_f[1:11],palette = 'autumn')

for p in ax.patches:

    width = p.get_width()

    plt.text(250+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:,}'.format(width),

             ha='center', va='center')

    

plt.text(5000,7, "As of March" +" " +str(d.day) , fontsize = 20, color='Red')

plt.text(5000,7.8, "total ROW confirmed cases " +"-- " + str(row_grouped['Confirmed'].sum()), fontsize = 20, color='Red')
china_cnf = china_grouped['Confirmed'].sum()

row_cnf = row_grouped['Confirmed'].sum()

t_cnf = china_cnf + row_cnf

label = ['China','ROW']

size = [china_cnf,row_cnf]

explode = (0.1, 0)

colors = ['Red', 'yellowgreen']

plt.pie(size, explode=explode, labels=label, colors=colors,autopct='%1.1f%%', startangle=40)



plt.axis('equal')

plt.show()
df = pd.melt(temp_f.head(10), id_vars="Country/Region", var_name="case types", value_name="count")

df = df.sort_values(by='count', ascending=False)

#df1 = df1.sort_values(by='count', ascending=True)

plt.figure(figsize=(20,10))

g=sns.barplot(x='Country/Region',y='count',data=df,hue='case types',palette="coolwarm")

ylabels = [format(y) + 'M' for y in g.get_yticks()/1000000]

g.set_yticklabels(ylabels)

plt.show()



#sns.catplot(x='Country/Region',data=df,y='count',hue='case types',kind='bar')
temp_flg = full_grouped[['Country/Region', 'Deaths']]

temp_flg = temp_flg.sort_values(by='Deaths', ascending=False)

temp_flg = temp_flg.reset_index(drop=True)

temp_flg = temp_flg[temp_flg['Deaths']>0][:10]

temp_flg.style.background_gradient(cmap='Reds')
temp = full_grouped[full_grouped['Recovered']==0]

temp = temp[['Country/Region', 'Confirmed', 'Deaths', 'Recovered']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Reds')
temp = full_grouped[full_grouped['Confirmed']==

                          full_grouped['Deaths']]

temp = temp[['Country/Region', 'Confirmed', 'Deaths']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Reds')
temp = full_grouped[full_grouped['Confirmed']==

                          full_grouped['Recovered']]

temp = temp[['Country/Region', 'Confirmed', 'Recovered']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Greens')
temp = row_latest_grouped[row_latest_grouped['Confirmed']==

                          row_latest_grouped['Deaths']+

                          row_latest_grouped['Recovered']]

temp = temp[['Country/Region', 'Confirmed', 'Deaths', 'Recovered']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Greens')
temp_f = china_grouped[['Province/State', 'Confirmed', 'Deaths', 'Recovered']]

temp_f = temp_f.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.head(10).style.background_gradient(cmap='Blues')
temp = china_grouped[china_grouped['Recovered']==0]

temp = temp[['Province/State', 'Confirmed', 'Deaths', 'Recovered']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Pastel1_r')
temp = china_grouped[china_grouped['Confirmed']==

                          china_grouped['Deaths']]

temp = temp[['Province/State', 'Confirmed', 'Deaths', 'Recovered']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Greens')
temp = china_grouped[china_grouped['Confirmed']==

                          china_latest_grouped['Recovered']]

temp = temp[['Province/State', 'Confirmed', 'Recovered']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Greens')
temp = china_grouped[china_grouped['Confirmed']==

                          china_grouped['Deaths']+

                          china_grouped['Recovered']]

temp = temp[['Province/State', 'Confirmed', 'Deaths', 'Recovered']]

temp = temp.sort_values('Confirmed', ascending=False)

temp = temp.reset_index(drop=True)

temp.style.background_gradient(cmap='Greens')
# World wide



m = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1)



for i in range(0, len(full_table)):

    folium.Circle(

        location=[full_table.iloc[i]['Lat'], full_table.iloc[i]['Long']],

        color='red', fill=True,

      fill_color='crimson',

        tooltip =   '<li><bold>Country : '+str(full_table.iloc[i]['Country/Region'])+

                    '<li><bold>Province : '+str(full_table.iloc[i]['Province/State'])+

                    '<li><bold>Confirmed : '+str(full_table.iloc[i]['Confirmed'])+

                    '<li><bold>Deaths : '+str(full_table.iloc[i]['Deaths'])+

                    '<li><bold>Recovered : '+str(full_table.iloc[i]['Recovered']),

        radius=int(full_table.iloc[i]['Confirmed'])*10).add_to(m)

m.show()
# China 



m = folium.Map(location=[30, 116], tiles='cartodbpositron',

               min_zoom=2, max_zoom=5, zoom_start=3)



for i in range(0, len(china)):

    folium.Circle(

        location=[china.iloc[i]['Lat'], china.iloc[i]['Long']],

        color='crimson', 

        tooltip =   '<li><bold>Country : '+str(china.iloc[i]['Country/Region'])+

                    '<li><bold>Province : '+str(china.iloc[i]['Province/State'])+

                    '<li><bold>Confirmed : '+str(china.iloc[i]['Confirmed'])+

                    '<li><bold>Deaths : '+str(china.iloc[i]['Deaths'])+

                    '<li><bold>Recovered : '+str(china.iloc[i]['Recovered']),

        radius=int(china.iloc[i]['Confirmed'])).add_to(m)

m
# China 



m = folium.Map(location=[30, 116], tiles='cartodbpositron',

               min_zoom=2, max_zoom=5, zoom_start=3)



for i in range(0, len(china)):

    folium.Marker(

        location=[china.iloc[i]['Lat'], china.iloc[i]['Long']],icon=folium.Icon(color='red'),

        color='crimson',

        tooltip =   '<li><bold>Country : '+str(china.iloc[i]['Country/Region'])+

                    '<li><bold>Province : '+str(china.iloc[i]['Province/State'])+

                    '<li><bold>Confirmed : '+str(china.iloc[i]['Confirmed'])+

                    '<li><bold>Deaths : '+str(china.iloc[i]['Deaths'])+

                    '<li><bold>Recovered : '+str(china.iloc[i]['Recovered'])).add_to(m)

m
china_latest
# Cases in the Diamond Princess Cruise Ship

temp = ship.sort_values(by='Date', ascending=False).head(1)

temp = temp[['Province/State', 'Confirmed', 'Deaths', 'Recovered']].reset_index(drop=True)

temp.style.background_gradient(cmap='rainbow')
temp = ship[ship['Date'] == max(ship['Date'])].reset_index()



m = folium.Map(location=[35.4437, 139.638], tiles='cartodbpositron',

               min_zoom=8, max_zoom=12, zoom_start=10)



folium.Circle(location=[temp.iloc[0]['Lat'], temp.iloc[0]['Long']],

        color='crimson', 

        tooltip =   '<li><bold>Ship : '+str(temp.iloc[0]['Province/State'])+

                    '<li><bold>Confirmed : '+str(temp.iloc[0]['Confirmed'])+

                    '<li><bold>Deaths : '+str(temp.iloc[0]['Deaths'])+

                    '<li><bold>Recovered : '+str(temp.iloc[0]['Recovered']),

        radius=int(temp.iloc[0]['Confirmed'])**1).add_to(m)

m
fig = px.choropleth(full_grouped, locations="Country/Region", 

                    locationmode='country names',  color = 'Confirmed',

                    hover_name="Country/Region", range_color=[1,1000], 

                    color_continuous_scale='Oranges', 

                    title='Countries with Confirmed Cases')

fig.update(layout_coloraxis_showscale=False)

fig.show()



# ------------------------------------------------------------------------



fig = px.choropleth(full_grouped[full_grouped['Deaths']>0], 

                    locations="Country/Region", locationmode='country names',

                    color="Deaths", hover_name="Country/Region", 

                    range_color=[1,1000], color_continuous_scale="YlOrRd",

                    title='Countries with Deaths Reported')

fig.update(layout_coloraxis_showscale=False)

fig.show()
formated_gdf = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf = formated_gdf[formated_gdf['Country/Region']!='China']

formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])

formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.5)



fig = px.scatter_geo(formated_gdf[formated_gdf['Country/Region']!='China'], 

                     locations="Country/Region", locationmode='country names', 

                     color="Confirmed", size='size', hover_name="Country/Region", 

                     range_color= [0, max(formated_gdf['Confirmed'])+2], 

                     projection="natural earth", animation_frame="Date", 

                     title='Spread outside China over time')

fig.update(layout_coloraxis_showscale=False)

fig.show()



# -----------------------------------------------------------------------------------



china_map = china.groupby(['Date', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 

                                                      'Lat', 'Long'].max()

china_map = china_map.reset_index()

china_map['size'] = china_map['Confirmed'].pow(0.5)

china_map['Date'] = pd.to_datetime(china_map['Date'])

china_map['Date'] = china_map['Date'].dt.strftime('%m/%d/%Y')

china_map.head()



fig = px.scatter_geo(china_map, lat='Lat', lon='Long', scope='asia',

                     color="size", size='size', hover_name='Province/State', 

                     hover_data=['Confirmed', 'Deaths', 'Recovered'],

                     projection="natural earth", animation_frame="Date", 

                     title='Spread in China over time')

fig.update(layout_coloraxis_showscale=False)

fig.show()
fig = px.bar(full_grouped[['Country/Region', 'Confirmed']].sort_values('Confirmed', ascending=False), 

             x="Confirmed", y="Country/Region", color='Country/Region', orientation='h',

             log_x=True, color_discrete_sequence = px.colors.qualitative.Bold, title='Confirmed Cases', width=900, height=1200)

fig.show()



temp = full_grouped[['Country/Region', 'Deaths']].sort_values('Deaths', ascending=False)

fig = px.bar(temp[temp['Deaths']>0], 

             x="Deaths", y="Country/Region", color='Country/Region', title='Deaths', orientation='h',

             log_x=True, color_discrete_sequence = px.colors.qualitative.Bold, width=900)

fig.show()
temp = full_table.groupby(['Country/Region', 'Date'])['Confirmed', 'Deaths', 'Recovered'].sum()

temp = temp.reset_index()

temp = temp[temp['Country/Region']!="China"]

# temp.head()



fig = px.bar(temp, x="Date", y="Confirmed", color='Country/Region', orientation='v', height=600,

             title='Confirmed', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()



fig = px.bar(temp, x="Date", y="Deaths", color='Country/Region', orientation='v', height=600,

             title='Deaths', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
temp.head()
# In China

temp = china.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

temp = temp.reset_index()

temp = temp.melt(id_vars="Date", 

                 value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(temp, x="Date", y="value", color='variable', 

             title='In China',

             color_discrete_sequence=idr)

fig.update_layout(barmode='group')

fig.show()



#-----------------------------------------------------------------------------



# ROW

temp = row.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

temp = temp.reset_index()

temp = temp.melt(id_vars="Date", 

                 value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(temp, x="Date", y="value", color='variable', 

             title='Outside China',

             color_discrete_sequence=idr)

fig.update_layout(barmode='group')

fig.show()
def from_china_or_not(row):

    if row['Country/Region']=='China':

        return 'From China'

    else:

        return 'Outside China'

    

temp = full_table.copy()

temp['Region'] = temp.apply(from_china_or_not, axis=1)

temp = temp.groupby(['Region', 'Date'])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()

mask = temp['Region'] != temp['Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



fig = px.bar(temp, x='Date', y='Confirmed', color='Region', barmode='group', 

             text='Confirmed', title='Confirmed', color_discrete_sequence= idr)

fig.update_traces(textposition='outside')

fig.show()



fig = px.bar(temp, x='Date', y='Deaths', color='Region', barmode='group', 

             text='Confirmed', title='Deaths', color_discrete_sequence= idr)

fig.update_traces(textposition='outside')

fig.update_traces(textangle=-90)

fig.show()
temp = full_table.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



fig = px.bar(temp, x="Date", y="Confirmed", color='Country/Region',

             title='Number of new cases everyday')

fig.show()



fig = px.bar(temp[temp['Country/Region']!='China'], x="Date", y="Confirmed", color='Country/Region',

             title='Number of new cases outside China everyday')

fig.show()



fig = px.bar(temp, x="Date", y="Deaths", color='Country/Region',

             title='Number of new death case reported outside China everyday')

fig.show()



fig = px.bar(temp[temp['Country/Region']!='China'], x="Date", y="Deaths", color='Country/Region',

             title='Number of new death case reported outside China everyday')

fig.show()
c_spread = china[china['Confirmed']!=0].groupby('Date')['Province/State'].unique().apply(len)

c_spread = pd.DataFrame(c_spread).reset_index()



fig = px.line(c_spread, x='Date', y='Province/State', 

              title='Number of Provinces/States/Regions of China to which COVID-19 spread over the time',

             color_discrete_sequence=cdr)

fig.show()



# ------------------------------------------------------------------------------------------



spread = full_table[full_table['Confirmed']!=0].groupby('Date')['Country/Region'].unique().apply(len)

spread = pd.DataFrame(spread).reset_index()



fig = px.line(spread, x='Date', y='Country/Region', 

              title='Number of Countries/Regions to which COVID-19 spread over the time',

             color_discrete_sequence=cdr)

fig.show()
gdf = gdf = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()

gdf = gdf.reset_index()



temp = gdf[gdf['Country/Region']=='China'].reset_index()

temp = temp.melt(id_vars='Date', value_vars=['Confirmed', 'Deaths', 'Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp, x="Date", y="Count", color='Case', facet_col="Case",

            title='Cases in China', color_discrete_sequence=cdr)

fig.show()



temp = gdf[gdf['Country/Region']!='China'].groupby('Date').sum().reset_index()

temp = temp.melt(id_vars='Date', value_vars=['Confirmed', 'Deaths', 'Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp, x="Date", y="Count", color='Case', facet_col="Case",

             title='Cases Outside China', color_discrete_sequence=cdr)

fig.show()
def location(row):

    if row['Country/Region']=='China':

        if row['Province/State']=='Hubei':

            return 'Hubei'

        else:

            return 'Other Chinese Provinces'

    else:

        return 'Rest of the World'



temp = full_table.copy()

temp['Region'] = temp.apply(location, axis=1)

temp['Date'] = temp['Date'].dt.strftime('%Y-%m-%d')

temp = temp.groupby(['Region', 'Date'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

temp = temp.melt(id_vars=['Region', 'Date'], value_vars=['Confirmed', 'Deaths', 'Recovered'], 

                 var_name='Case', value_name='Count').sort_values('Count')

# temp = temp.sort_values(['Date', 'Region', 'Case']).reset_index()

temp.head()



fig = px.bar(temp, y='Region', x='Count', color='Case', barmode='group', orientation='h',

             text='Count', title='Hubei - China - World', animation_frame='Date',

             color_discrete_sequence= [d, r, c], range_x=[0, 70000])

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.update_layout(yaxis={'categoryorder':'array', 

                         'categoryarray':['Hubei','Other Chinese Provinces','Rest of the World']})

fig.show()



temp = full_latest.copy()

temp['Region'] = temp.apply(location, axis=1)

temp = temp.groupby('Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

temp = temp.melt(id_vars='Region', value_vars=['Confirmed', 'Deaths', 'Recovered'], 

                 var_name='Case', value_name='Count').sort_values('Count')

temp.head()



fig = px.bar(temp, y='Region', x='Count', color='Case', barmode='group',

             text='Count', title='Hubei - China - World', 

             color_discrete_sequence= [d, r, c])

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
temp = full_table.groupby('Date').sum().reset_index()

temp.head()



# adding two more columns

temp['No. of Deaths to 100 Confirmed Cases'] = round(temp['Deaths']/

                                                     temp['Confirmed'], 3)*100

temp['No. of Recovered to 100 Confirmed Cases'] = round(temp['Recovered']/

                                                        temp['Confirmed'], 3)*100

temp['No. of Recovered to 1 Death Case'] = round(temp['Recovered']/

                                                 temp['Deaths'], 3)



temp = temp.melt(id_vars='Date', 

                 value_vars=['No. of Deaths to 100 Confirmed Cases', 

                             'No. of Recovered to 100 Confirmed Cases', 

                             'No. of Recovered to 1 Death Case'], 

                 var_name='Ratio', 

                 value_name='Value')



fig = px.line(temp, x="Date", y="Value", color='Ratio', 

              title='Recovery and Mortality Rate Over The Time',color_discrete_sequence=cdr)

fig.show()
rl = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum()

rl = rl.reset_index().sort_values(by='Confirmed', ascending=False).reset_index(drop=True)

rl.head().style.background_gradient(cmap='rainbow')



ncl = rl.copy()

ncl['Affected'] = ncl['Confirmed'] - ncl['Deaths'] - ncl['Recovered']

ncl = ncl.melt(id_vars="Country/Region", value_vars=['Affected', 'Recovered', 'Deaths'])



fig = px.bar(ncl.sort_values(['variable', 'value']), 

             x="Country/Region", y="value", color='variable', orientation='v', height=800,

             # height=600, width=1000,

             title='Number of Cases outside China', color_discrete_sequence=cdr)

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()



# ------------------------------------------



cl = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum()

cl = cl.reset_index().sort_values(by='Confirmed', ascending=False).reset_index(drop=True)

# cl.head().style.background_gradient(cmap='rainbow')



ncl = cl.copy()

ncl['Affected'] = ncl['Confirmed'] - ncl['Deaths'] - ncl['Recovered']

ncl = ncl.melt(id_vars="Province/State", value_vars=['Affected', 'Recovered', 'Deaths'])



fig = px.bar(ncl.sort_values(['variable', 'value']), 

             y="Province/State", x="value", color='variable', orientation='h', height=800,

             # height=600, width=1000,

             title='Number of Cases in China', color_discrete_sequence=cdr)

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = px.treemap(china.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

                 path=["Province/State"], values="Confirmed",

                 title='Number of Confirmed Cases in Chinese Provinces',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.show()



fig = px.treemap(china.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 

                 path=["Province/State"], values="Deaths", 

                 title='Number of Deaths Reported in Chinese Provinces',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.show()



fig = px.treemap(china.sort_values(by='Recovered', ascending=False).reset_index(drop=True), 

                 path=["Province/State"], values="Recovered", 

                 title='Number of Recovered Cases in Chinese Provinces',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.show()



# ----------------------------------------------------------------------------



fig = px.treemap(row, path=["Country/Region"], values="Confirmed", 

                 title='Number of Confirmed Cases outside china',

                 color_discrete_sequence = px.colors.qualitative.Pastel)

fig.show()



fig = px.treemap(row, path=["Country/Region"], values="Deaths", 

                 title='Number of Deaths outside china',

                 color_discrete_sequence = px.colors.qualitative.Pastel)

fig.show()



fig = px.treemap(row, path=["Country/Region"], values="Recovered", 

                 title='Number of Recovered Cases outside china',

                 color_discrete_sequence = px.colors.qualitative.Pastel)

fig.show()
temp = full_table.groupby(['Date', 'Country/Region'])['Confirmed'].sum()

temp = temp.reset_index().sort_values(by=['Date', 'Country/Region'])

temp.head()
px.line(temp, x='Date', y='Confirmed')
temp = full_table.groupby(['Date', 'Country/Region'])['Confirmed'].sum()

temp = temp.reset_index().sort_values(by=['Date', 'Country/Region'])



plt.style.use('seaborn')

g = sns.FacetGrid(temp, col="Country/Region", hue="Country/Region", 

                  sharey=False, col_wrap=5)

g = g.map(plt.plot, "Date", "Confirmed")

g.set_xticklabels(rotation=90)

plt.show()
temp = full_table.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



plt.style.use('seaborn')

g = sns.FacetGrid(temp, col="Country/Region", hue="Country/Region", 

                  sharey=False, col_wrap=5)

g = g.map(sns.lineplot, "Date", "Confirmed")

g.set_xticklabels(rotation=90)

plt.show()
temp = full_table.groupby('Date')['Confirmed'].sum()

temp = temp.diff()



plt.figure(figsize=(20, 5))

calmap.yearplot(temp, fillcolor='white', cmap='Reds', linewidth=0.5)

plt.plot()
spread = full_table[full_table['Confirmed']!=0].groupby('Date')

spread = spread['Country/Region'].unique().apply(len).diff()



plt.figure(figsize=(20, 5))

calmap.yearplot(spread, fillcolor='white', cmap='Greens', linewidth=0.5)

plt.plot()