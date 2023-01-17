import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns
%matplotlib inline
barrolee=pd.read_csv('../input/split-data-from-edu-stat/barrolee.csv')
barrolee.head()
# Choose one among numerous indicators: Average years of total schooling, age 25+
schooling=barrolee[barrolee['Indicator'].str.contains('Average years of total schooling, age 25+')]
schooling.head()
schooling['Indicator'].unique()
# Extract data that includes 'total' and '25+'
finalschooling=schooling[~schooling['Indicator'].str.contains('female|29')]
finalschooling['Indicator'].unique()
finalschooling=finalschooling.reset_index(drop=True)
# Draw graph with 2010 data
finalschooling=finalschooling.sort_values(by='2010', ascending=True)
import matplotlib.style as style
style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(20,35))
mygraph = ax.barh(finalschooling['Country'], finalschooling['2010'])
plt.axvline(x = np.mean(finalschooling['2010']), color = 'black', linewidth = 1.5, alpha = .8)
plt.title('Average years of total schooling, age 25+',fontsize=32,fontweight='bold', loc='left')
test=pd.read_csv('../input/split-data-from-edu-stat/test.csv')
test.shape
pisa=test[test['Indicator'].str.contains('PISA')]
pisa.shape
pirls=test[test['Indicator'].str.contains('PIRLS')]
pirls.shape
llece=test[test['Indicator'].str.contains('LLECE')]
llece.shape
pisa.isnull().sum()
# drop years with too many NaN values
pisa=pisa.drop(['1995','1997', '1999', '2001', '2004','2005','2007','2008','2010','2011','2013','2014','2016'], axis=1)
pisa.head()
# Extract my interests
criteria = pisa['Indicator'].isin(['PISA: 15-year-olds by mathematics proficiency level (%). Below Level 1',
       'PISA: 15-year-olds by mathematics proficiency level (%). Level 1',
       'PISA: 15-year-olds by mathematics proficiency level (%). Level 2',
       'PISA: 15-year-olds by mathematics proficiency level (%). Level 3',
       'PISA: 15-year-olds by mathematics proficiency level (%). Level 4',
       'PISA: 15-year-olds by mathematics proficiency level (%). Level 5',
       'PISA: 15-year-olds by mathematics proficiency level (%). Level 6',
       'PISA: 15-year-olds by reading proficiency level (%). Below Level 1B',
       'PISA: 15-year-olds by reading proficiency level (%). Level 1A',
       'PISA: 15-year-olds by reading proficiency level (%). Level 1B',
       'PISA: 15-year-olds by reading proficiency level (%). Level 2',
       'PISA: 15-year-olds by reading proficiency level (%). Level 3',
       'PISA: 15-year-olds by reading proficiency level (%). Level 4',
       'PISA: 15-year-olds by reading proficiency level (%). Level 5',
       'PISA: 15-year-olds by reading proficiency level (%). Level 6',
       'PISA: 15-year-olds by science proficiency level (%). Below Level 1B',
       'PISA: 15-year-olds by science proficiency level (%). Level 1A',
       'PISA: 15-year-olds by science proficiency level (%). Level 1B',
       'PISA: 15-year-olds by science proficiency level (%). Level 2',
       'PISA: 15-year-olds by science proficiency level (%). Level 3',
       'PISA: 15-year-olds by science proficiency level (%). Level 4',
       'PISA: 15-year-olds by science proficiency level (%). Level 5',
       'PISA: 15-year-olds by science proficiency level (%). Level 6'])
newpisa=pisa[criteria]
newpisa.shape
# reading data
pisareading=newpisa[newpisa['Indicator'].str.contains('reading')]
pisareading.head()
# Make pivot table for drawing horizontal stacked bar graph
pisa_reading=pd.pivot_table(pisareading, index='Code', columns='Indicator', values=['2015'], dropna=True)
pisa_reading['sum']=pisa_reading.sum(axis=1)
pisa_reading=pisa_reading[pisa_reading['sum']!=0]
pisa_reading=pisa_reading.drop('sum',axis=1)
pisa_reading
# Change row orders -> enables to draw bar graphs alphabetical order
pisa_reading=pisa_reading[::-1]
pal=sns.cubehelix_palette(8)
pisareadinggraph=pisa_reading.plot(kind='barh', stacked=True, figsize=(20,36), color=pal, title='PISA: 15-year-olds by reading proficiency level(%) [Below Level1B - Level1A - Level1B - Level2 - Level3 - Level4 - Level5 - Level6]', legend=False, xlim=(0,100))
pisareadinggraph
# Math data
pisamath=newpisa[newpisa['Indicator'].str.contains('mathematics')]
pisamath.shape
pisa_math=pd.pivot_table(pisamath,index='Code',columns='Indicator',values='2015')
pisa_math=pisa_math[::-1]
pisa_math['sum']=pisa_math.sum(axis=1)
pisa_math=pisa_math[pisa_math['sum']!=0]
pisa_math=pisa_math.drop('sum',axis=1)
pal2=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=False)
pisamathgraph=pisa_math.plot(kind='barh', stacked=True, figsize=(20,36), color=pal2, title='PISA: 15-year-olds by mathematics proficiency level(%) [Below Level1B - Level1A - Level1B - Level2 - Level3 - Level4 - Level5 - Level6]',legend=False, xlim=(0,100))
pisamathgraph
# science data
pisascience=newpisa[newpisa['Indicator'].str.contains('science')]
pisascience.shape
pisa_science=pd.pivot_table(pisascience,index='Code', columns='Indicator', values='2015')
pisa_science=pisa_science[::-1]
pisa_science['sum']=pisa_science.sum(axis=1)
pisa_science=pisa_science[pisa_science['sum']!=0]
pisa_science=pisa_science.drop('sum',axis=1)
pal3=sns.light_palette("Navy", as_cmap=True)
pisasciencegraph=pisa_science.plot(kind='barh', stacked=True, figsize=(20,36), cmap=pal3, title='PISA: 15-year-olds by science proficiency level(%) [Below Level1B - Level1A - Level1B - Level2 - Level3 - Level4 - Level5 - Level6]', legend=False, xlim=(0,100))
criteria = pirls['Indicator'].isin(['PIRLS: Distribution of Reading Scores: 5th Percentile Score', 'PIRLS: Distribution of Reading Scores: 25th Percentile Score', 'PIRLS: Distribution of Reading Scores: 50th Percentile Score', 'PIRLS: Distribution of Reading Scores: 75th Percentile Score', 'PIRLS: Distribution of Reading Scores: 95th Percentile Score'])
pirlsreading=pirls[criteria]
pirlsreading=pirlsreading.reset_index(drop=True)
pirlsreading.head()
pirlsreading.shape
pirls_reading=pd.pivot_table(pirlsreading, index='Indicator', columns='Code', values='2016')
# Change the columns' order
pirls_reading = pirls_reading.sort_index(axis=1 ,ascending=False)
pirlsreadinggraph=pirls_reading.plot(kind='box', vert=0, figsize=(20,36), title='PIRLS: Distribution of Reading Scores [5th Percentile - 25th Percentile - 50th Percentile - 75th Percentile - 95th Percentile]')
llece.isnull().sum()
# Drop years with too many NaN values
llece=llece.drop(['1995','1999', '2000', '2001','2003','2004','2005','2007','2008','2009','2010','2011','2012','2014','2015','2016'], axis=1)
llece.head()
llece=llece[llece['Indicator'].str.contains('Mean performance on the')]
llece.head()
llece=llece[~llece['Indicator'].str.contains('total')]
llece.head()
llece['Indicator'].unique()
llece3rd=llece[llece['Indicator'].str.contains('3rd')]
llece3rd=llece3rd.reset_index(drop=True)
# Only 2006 data remains
llece3rd=llece3rd.drop(['1997','2013'],axis=1)
llece3rd.head()
# Make a new column called gender_gap
llece3rd['gender_gap']=np.repeat([abs(llece3rd.iloc[2*i,3]-llece3rd.iloc[2*i+1,3]) for i in range(34)],2)
# rearrange order by values
llece3rd2=llece3rd.sort_values(by='gender_gap', ascending=False)
llece3rd2.head(20)
list(llece3rd2['Indicator'].unique())
llece3rdfinal=pd.read_csv('../input/split-data-from-edu-stat/llece3rd2.csv')
# Delete rows with na values
llece3rdfinal=llece3rdfinal.dropna()
llece3rdfinal.columns
# Extract top5 gender gap countries
top_5_gap_countries = ['BRA', 'ARG', 'PRY', 'PAN', 'CHL']
llece3rdfinal.plot.bar(figsize=(16,8), color=['r','r','b','b'], grid=True, width=0.8, x='Code',y=['reading scale mean for 3rd grade students, female',
       'reading scale mean for 3rd grade students, male',
       'mathematics scale mean for 3rd grade students, female',
       'mathematics scale mean for 3rd grade students, male'], ylim=(300,700), legend='topright', title='LLECE Score')
income_oecd=pd.read_csv('../input/split-data-from-edu-stat/income_oecd.csv')
income_oecd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format', '{:.2f}'.format)
array=['GDP per capita (current US$)', 'Labor force, total','Population growth (annual %)', 'Unemployment, total (% of total labor force)']
oecd=income_oecd.loc[income_oecd['Indicator'].isin(array)]
# Delete years that data is unavailable
oecd=oecd.iloc[:,:51]
import matplotlib.style as style
style.use('default')
x1=list(range(1970,2018))
xi=[i for i in x1]
plt.subplots(1,1, figsize=(16,8))
plt.plot(xi, oecd.iloc[0,3:], linewidth=4)
plt.xticks(np.arange(1970,2018,step=3))
plt.plot(xi, oecd.iloc[4,3:], linewidth=4)
plt.grid(color='grey', linestyle='-', linewidth=0.4)
plt.title('GDP per capita(current US$)',fontsize=24, y=1.04)
plt.legend(['OECD','World'],loc=(0.92,1.02))
plt.show()
style.use('fivethirtyeight')
x2=list(range(1989,2018))
xj=[j for j in x2]
x3=list(range(1990,2019))
xk=[k for k in x3]
f = plt.figure(figsize=(16,18))
ax=f.add_subplot(311)
ax.set_title('Labor Force, total', fontsize=16)
ax.plot(xk, oecd.iloc[1,22:])
ax.plot(xk, oecd.iloc[5,22:])
ax.set_xticks(np.arange(1990,2018,step=3))
ax.set_ylim([200000000, 3700000000])
ax.legend(['OECD','World'],loc=(0.85,0.95))

ax2=f.add_subplot(312)
ax2.set_title('Unemployment, total (% of total labor force)', fontsize=16)
ax2.plot(xk, oecd.iloc[3,22:])
ax2.plot(xk, oecd.iloc[7,22:])
ax2.set_xticks(np.arange(1990,2018,step=3))
ax2.legend(['OECD','World'],loc=(0.85,0.95))
ax2.set_ylim([5.2,9])

ax3=f.add_subplot(313)
ax3.set_title('Population growth (annual %)', fontsize=16)
ax3.plot(xk, oecd.iloc[2,22:])
ax3.plot(xk, oecd.iloc[6,22:])
ax3.set_xticks(np.arange(1990,2018,step=3))
ax3.legend(['OECD','World'], loc=(0.85,0.95))
ax3.set_ylim([-0.3,3])

plt.show()
mobility_cont=pd.read_csv('../input/split-data-from-edu-stat/mobility_cont.csv')
mobility_cont['Indicator'].unique()
array=['Inbound mobility rate, both sexes (%)', 'Outbound mobility ratio, all regions, both sexes (%)', 'Total outbound internationally mobile tertiary students studying abroad, all countries, both sexes (number)', 'Net flow of internationally mobile students (inbound - outbound), both sexes (number)']
mobility=mobility_cont.loc[mobility_cont['Indicator'].isin(array)]
mobility['Country'].unique()
array3=['Arab World', 'East Asia & Pacific',
       'East Asia & Pacific (excluding high income)', 'Euro area',
       'Europe & Central Asia', 'Europe & Central Asia (excluding high income)', 'European Union',
       'Latin America & Caribbean', 'Latin America & Caribbean (excluding high income)',
       'Middle East & North Africa', 'Middle East & North Africa (excluding high income)',
       'North America', 'South Asia', 'Sub-Saharan Africa', 'Sub-Saharan Africa (excluding high income)']
mobility_continent=mobility.loc[mobility['Country'].isin(array3)]
mobility_continent
mobility_continent_in=mobility_continent[mobility_continent['Indicator'].str.contains('Inbound')]
x4=list(range(1997,2015))
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors=['black','r','darkred','gold','orange','brown','moccasin','m','darkmagenta','lightslategrey','darkgrey','b', 'indianred','lawngreen','darkgreen']
mobility_continent_out=mobility_continent[mobility_continent['Indicator'].str.contains('Outbound|outbound')].reset_index(drop=True)
mobility_continent_out['Country'].unique()
import matplotlib.style as style
import matplotlib.style as style
style.use('default')
f = plt.figure(figsize=(16,24))
ax=f.add_subplot(311)
ax.set_title('Inbound mobility rate(%)', fontsize=16)
for i,j in zip(range(0,15),colors):
    ax.plot(x4,mobility_continent_in.iloc[i,3:],c=j)
ax.set_ylim([-0.3,8])
ax.set_xticks(np.arange(1997,2014,step=2))
ax.grid(linewidth=0.5)
ax.legend(mobility_continent_in['Code'])

ax2=f.add_subplot(312)
ax2.set_title('Outbound mobility rate(%)', fontsize=16)
for k,l in zip(range(0,15),colors):
    ax2.plot(x4, mobility_continent_out.iloc[2*k,3:],c=l)
ax2.set_ylim([0,7])
ax2.set_xticks(np.arange(1997,2014,step=2))
ax2.grid(linewidth=0.5)
ax2.legend(mobility_continent_out['Code'],loc='upper right', ncol=5)

ax3=f.add_subplot(313)
ax3.set_title('Total outbound internationally mobile students(number) (YEAR 2014)', fontsize=16)
for m,n in zip(range(0,15),colors):
    ax3.bar(mobility_continent_out.iloc[2*m+1,1], mobility_continent_out.iloc[2*m+1,19], color=n)
ax3.set_ylim([0,1300000])
ax3.legend(mobility_continent_out['Code'],loc='upper right', ncol=3)
ax3.grid(linewidth=0.5)
afc=pd.read_csv('../input/split-data-from-edu-stat/africafacilities_country.csv')
afc=afc.drop(['2007','2008','2009','2010','2011','2013','2014','2015'],axis=1)
afc.head()
criteria2 = afc['Indicator'].isin(['Africa Dataset: Percentage of lower secondary schools with access to electricity (%)',
       'Africa Dataset: Percentage of lower secondary schools with access to potable water (%)', 
       'Africa Dataset: Percentage of lower secondary schools with toilets (%)',
       'Africa Dataset: Percentage of primary schools with access to electricity (%)',
       'Africa Dataset: Percentage of primary schools with access to potable water (%)',
       'Africa Dataset: Percentage of primary schools with toilets (%)'])
afc=afc[criteria2].reset_index(drop=True)
afc.head()
import folium
import json
import cufflinks as cf
import seaborn as sns
sns.set() #https://seaborn.pydata.org/generated/seaborn.set.html
cf.go_offline()  # required to use plotly offline (no account required)
%matplotlib inline 
%config InlineBackend.figure_format = 'retina' 
africa_map = folium.Map(location=[0.311106, 23.969148], zoom_start=3)
africa_map
afc_lower_elec=afc[afc['Indicator'].isin(['Africa Dataset: Percentage of lower secondary schools with access to electricity (%)'])].reset_index(drop=True)
afc_lower_water=afc[afc['Indicator'].isin(['Africa Dataset: Percentage of lower secondary schools with access to potable water (%)'])].reset_index(drop=True)
afc_lower_toilet=afc[afc['Indicator'].isin(['Africa Dataset: Percentage of lower secondary schools with toilets (%)'])].reset_index(drop=True)
afc_prim_elec=afc[afc['Indicator'].isin(['Africa Dataset: Percentage of primary schools with access to electricity (%)'])].reset_index(drop=True)
afc_prim_water=afc[afc['Indicator'].isin(['Africa Dataset: Percentage of primary schools with access to potable water (%)'])].reset_index(drop=True)
afc_prim_toilet=afc[afc['Indicator'].isin(['Africa Dataset: Percentage of primary schools with toilets (%)'])].reset_index(drop=True)
africa_map = folium.Map(location=[0, 23.011837], zoom_start=3)

world_geo = '../input/worldjson/world.json'
africa_map.choropleth(geo_data='../input/worldjson/world.json',
              data=afc_lower_elec, columns=['Country', '2012'], key_on='properties.name',
              fill_color='YlGn', fill_opacity=0.7, line_opacity=0.5)

africa_map
africa_map = folium.Map(location=[0, 23.011837], zoom_start=3)

world_geo = '../input/worldjson/world.json'
africa_map.choropleth(geo_data='../input/worldjson/world.json',
              data=afc_lower_water, columns=['Country', '2012'], key_on='properties.name',
              fill_color='PuBu', fill_opacity=0.7, line_opacity=0.5)

africa_map
africa_map = folium.Map(location=[0, 23.011837], zoom_start=3)

world_geo = '../input/worldjson/world.json'
africa_map.choropleth(geo_data='../input/worldjson/world.json',
              data=afc_lower_toilet, columns=['Country', '2012'], key_on='properties.name',
              fill_color='OrRd', fill_opacity=0.7, line_opacity=0.5)

africa_map
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
plotly.tools.set_credentials_file(username='HyunjooKim', api_key='v5zTFmz0DExZ1aODl1AZ')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go

init_notebook_mode(connected=True)

for col in afc_lower_elec.columns:
    afc_lower_elec[col] = afc_lower_elec[col].astype(str)

scl = [[0.0, 'rgb(206,251,201)'],[0.2, 'rgb(183,240,177)'],[0.4, 'rgb(134,229,127)'],\
            [0.6, 'rgb(71,200,62)'],[0.8, 'rgb(47,157,39)'],[1.0, 'rgb(34,116,28)']]

afc_lower_elec['text'] = afc_lower_elec['Country'] + '<br>' +\
    '(%)' + afc_lower_elec['2012']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = afc_lower_elec['Country'],
        z = afc_lower_elec['2012'].astype(float),
        locationmode = 'country names',
        text = afc_lower_elec['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Africa Dataset: Percentage of primary schools with access to electricity (%)")
        ) ]

layout = dict(
        title = 'Africa Dataset: Percentage of primary schools with access to electricity (%)<br>(Hover for breakdown)',
        geo = dict(
            scope='africa',
            projection=dict( type='mercator' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = go.Figure(dict( data=data, layout=layout ))
py.offline.iplot(fig)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go

init_notebook_mode(connected=True)

for col in afc_prim_water.columns:
    afc_prim_water[col] = afc_prim_water[col].astype(str)

scl = [[0.0, 'rgb(217,229,255)'],[0.2, 'rgb(178,204,255)'],[0.4, 'rgb(103,153,255)'],\
            [0.6, 'rgb(67,116,217)'],[0.8, 'rgb(0,51,153)'],[1.0, 'rgb(0,34,103)']]

afc_prim_water['text'] = afc_prim_water['Country'] + '<br>' +\
    '(%)' + afc_prim_water['2012']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = afc_prim_water['Country'],
        z = afc_prim_water['2012'].astype(float),
        locationmode = 'country names',
        text = afc_prim_water['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Africa Dataset: Percentage of primary schools with access to potable water (%)")
        ) ]

layout = dict(
        title = 'Africa Dataset: Percentage of primary schools with access to potable water (%)<br>(Hover for breakdown)',
        geo = dict(
            scope='africa',
            projection=dict( type='mercator' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = go.Figure(dict( data=data, layout=layout ))
py.offline.iplot(fig)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go

init_notebook_mode(connected=True)

for col in afc_prim_toilet.columns:
    afc_prim_toilet[col] = afc_prim_toilet[col].astype(str)

scl = [[0.0, 'rgb(255,216,216)'],[0.2, 'rgb(255,167,167)'],[0.4, 'rgb(241,95,95)'],\
            [0.6, 'rgb(204,61,61)'],[0.8, 'rgb(152,0,0)'],[1.0, 'rgb(103,0,0)']]

afc_prim_toilet['text'] = afc_prim_toilet['Country'] + '<br>' +\
    '(%)' + afc_prim_toilet['2012']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = afc_prim_toilet['Country'],
        z = afc_prim_toilet['2012'].astype(float),
        locationmode = 'country names',
        text = afc_prim_toilet['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Africa Dataset: Percentage of primary schools with access to toilets (%)")
        ) ]

layout = dict(
        title = 'Africa Dataset: Percentage of primary schools with access to toilets (%)<br>(Hover for breakdown)',
        geo = dict(
            scope='africa',
            projection=dict( type='mercator' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = go.Figure(dict( data=data, layout=layout ))
py.offline.iplot(fig)
