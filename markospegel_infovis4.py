from datetime import date
date.today()
import pandas as pd
import numpy as np
import math
from datetime import date
from copy import deepcopy
from pyproj import Transformer

from bokeh.models import ColumnDataSource, GMapOptions, CDSView
from bokeh.models import CustomJS, Slider, DatetimeTickFormatter, Panel
from bokeh.models.filters import CustomJSFilter
from bokeh.models.widgets import DateSlider, Tabs

from bokeh.tile_providers import get_provider, Vendors
from bokeh.plotting import ColumnDataSource, figure, output_file, show, reset_output
from bokeh.layouts import column, row
from bokeh.io import output_notebook
casesDf=pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
casesDf.head()
casesDf.shape
casesDf['Date']=casesDf['Date'].apply(lambda x: date(int(x.split('-')[0]),int(x.split('-')[1]),int(x.split('-')[2])))
casesDf.fillna(value={'Province/State':'No state or province'},inplace=True)
casesDf.head()
transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
casesDf['merLon'],casesDf['merLat']=transformer.transform(casesDf['Long'].values, casesDf['Lat'].values)
casesDf['Active']=casesDf['Active'].apply(lambda x: max(x,0)) # sanitize values to be >= 0
casesDf['size']=casesDf['Active'].apply(lambda x: math.log(x+1)*4)
casesDf.sort_values(['Province/State', 'Country/Region','Date'], ascending=[True, True,True],inplace=True)
casesDf.reset_index(drop=True,inplace=True)

colors=[]
casesDf.loc[0, 'color']='green'
for i in range(1, len(casesDf)):
    if(casesDf.loc[i-1,'Province/State']==casesDf.loc[i,'Province/State']
       and casesDf.loc[i-1,'Country/Region']==casesDf.loc[i,'Country/Region']):
        
        if(casesDf.loc[i-1,'Active']>=casesDf.loc[i,'Active']):
            casesDf.loc[i, 'color'] = 'green'
        else:
            casesDf.loc[i, 'color'] = 'red'
    else:
        casesDf.loc[i, 'color']='green'
casesDf.head()
activePerCountry=casesDf[['Country/Region','Date','Active']].groupby(['Country/Region','Date'],as_index=False).sum()
activePerCountry.head()
countriesPop=pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')
countriesPop.rename({'Country (or dependency)':'Country/Region'},axis=1,inplace=True)
countriesPop.head()
countriesContinent=pd.read_csv('../input/country-to-continent/countryContinent.csv',encoding='latin-1')
countriesContinent.rename({'country':'Country/Region'},axis=1,inplace=True)
countriesContinent.head()
mergedDf=activePerCountry.merge(countriesPop)
mergedDf=mergedDf.merge(countriesContinent)
mergedDf.rename({'Country/Region':'country'},axis=1,inplace=True)
mergedDf=mergedDf[['country','continent', 'Date', 'Active', 'Population (2020)', 'sub_region']]

mergedDf=mergedDf[mergedDf['Population (2020)']>100000]
mergedDf=mergedDf[mergedDf['country']!='Qatar']
mergedDf=mergedDf[mergedDf['country']!='Luxembourg']
mergedDf['casesPer100k'] = 100000. * mergedDf['Active']/mergedDf['Population (2020)']

continents=mergedDf.continent.unique()
colors=['green','blue','black','red','yellow']
continentColors=dict(zip(continents,colors))
print(continentColors)
print()
mergedDf['color']=mergedDf['continent'].apply(lambda x: continentColors[x])
mergedDf['dateStr']=mergedDf['Date'].apply(lambda x: str(x))

mergedDf.head(3)
continentDf=mergedDf\
    .groupby(['continent','Date'])\
    .agg({'Active': "sum", 'Population (2020)': "sum"})\
    .reset_index()
continentDf['casesPer100k']= 100000 * continentDf['Active']/continentDf['Population (2020)']
continentDf['dateStr']=continentDf['Date'].apply(lambda x: str(x))
continentDf['country']=continentDf['Date'].apply(lambda x: "All")
continentDf=continentDf[['country', 'continent', 'Date', 'casesPer100k', 'dateStr']]
continentDf.head(3)
regionDf=mergedDf\
    .groupby(['continent', 'sub_region', 'Date'])\
    .agg({'Active': "sum", 'Population (2020)': "sum"})\
    .reset_index()
regionDf['casesPer100k']= 100000 * regionDf['Active']/regionDf['Population (2020)']
regionDf['dateStr']=regionDf['Date'].apply(lambda x: str(x))
regionDf['country']=regionDf['sub_region']
regionDf=regionDf[['country', 'continent', 'Date', 'casesPer100k', 'dateStr']]
regionDf.head(3)
source = ColumnDataSource(data=dict(merLon=casesDf['merLon'], merLat=casesDf['merLat'], size=casesDf['size'],date=casesDf['Date'],
                                    Country=casesDf['Country/Region'],Confirmed=casesDf['Confirmed'],Deaths=casesDf['Deaths'],Recovered=casesDf['Recovered'],
                                    Active=casesDf['Active'],province=casesDf['Province/State'],color=casesDf['color']))

tile_provider = get_provider(Vendors.CARTODBPOSITRON)

TOOLTIPS = [
    ("Country", "@Country"),
    ("Province/state", "@province"),
    ("Active", "@Active"),
    ("Confirmed", "@Confirmed"),
    ("Deaths", "@Deaths"),
    ("Recovered", "@Recovered")
]

mapChart = figure(x_range=(-1.5e7, 1.8e7), y_range=(-5e6, 1e7),
           title = 'COVID-19 Cases Per Country/Region. Move slider to change date.',
           x_axis_type="mercator", y_axis_type="mercator", 
           plot_width=800, plot_height=450, tooltips=TOOLTIPS)

mapChart.add_tile(tile_provider)

date_range_slider = DateSlider(title="Date Range: ", start=min(casesDf['Date']), 
                               end=max(casesDf['Date']), value=max(casesDf['Date']))

# this filter selects rows of data source that satisfy the constraint
custom_filter = CustomJSFilter(args=dict(slider=date_range_slider), code="""
    function roundDate(timeStamp){
        timeStamp -= timeStamp % (24 * 60 * 60 * 1000);//subtract amount of time since midnight
        return timeStamp;
    }
    
    const indices = []
    for (var i = 0; i < source.get_length(); i++) {
        if (source.data['date'][i] == roundDate(slider.value)) {
            indices.push(true)
        } else {
            indices.push(false)
        }
    }
    return indices
""")

view = CDSView(source=source, filters=[custom_filter])

# re-render
date_range_slider.js_on_change('value', CustomJS(args=dict(source=source), code="""
   source.change.emit()
"""))

mapChart.circle(x="merLon", y="merLat", size='size', fill_color="color", 
                fill_alpha=0.8, source=source,view=view)
TOOLTIPS = [("Date", "@dateStr"), ("Cases per 100.000", "@casesPer100k"),\
        ("Region", "@country"), ("Continent", "@continent")]

lineChart = figure(plot_width=800, plot_height=350, tooltips=TOOLTIPS, 
    title = 'COVID-19 Cases Per 100.000 People, Continents and Regions, Linear Scale. Click legend items to mute.',
    x_axis_label = 'Date', y_axis_label = 'Cases per 100.000 (linear scale)')

for cont, c in zip(continents, colors):
    dfC = continentDf[continentDf['continent']==cont]
    sourceC = ColumnDataSource(dfC)
    lineChart.line(x='Date', y='casesPer100k',color=c,alpha=0.9,
                   muted_color=c, muted_alpha=0.1,
                   line_width=2,source=sourceC,legend_label=cont)

    regions = regionDf[regionDf['continent']==cont].country.unique()
    for reg in regions:
        dfR = regionDf[regionDf['country']==reg]
        sourceR = ColumnDataSource(dfR)
        lineChart.line(x='Date', y='casesPer100k',color=c,
                       muted_color=c, muted_alpha=0.2,
                       line_width=0.5,source=sourceR,legend_label=cont)

lineChart.legend.location = "top_left"
lineChart.legend.click_policy = "mute"

lineChart.xaxis.formatter=DatetimeTickFormatter(hours=["%d %B %Y"],
            days=["%d %B %Y"],  months=["%d %B %Y"], years=["%d %B %Y"])
TOOLTIPS2 = [("Date", "@dateStr"), ("Cases per 100.000", "@casesPer100k"),
        ("Country", "@country"), ("Continent", "@continent")]

lineChart2 = figure(plot_width=800, plot_height=450, tooltips=TOOLTIPS2, y_axis_type="log", 
    title = 'COVID-19 Cases Per 100.000 People, Continents and Countries, Logarithmic Scale. Click legend items to hide.',
    x_axis_label = 'Date', y_axis_label = 'Cases per 100.000 (log scale)')

for cont, c in zip(continents, colors):
    dfC2 = continentDf[continentDf['continent']==cont]
    sourceC2 = ColumnDataSource(dfC2)
    lineChart2.line(x='Date', y='casesPer100k',color=c,alpha=0.9,
                                   line_width=2.5,source=sourceC2,legend_label=cont)

    countries = mergedDf[mergedDf['continent']==cont].country.unique()
    for country in countries:
        dfR2 = mergedDf[mergedDf['country']==country]
        sourceR2 = ColumnDataSource(dfR2)
        lineChart2.line(x='Date', y='casesPer100k',color=c,alpha=0.7,
                       line_width=0.5,source=sourceR2,legend_label=cont)

lineChart2.legend.location = "top_left"
lineChart2.legend.click_policy = "hide"
lineChart2.xaxis.formatter=DatetimeTickFormatter(hours=["%d %B %Y"],
            days=["%d %B %Y"],  months=["%d %B %Y"], years=["%d %B %Y"])
source2 = ColumnDataSource(mergedDf)

TOOLTIPS = [
    ("Date", "@dateStr"),
    ("Cases per 100.000", "@casesPer100k"),
    ("Country", "@country"),
    ("Continent", "@continent"),
]
scatterChart = figure(plot_width=800, plot_height=450,tooltips=TOOLTIPS,
                    title = 'COVID-19 Cases Per 100.000 People, Countries',
                    x_axis_label = 'Date', y_axis_label = 'Cases per 100.000')
scatterChart.circle(x='Date', y='casesPer100k',color='color',source=source2,fill_color="white")
scatterChart.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d %B %Y"],
        days=["%d %B %Y"],
        months=["%d %B %Y"],
        years=["%d %B %Y"],
    )
output_notebook()
layout = row( column(date_range_slider,mapChart,lineChart,lineChart2,scatterChart) )
show (layout)