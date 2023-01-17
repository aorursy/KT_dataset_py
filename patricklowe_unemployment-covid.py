import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import plotly.express as px # For Graphing

import pandas as pd

import numpy as np

import plotly

import plotly.graph_objects as go

from plotly.subplots import make_subplots



# Dataset description

"""

S_ADJ: (Seasonally Adjusted Data)

    NSA - Not Seasonally Adjusted

    TC - Trend Cycle

    SA - Seasonally Adjusted

AGE:

    TOTAL - Total 

    Y25-74 - 25 to 74

    Y_LT25 - less than 25

UNIT:

    PC_ACT - % of active population

    THS_PER - Thousand persons

SEX: 

    F - Female

    M - Male

    T - Total

GEO_TIME:

    Country Code

    

Data Flags:

    b - break in time series

    e - estimated

    p - provisional

    u - low reliability

    c- confidential

    f - forecast

    r- revised

    z - not applicable

    d - definition differs

    n - not significant

    s - eurostat estimate

    : - unavailable

"""
# Load the data

df = pd.read_csv('../input/unemployment-in-european-union/une_rt_m.tsv',sep='\t')

df.head(5)
# Expand first column into separate columns

df2 = df.iloc[:,0].str.split(',',expand=True) # Creates new dataframe (df)

df2 = df2.rename(columns={0: "s_adj", 1: "age", 2: "unit", 3: "sex", 4: "geo-time"}) # Rename from 0,1,2, ...

# Combine expanded column with data

result = pd.concat([df2, df], axis=1)

result = result.drop('s_adj,age,unit,sex,geo\\time', 1)

result.head(3)
# Remove characters from numeric fields, replace : with NaN

cols = list(result.columns[5:-1].values)

rep = ['e','b','p','u','c','f','r','z','d','n','s']

result[cols] = result[cols].replace(rep, '', regex=True) # Replace unwanted char

result[cols] = result[cols].replace(':', np.NaN, regex=True) # Replace : with NaN

result.head(3)
# Convert Country Codes

codes = pd.read_csv('../input/country-codes/country_codes.csv') # Import country codes 

result['geo-time'] = result['geo-time'].map(codes.set_index('Alpha-2 code')['Alpha-3 code']) #Convert 2 char codes to 3 char
# Drop entries from multiple countries (now NaN country name)

result = result[result['geo-time'].notna()] # Remove countries that did not have an alpha-3 code (sub groups of EU countries)



# Keep data from 2019 forward, removing 1983-2018

result = result.drop(result.columns[12:-1].values, 1)

result = result.iloc[:,:-1]

result.head(3)
# Pivot Data

result = result.melt(id_vars=["geo-time", "age", "unit", "sex", "s_adj"], var_name="Date", value_name="Value")

result['Date'] = result['Date'].replace('M', '-', regex=True)

result['Value'] = pd.to_numeric(result['Value'], errors='coerce').fillna(0)

result.head(3)
# Filter Geo Data

geoData = result[ (result['sex']== 'T') 

                 & (result['unit']== 'PC_ACT')

                 & (result['s_adj']== 'SA')

                 & (result['Value'].notna())]



geoData['Value'] = pd.to_numeric(geoData['Value'],errors='coerce')

geoData = geoData.sort_values(by=['geo-time','age','Date'], ascending=True).reset_index(drop=True)
geoData['diff'] = ( (geoData["Value"] - geoData["Value"].shift(1) ) / geoData["Value"]) * 100

geoData.loc[geoData.Date == '2019-12 ', ['diff']] = 0

geoData['diff'] = geoData['diff'].fillna(0)

geoData.loc[geoData['diff'] == float("-inf"), ['diff']] = 0

geoData[geoData['geo-time']== 'BGR']
geoData = geoData[geoData['geo-time']!= 'JPN']

geoData = geoData[geoData['geo-time']!= 'USA']

geoData = geoData[geoData['Date']!= '2019-12 ']



geoData.head(3)
covid_cases = pd.read_csv('../input/covid-cases/covid_cases.csv')

covid_cases = covid_cases.drop(['dateRep','day','deaths','countriesAndTerritories','geoId','popData2019','continentExp','Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'],1)

covid_cases.head(3)
cov_cntry = covid_cases.groupby(['month','year','countryterritoryCode'])['cases'].sum().reset_index()

cov_cntry.head(3)
cov_cntry = cov_cntry[cov_cntry['countryterritoryCode'].isin(geoData['geo-time'])].reset_index(drop=True)

cov_cntry['Date'] = cov_cntry['year'].astype(str) + '-' + cov_cntry['month'].astype(str).str.pad(2,fillchar='0').astype(str) + ' '

cov_cntry = cov_cntry.drop(['month','year'],1)

cov_cntry.head()
fig = make_subplots(

    rows=3, 

    cols=2,

    subplot_titles=("Unemployment Geomap", "% Unemployed", "Perc. Change from Prev Mnth", "COVID-19 Cases per Country"),

    specs=[ [{"type": "scattergeo", "rowspan": 3},{"type": "bar"}],

            [None,{"type": "bar"}],

            [None,{"type": "bar"}] ])



colors = ['teal',] * 60

colors[1::2] = ['thistle' for x in colors[1::2]]



for step in geoData['Date'].unique():

    fig.append_trace(

        go.Choropleth(

            visible=False,

            locations = geoData[ (geoData['Date'] == step) & (geoData['age'] == 'TOTAL') ]['geo-time'],

            z = geoData[ (geoData['Date'] == step) & (geoData['age'] == 'TOTAL')]['Value'],

            colorscale = 'Reds',

            zmax=15.6,

            zmin=0,

            colorbar_title = "Unemp %", 

        ),

        row=1,

        col=1

    )

    

    fig.append_trace(

        go.Bar(

            visible=False,

            x=geoData[ (geoData['Date'] == step) & (geoData['age'] != 'TOTAL')]['geo-time'],

            y=geoData[ (geoData['Date'] == step) & (geoData['age'] != 'TOTAL')]["Value"], 

            marker_color=colors,

            showlegend=False,

        ),

        row=1, 

        col=2

    )



    fig.append_trace(

        go.Bar(

            visible=False,

            x=geoData[(geoData['Date'] == step) & (geoData['age'] != 'TOTAL')]['geo-time'],

            y=geoData[(geoData['Date'] == step) & (geoData['age'] != 'TOTAL')]["diff"], 

            marker_color=colors,

            name='25-74',

        ),

        row=2, 

        col=2

    )

    

    fig.append_trace(

        go.Bar(

            visible=False,

            x = cov_cntry[ cov_cntry['Date'] == step ]['countryterritoryCode'],

            y = cov_cntry[ cov_cntry['Date'] == step ]["cases"], 

            showlegend=False,

            marker_color='yellow',

        ),

        row=3, 

        col=2

    )

 

fig.data[0].visible = True



dates = ['01','02','03','04','05','06']

steps = []

j= 0

num_steps = 6

for i in range(0, len(fig.data), 4):

    step = dict(method="restyle", args=["visible", [False] * len(fig.data)],label='2020- {}'.format(dates[j]))

    j += 1

    step['args'][1][i] = True

    step['args'][1][i+1] = True

    step['args'][1][i+2] = True

    step['args'][1][i+3] = True

    steps.append(step)



sliders = [dict(active=0,pad={"t": 50},steps=steps)]



# Update geo subplot properties

fig.update_geos(

    projection_type="miller",

    fitbounds="locations",

    resolution=50,

    landcolor="silver",

    oceancolor="cadetblue",

    showocean=True,

    showlakes=False,

)



# Set theme, margin, and annotation in layout

fig.update_layout(

    template="plotly_dark",

    height = 800,

    margin=dict(r=10, t=55, b=40, l=20),

    title_text = 'Unemployment % by Country, Jan to June 2020',

    sliders=sliders,

    legend=dict(

        orientation="h",

        yanchor="bottom",

        y=1.02,

        xanchor="right",

        x=1

    ),

)



fig.show()