import numpy as np

import pandas as pd

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

import plotly.graph_objects as go



import warnings



def fxn():

    warnings.warn("deprecated", DeprecationWarning)



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    fxn()
# download data

data = pd.read_csv("../input/usa-cers-dataset/USA_cars_datasets.csv")

data.drop('Unnamed: 0', axis=1, inplace=True)

data_acc = pd.read_csv('../input/us-accidents/US_Accidents_Dec19.csv')
data.head()
data = data[data['country']==' usa']
states = {

        'AK': 'alaska',

        'AL': 'alabama',

        'AR': 'arkansas',

        'AS': 'american samoa',

        'AZ': 'arizona',

        'CA': 'california',

        'CO': 'colorado',

        'CT': 'connecticut',

        'DC': 'district of columbia',

        'DE': 'delaware',

        'FL': 'florida',

        'GA': 'georgia',

        'GU': 'guam',

        'HI': 'hawaii',

        'IA': 'iowa',

        'ID': 'idaho',

        'IL': 'illinois',

        'IN': 'indiana',

        'KS': 'kansas',

        'KY': 'kentucky',

        'LA': 'louisiana',

        'MA': 'massachusetts',

        'MD': 'maryland',

        'ME': 'maine',

        'MI': 'michigan',

        'MN': 'minnesota',

        'MO': 'missouri',

        'MP': 'northern mariana islands',

        'MS': 'mississippi',

        'MT': 'montana',

        'NA': 'national',

        'NC': 'north carolina',

        'ND': 'north dakota',

        'NE': 'nebraska',

        'NH': 'new hampshire',

        'NJ': 'new jersey',

        'NM': 'new mexico',

        'NV': 'nevada',

        'NY': 'new york',

        'OH': 'ohio',

        'OK': 'oklahoma',

        'OR': 'oregon',

        'PA': 'pennsylvania',

        'PR': 'puerto rico',

        'RI': 'rhode island',

        'SC': 'south carolina',

        'SD': 'south dakota',

        'TN': 'tennessee',

        'TX': 'texas',

        'UT': 'utah',

        'VA': 'virginia',

        'VI': 'virgin islands',

        'VT': 'vermont',

        'WA': 'washington',

        'WI': 'wisconsin',

        'WV': 'west virginia',

        'WY': 'wyoming'

}

inv_states = {v: k for k, v in states.items()}
data['state'] = data['state'].apply(lambda x: inv_states[x])



acc_count = pd.DataFrame(data_acc['State'].value_counts())

acc_count.columns = ['acc_counts']
data = data.merge(acc_count, how='inner', right_index=True, left_on='state')

corr_data = data[['state', 'acc_counts']].merge(data['state'].value_counts(), how='inner', right_index=True, left_on='state')

corr_data[['acc_counts','state_y']].corr()
plt.scatter(corr_data['acc_counts'], corr_data['state_y']);

corr_value = 'corr = ' + str(round(corr_data['acc_counts'].corr(corr_data['state_y']), 2))

plt.text(300000, 5, corr_value, fontsize=12);
data_cars = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')



# some filters

data_cars = data_cars[data_cars['price']!=0]

print(data_cars.shape)
data_cars.head()
data_cars['state'] = data_cars['state'].apply(lambda x: x.upper())

corr_data_2 = acc_count.merge(data_cars['state'].value_counts(), how='inner', right_index=True, left_index=True)



plt.scatter(corr_data_2['acc_counts'], corr_data_2['state']);

corr_value = 'corr = ' + str(round(corr_data_2['acc_counts'].corr(corr_data_2['state']), 2))

plt.text(300000, 5, corr_value, fontsize=12);
state_count_acc = pd.value_counts(data_acc['State'])



fig = go.Figure(data=go.Choropleth(

    locations=state_count_acc.index,

    z = state_count_acc.values.astype(float),

    locationmode = 'USA-states',

    colorscale = 'Reds',

    colorbar_title = "Count Accidents",

))



fig.update_layout(

    title_text = 'US Traffic Accident Dataset by State',

    geo_scope='usa',

)



fig.show()
data_sever = data_acc.sample(n=10000)



fig = go.Figure(data=go.Scattergeo(

        locationmode = 'USA-states',

        lon = data_sever['Start_Lng'],

        lat = data_sever['Start_Lat'],

        text = data_sever['City'],

        mode = 'markers',

        marker = dict(

            size = 8,

            opacity = 0.8,

            reversescale = True,

            autocolorscale = False,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            ),

            colorscale = 'Reds',

            cmin = data_sever['Severity'].max(),

        color = data_sever['Severity'],

        cmax = 1,

            colorbar_title="Severity"

        )))



fig.update_layout(

        title = 'Severity of accidents',

        geo = dict(

            scope='usa',

            projection_type='albers usa',

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.7,

            subunitwidth = 0.7

        ),

    )

fig.show()
print('most dangerous -', acc_count.iloc[:5].index.tolist())

print('most safest -', acc_count.iloc[-5:].index.tolist())



dan = data_cars[data_cars['state'].isin(acc_count.iloc[:5].index.tolist())]

saf = data_cars[data_cars['state'].isin(acc_count.iloc[-5:].index.tolist())]
fig = go.Figure()

fig.add_trace(go.Histogram(

    x=dan[dan['price']<1e5]['price'].values,

    histnorm='percent',

    xbins=dict( 

        start=0.0,

        end=80000,

        size=1000

    ),

    name='dangerous'

))

fig.add_trace(go.Histogram(

    x=saf[saf['price']<1e5]['price'].values,

    histnorm='percent',

    name='safety'

))



fig.update_layout(

    title="Price distribution",

    xaxis_title="Price",

    yaxis_title="Count",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)
fig = go.Figure()

fig.add_trace(go.Histogram(

    x=dan[(dan['odometer']<0.5e6) & (dan['odometer']!=0)]['odometer'].values,

    histnorm='percent',

    xbins=dict( 

        start=0.0,

        end=4e5,

        size=5000

    ),

    name='dangerous'

))

fig.add_trace(go.Histogram(

    x=saf[(saf['odometer']<0.5e6) & (saf['odometer']!=0)]['odometer'].values,

    histnorm='percent',

    name='safety'

))



fig.update_layout(

    title="Odometer distribution",

    xaxis_title="Odometer",

    yaxis_title="Count",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)
fig = go.Figure(go.Histogram(

    y=dan['manufacturer'],

    name='dangerous',

    histnorm='percent',

    bingroup=1))



fig.add_trace(go.Histogram(

    y=saf['manufacturer'],

    name='safety', 

    histnorm='percent',

    bingroup=1))



fig.update_layout(

    title="Manufacturer distribution",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)



fig.show()
fig = go.Figure(go.Histogram(

    y=dan['model'],

    name='dangerous',

    bingroup=1))



fig.add_trace(go.Histogram(

    y=saf['model'],

    name='safety', 

    bingroup=1))



fig.update_layout(

    title="Models distribution",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)



fig.show()
fig = go.Figure(go.Histogram(

    y=dan['condition'],

    name='dangerous',

    histnorm='percent',

    bingroup=1))



fig.add_trace(go.Histogram(

    y=saf['condition'],

    name='safety', 

    histnorm='percent',

    bingroup=1))



fig.update_layout(

    title="Condition distribution",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)



fig.show()
fig = go.Figure(go.Histogram(

    y=dan['type'],

    name='dangerous',

    histnorm='percent',

    bingroup=1))



fig.add_trace(go.Histogram(

    y=saf['type'],

    name='safety', 

    histnorm='percent',

    bingroup=1))



fig.update_layout(

    title="Type distribution",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)



fig.show()
fig = go.Figure(go.Histogram(

    y=dan['fuel'],

    name='dangerous',

    histnorm='percent',

    bingroup=1))



fig.add_trace(go.Histogram(

    y=saf['fuel'],

    name='safety', 

    histnorm='percent',

    bingroup=1))



fig.update_layout(

    title="Fuel distribution",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)



fig.show()