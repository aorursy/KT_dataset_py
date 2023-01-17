# IMPORTING LIBRARIES



import os

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from urllib.request import urlopen

import json

from tqdm import tqdm

import warnings 



warnings.filterwarnings(action='ignore')
# load the datasets

train = pd.read_csv(r'/kaggle/input/covid19-global-forecasting-week-4/train.csv')

train['Date'] = pd.to_datetime(train['Date'])



test = pd.read_csv(r'/kaggle/input/covid19-global-forecasting-week-4/test.csv')

test['Date'] = pd.to_datetime(test['Date'])





submission = pd.read_csv(r'/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
submission.head(3)
test.head(3)
train.head()
# Lets take USA data

train_USA = train[train['Country_Region']=='US']



# grouped data sum

train_USA_grouped_sum = train_USA.groupby(by='Province_State').sum()[['ConfirmedCases', 'Fatalities']]

train_USA_grouped_sum = train_USA_grouped_sum.reset_index()

train_USA_grouped_sum['Death_Rate'] = train_USA_grouped_sum['Fatalities']/train_USA_grouped_sum['ConfirmedCases']*100

train_USA_grouped_sum = train_USA_grouped_sum.sort_values(by='Death_Rate', ascending=True)
# CONFIRMED CASES PER EACH DAY CHANGE OVER TIME FOR USA; 

fig = px.line(train_USA, x="Date", y="ConfirmedCases", color="Province_State", title='CONFIRMED CASES PER EACH DAY CHANGE OVER TIME FOR USA')

fig.show()
# NUMBER OF FATALITIES PER EACH DAY OVER TIME FOR USA; 

fig = px.line(train_USA, x="Date", y="Fatalities", color="Province_State", title='NUMBER OF FATALITIES PER EACH DAY OVER TIME FOR USA')

fig.show()
df_USA = train_USA.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()

df_USA['ConfirmedCases_change'] = df_USA['ConfirmedCases'].shift(-1) - df_USA['ConfirmedCases']

df_USA['Fatalities_change'] = df_USA['Fatalities'].shift(-1) - df_USA['Fatalities']



fig = go.Figure(data=[

    go.Bar(name='Cases', x=df_USA['Date'], y=df_USA['ConfirmedCases']),

    go.Bar(name='Deaths', x=df_USA['Date'], y=df_USA['Fatalities'])])



# Change the bar mode

fig.update_layout(barmode='overlay', title='Number of ConfirmedCases and Total Fatalities in USA')

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Cases', x=df_USA['Date'], y=df_USA['ConfirmedCases_change']),

    go.Bar(name='Deaths', x=df_USA['Date'], y=df_USA['Fatalities_change'])])



# Change the bar mode

fig.update_layout(barmode='overlay', title='Number of ConfirmedCases and Total Fatalities Change in USA')

fig.show()
train_USA.head(3)
train_USA['ConfirmedCases_shifted'] = train_USA.groupby(by='Province_State')['ConfirmedCases'].shift(-1)

train_USA['Fatalities_shifted'] = train_USA.groupby(by='Province_State')['Fatalities'].shift(-1)



def calc_percent_change(x1, x2):

    if x1==0:

        return 0

    else:

        return round(abs(x2-x1)/x1*100,2)

    

train_USA['ConfirmedCases_%_change'] = train_USA.apply(lambda x: calc_percent_change(x['ConfirmedCases'], x['ConfirmedCases_shifted']),axis=1 )

train_USA['Fatalities_%_change'] = train_USA.apply(lambda x: calc_percent_change(x['Fatalities'], x['Fatalities_shifted']),axis=1 )



train_USA['ConfirmedCases_delta'] = train_USA['ConfirmedCases_shifted'] - train_USA['ConfirmedCases']

train_USA['Fatalities_delta'] = train_USA['Fatalities_shifted'] - train_USA['Fatalities']

# CONFIRMED CASES CHANGE IN % TIME FOR USA; 

fig = px.line(train_USA, x="Date", y="ConfirmedCases_%_change", color="Province_State", title='CONFIRMED CASES PER EACH DAY  CHANGE OVER TIME IN % FOR USA')

fig.show()
# CONFIRMED CASES DELTA OVER TIME FOR USA; 

fig = px.line(train_USA, x="Date", y="ConfirmedCases_delta", color="Province_State", title='CONFIRMED CASES DELTA FOR USA')

fig.show()
# CONFIRMED CASES DELTA OVER TIME FOR USA; 



for state in train_USA.Province_State.unique().tolist():

    fig = px.line(train_USA[train_USA['Province_State']==state], x="Date", y="ConfirmedCases_delta", color="Province_State", title=state, 

                 width=1000, height=300)

    fig.show()
# FATALITIES % CHANGE OVER TIME FOR USA; 

fig = px.line(train_USA, x="Date", y="Fatalities_%_change", color="Province_State", title='FATALITIES % CHANGE OVER TIME FOR USA')

fig.show()
# FATALITIES %DELTA CHANGE OVER TIME FOR USA; 

fig = px.line(train_USA, x="Date", y="Fatalities_delta", color="Province_State", title='FATALITIES %DELTA CHANGE OVER TIME FOR USA')

fig.show()
train_USA.head(3)
# CREATE PIE CHART FOR TOTAL CONFIRMED CASES AND FATALITIES IN USA 

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])



fig.add_trace(go.Pie(values=train_USA_grouped_sum['ConfirmedCases'], labels=train_USA_grouped_sum['Province_State'], name="ConfirmedCases"), 1, 1)

fig.add_trace(go.Pie(values=train_USA_grouped_sum['Fatalities'], labels=train_USA_grouped_sum['Province_State'], name="Fatalities"), 1, 2)

fig.update_layout(

    title_text="USA Total ConfirmedCases and Fatalities by Province for 2020-04-11")



fig.show()
fig = go.Figure(go.Bar(

            x=train_USA_grouped_sum['Death_Rate'],

            y=train_USA_grouped_sum['Province_State'],

            orientation='h'))

fig.update_layout(

    title_text="USA Death_Rate Horizontal Bar Chart by Province for 2020-04-11")

fig.show()



# Load data frame and tidy it.

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')

df = df[['code', 'state']]

df = df.set_index('state')

df = df.to_dict()['code']

# converter to convert Province_States 





def dict_convert(row, dicti):

    if row in dicti.keys():

        return dicti[row]

    else:

        return None

    

    

train_USA_grouped_sum['Province_State_index'] = train_USA_grouped_sum['Province_State'].apply(lambda x: dict_convert(row=x, dicti=df))



# COVID USA CONFIRMED_CASES MAP

fig = go.Figure(data=go.Choropleth(

    locations=train_USA_grouped_sum['Province_State_index'], # Spatial coordinates

    z = train_USA_grouped_sum['ConfirmedCases'].astype(float), # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale = 'Blackbody',

    colorbar_title = "ConfirmedCases",

))



fig.update_layout(

    title_text = 'Covid USA ConfirmedCases',

    geo_scope='usa', # limite map scope to USA

)



fig.show()
# COVID USA FATALITIES MAP

fig = go.Figure(data=go.Choropleth(

    locations=train_USA_grouped_sum['Province_State_index'], # Spatial coordinates

    z = train_USA_grouped_sum['Fatalities'].astype(float), # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale = 'Blackbody',

    colorbar_title = "Fatalities",

))



fig.update_layout(

    title_text = 'Covid USA Fatalities',

    geo_scope='usa', # limite map scope to USA

)



fig.show()