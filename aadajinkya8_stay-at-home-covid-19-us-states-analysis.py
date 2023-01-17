import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

pd.set_option('display.max_rows', None)

from plotly.subplots import make_subplots

import seaborn as sns

from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets

import datetime

from scipy import stats

import operator

import plotly.figure_factory as ff

from scipy.signal import savgol_filter
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data.head()
data.info()
data['Province/State'] = data['Province/State'].fillna('Not Known')

data.info()
data = data.rename(columns={"Province/State":"State","Country/Region": "Country"})

data["Active"] = data["Confirmed"]-data["Recovered"]-data["Deaths"]
data['Date'] = data['ObservationDate'].apply(pd.to_datetime)

data.drop(['SNo'],axis=1,inplace=True)
data[["Confirmed","Deaths","Recovered", "Active"]] =data[["Confirmed","Deaths","Recovered","Active"]].astype(int)

data.head()
data_usa= data[(data['Country'] == 'US')].reset_index(drop=True)
latest_date = data_usa[data_usa['ObservationDate'] == max(data_usa['ObservationDate'])].reset_index()

world = latest_date.groupby(["ObservationDate"])[["Confirmed","Active","Recovered","Deaths"]].sum().reset_index()
val = [world.iloc[0][2], world.iloc[0][3], world.iloc[0][4]]



fig = go.Figure(data=[go.Pie(values = val, labels = ['Active', 'Recovered','Deaths'])])



fig.update_traces(hole=.4)



fig.update_layout(

    title='Situation of Coronavirus in the US as of {} : '.format(world.iloc[0][0]),

    annotations=[dict(text='Total : {0}'.format(world.iloc[0][1]), x=0.5, y=0.5, font_size=15, showarrow=False)]

)

fig.show()
data_usa['State'].unique()
code_to_state = {'AL': 'Alabama',

 'AK': 'Alaska',

 'AS': 'American Samoa',

 'AZ': 'Arizona',

 'AR': 'Arkansas',

 'CA': 'California',

 'CO': 'Colorado',

 'CT': 'Connecticut',

 'DE': 'Delaware',

 'DC': 'District of Columbia',

 'D.C.': 'District of Columbia',

 'FM': 'Federated States of Micronesia',

 'FL': 'Florida',

 'GA': 'Georgia',

 'GU': 'Guam',

 'HI': 'Hawaii',

 'ID': 'Idaho',

 'IL': 'Illinois',

 'IN': 'Indiana',

 'IA': 'Iowa',

 'KS': 'Kansas',

 'KY': 'Kentucky',

 'LA': 'Louisiana',

 'ME': 'Maine',

 'MH': 'Marshall Islands',

 'MD': 'Maryland',

 'MA': 'Massachusetts',

 'MI': 'Michigan',

 'MN': 'Minnesota',

 'MS': 'Mississippi',

 'MO': 'Missouri',

 'MT': 'Montana',

 'NE': 'Nebraska',

 'NV': 'Nevada',

 'NH': 'New Hampshire',

 'NJ': 'New Jersey',

 'NM': 'New Mexico',

 'NY': 'New York',

 'NC': 'North Carolina',

 'ND': 'North Dakota',

 'MP': 'Northern Mariana Islands',

 'OH': 'Ohio',

 'OK': 'Oklahoma',

 'OR': 'Oregon',

 'PW': 'Palau',

 'PA': 'Pennsylvania',

 'PR': 'Puerto Rico',

 'RI': 'Rhode Island',

 'SC': 'South Carolina',

 'SD': 'South Dakota',

 'TN': 'Tennessee',

 'TX': 'Texas',

 'UT': 'Utah',

 'VT': 'Vermont',

 'VI': 'Virgin Islands',

 'VA': 'Virginia',

 'WA': 'Washington',

 'WV': 'West Virginia',

 'WI': 'Wisconsin',

 'WY': 'Wyoming'}



def agg_state(x):

    try:

        return code_to_state[x.split(",")[-1].strip()]

    except:

        return x.strip()

    

data_usa['State'] = data_usa['State'].apply(agg_state)

data_usa['State'] = data_usa['State'].replace('United States Virgin Islands', 'Virgin Islands')

data_usa['State'] = data_usa['State'].replace('Virgin Islands, U.S.', 'Virgin Islands')

data_usa['State'] = data_usa['State'].replace('Omaha, NE (From Diamond Princess)', 'Nebraska')

data_usa['State'] = data_usa['State'].replace('Travis, CA (From Diamond Princess)', 'California')

data_usa['State'] = data_usa['State'].replace('Lackland, TX (From Diamond Princess)', 'Texas')

data_usa['State'] = data_usa['State'].replace('Unassigned Location (From Diamond Princess)', 'Others')

data_usa['State'] = data_usa['State'].replace('Chicago', 'Illinois')

data_usa['State'].unique()
data_agg = data_usa.groupby(["ObservationDate"])[["Confirmed","Active","Recovered","Deaths"]].sum().reset_index()
data_agg['New'] = data_agg['Active'].diff().mask(data_agg['Active'].diff().lt(0,0))

data_agg['New_death'] = data_agg['Deaths'].diff().mask(data_agg['Deaths'].diff().lt(0,0))

data_agg = data_agg.fillna(0)

#print(data_agg.tail())

fig = go.Figure()

fig.add_trace(go.Bar(x=data_agg["ObservationDate"], y = data_agg["New"],name='Distribution of New Cases', marker_color = 'purple'))

yhat = savgol_filter(data_agg['New'], 31, 4) # window size 51, polynomial order 3

fig.add_trace(go.Scatter(x=data_agg['ObservationDate'], y=yhat,

                        marker_color='blue',

                        mode="lines+text",

                        name='Trend in New Cases',      

                        ))

fig.add_trace(go.Bar(x=data_agg["ObservationDate"], y = data_agg["New_death"], name='Distribution of New Deaths', marker_color = 'red'))





fig.update_layout(

    title_text='New cases in the US over time', # title of plot

    xaxis_title_text='Date', # xaxis label

    yaxis_title_text='Count of New Cases', # yaxis label

    bargap=0.2, # gap between bars of adjacent location coordinates

    bargroupgap=0.1,

    barmode = 'overlay',# gap between bars of the same location coordinates

    legend=dict(x=0.7, y=1.3),

    width=700,

    height=480,

)

fig.show()
def get_data(list_state):

    tab_data = {}

    count = 0

    data1 = data_usa[data_usa['State'] == 'Washington']

    data1 = data1.groupby(["ObservationDate"])[["Confirmed","Active","Recovered","Deaths"]].sum().reset_index()

    for state in list_state:

        data_state = data_usa[data_usa['State'] == state]

        data_state = data_state.groupby(["ObservationDate"])[["Confirmed","Active","Recovered","Deaths"]].sum().reset_index()

        lis1 = data_state.iloc[:, 0].tolist()

        lis2 = data_state.iloc[:, 2].tolist()

        dic = {'Date': lis1, state: lis2}

        output = pd.DataFrame.from_dict(dic, orient = 'index') 

        count +=1

        if count == 1:

            res = output.transpose()

        if count > 1:

            output = output.transpose()

            res = res.merge(output, on = 'Date', how = 'left')

    return res
states = ['Arkansas', 'Iowa', 'Nebraska', 'North Dakota', 'South Dakota', 'Oklahoma', 'Utah', 'Wyoming','Diamond Princess cruise ship','US', 'Wuhan Evacuee', 'Recovered', 'Others','Grand Princess Cruise Ship', 'American Samoa',

        'Guam', 'Northern Mariana Islands' , 'Puerto Rico' , 'Virgin Islands']

data_final = data_usa[~data_usa.isin(states)]
state_list = data_final.State.unique().tolist()

state_list.pop(7)
result = get_data(state_list)

result = result.set_index('Date')

result = result.drop(['Grand Princess'], axis = 1)

result = result.fillna(0)

result.head()
order = pd.read_csv('../input/data-1/date_data.csv')



order['Start Date'] = pd.to_datetime(order['Start Date'])

order['End Date'] = pd.to_datetime(order['End Date'])



order['Start Date'] = order['Start Date'].dt.strftime("%m/%d/%Y")

order['End Date'] = order['End Date'].dt.strftime("%m/%d/%Y")



order = order.reset_index(drop = True)

order = order.fillna(-1)
def multi_plot(df):

    fig=go.Figure()

    region_plot_names = []

    buttons=[]

    df1 = df.reset_index()

    default_state = 'Washington'

    for region_name in df.columns:

        start_date = order[order['State_Name'] == region_name]

        start_date = start_date.iloc[:,1].tolist()



        end_date = order[order['State_Name'] == region_name]

        end_date = end_date.iloc[:,2].tolist()



        fig.add_trace(go.Scatter(x = df1['Date'], y = df1[region_name].diff().mask(df1[region_name].diff().lt(0),0), visible=(region_name==default_state), marker_color='#ffc400',

                        mode="lines+text",

                        name='Daily New cases'))

        

        fig.add_trace(go.Scatter(x = [start_date[0], start_date[0]], y = [0, df1[region_name].diff().max()],name = 'Stay at home order issued', visible=(region_name==default_state)))

        if end_date[0] == -1:

            region_plot_names.extend([region_name]*2)

        else:

            fig.add_trace(go.Scatter(x = [end_date[0], end_date[0]], y = [0, df1[region_name].diff().max()], visible=(region_name==default_state), name = 'Stay at home order lifted'))    

            region_plot_names.extend([region_name]*3)

    

    

    for region_name in df.columns:

        buttons.append(dict(method='update',

                        label=region_name,

                        args = [{'visible': [region_name==r for r in region_plot_names]}]))



# Add dropdown menus to the figure

    fig.update_layout(showlegend=True, updatemenus=[{"buttons": buttons, "direction": "down",  "showactive": True, "x": 0.63, "y": 1.13}], title='Evolution of new cases over time', legend=dict(x=0.7, y=1.3),

                     width=700,

    height=480)

    fig.show()

multi_plot(result)
trend_dic = {}

for state in result.columns:

    trend = result.reset_index()

    trend.Date =pd.to_datetime(trend.Date)

    trend['date_ordinal'] = pd.to_datetime(trend['Date']).map(datetime.datetime.toordinal)

    slope, intercept, r_value, p_value, std_err = stats.linregress(trend['date_ordinal'].tail(30), trend[state].diff().tail(30))

    trend_dic[state] = slope
c_sorted_trend = sorted(trend_dic.items(), key=operator.itemgetter(0))

df1 = pd.DataFrame(c_sorted_trend, columns = ['State', 'Confirmed Trend Value'])

order = order.merge(df1, left_on = 'State_Name', right_on = 'State', how = 'left')

order.iloc[:, 8] *= 100

order = order.round(2)
fig = px.choropleth(order,

    locations='State_Code', # Spatial coordinates

    color = 'Trend in New Cases', # Data to be color-coded

    locationmode = 'USA-states',# set of locations match entries in `locations`

    hover_data = ['Confirmed Trend Value', 'State_Name']

)



fig.update_layout(

    geo_scope='usa', # limite map scope to USA

    title='Trend in New Confirmed cases post lifting stay at home orders in US (Hotspots)',

    width=700,

    height=480,

)



fig.show()
sorted_trend = sorted(trend_dic.items(), key=operator.itemgetter(1), reverse=True)

lis1 = []

lis2 = []

for x in sorted_trend[:11]:

    lis1.append(x[0])

    lis2.append(x[1].astype(int) * 100)

plot = dict(

    number=lis2,

    State=lis1)

fig = px.funnel(plot, x='number', y='State')

fig.update_layout(

    title='Top 10 states with highest increase in new cases (% increase)',

    width=700,

    height=480,

)

fig.show()
def get_deaths_data(list_state):

    tab_data = {}

    count = 0

    data1 = data_usa[data_usa['State'] == 'Washington']

    data1 = data1.groupby(["ObservationDate"])[["Confirmed","Active","Recovered","Deaths"]].sum().reset_index()

    for state in list_state:

        data_state = data_usa[data_usa['State'] == state]

        data_state = data_state.groupby(["ObservationDate"])[["Confirmed","Active","Recovered","Deaths"]].sum().reset_index()

        lis1 = data_state.iloc[:, 0].tolist()

        lis2 = data_state.iloc[:, 4].tolist()

        dic = {'Date': lis1, state: lis2}

        output = pd.DataFrame.from_dict(dic, orient = 'index') 

        count +=1

        if count == 1:

            res = output.transpose()

        if count > 1:

            output = output.transpose()

            res = res.merge(output, on = 'Date', how = 'left')

    return res
deaths = get_deaths_data(state_list)

deaths = deaths.set_index('Date')

deaths =deaths.drop(['Grand Princess'], axis = 1)

deaths = deaths.fillna(0)

deaths.head()
def multi_plot_death(df):

    fig=go.Figure()

    region_plot_names = []

    buttons=[]

    df1 = df.reset_index()

    default_state = 'Washington'

    for region_name in df.columns:

        start_date = order[order['State_Name'] == region_name]

        start_date = start_date.iloc[:,1].tolist()



        end_date = order[order['State_Name'] == region_name]

        end_date = end_date.iloc[:,2].tolist()



        fig.add_trace(go.Scatter(x = df1['Date'], y = df1[region_name].diff().mask(df1[region_name].diff().lt(0),0), visible=(region_name==default_state), marker_color='#ffc400',

                        mode="lines+text",

                        name='Daily New cases'))

        

        fig.add_trace(go.Scatter(x = [start_date[0], start_date[0]], y = [0, df1[region_name].diff().max()],name = 'Stay at home order issued', visible=(region_name==default_state)))

        if end_date[0] == -1:

            region_plot_names.extend([region_name]*2)

        else:

            fig.add_trace(go.Scatter(x = [end_date[0], end_date[0]], y = [0, df1[region_name].diff().max()], visible=(region_name==default_state), name = 'Stay at home order lifted'))    

            region_plot_names.extend([region_name]*3)

    

    

    for region_name in df.columns:

        buttons.append(dict(method='update',

                        label=region_name,

                        args = [{'visible': [region_name==r for r in region_plot_names]}]))



# Add dropdown menus to the figure

    fig.update_layout(showlegend=True, updatemenus=[{"buttons": buttons, "direction": "down",  "showactive": True, "x": 0.6, "y": 1.13}], title='Evolution of new death cases over time', legend=dict(x=0.7, y=1.3), width=700,

    height=480,)

    fig.show()

multi_plot_death(deaths)
deaths_trend_dic = {}

for state in deaths.columns:

    death_trend = deaths.reset_index()

    death_trend.Date =pd.to_datetime(death_trend.Date)

    death_trend['date_ordinal'] = pd.to_datetime(death_trend['Date']).map(datetime.datetime.toordinal)

    slope, intercept, r_value, p_value, std_err = stats.linregress(death_trend['date_ordinal'].tail(30), death_trend[state].diff().tail(30))

    deaths_trend_dic[state] = slope
d_sorted_trend = sorted(deaths_trend_dic.items(), key=operator.itemgetter(0))

df = pd.DataFrame(d_sorted_trend, columns = ['State', 'Trend Value'])

order = order.merge(df, left_on = 'State_Name', right_on = 'State', how = 'left')

order.iloc[:, 9] *= 100

order = order.round(2)
fig = px.choropleth(order,

    locations='State_Code', # Spatial coordinates

    color = 'Trend in New death Cases', # Data to be color-coded

    locationmode = 'USA-states',# set of locations match entries in `locations`

    hover_data=['Trend Value', 'State_Name'],

)



fig.update_layout(

    geo_scope='usa', # limite map scope to USA

    title='Trend in New death cases post lifting stay at home orders in US',

    width=700,

    height=480,

)



fig.show() 
death_sorted_trend = sorted(deaths_trend_dic.items(), key=operator.itemgetter(1), reverse=True)

lis1 = []

lis2 = []

for x in death_sorted_trend[:11]:

    lis1.append(x[0])

    lis2.append(round(x[1]*100,2))

plot = dict(

    number=lis2,

    State=lis1)

fig = px.funnel(plot, x='number', y='State')

fig.update_layout(

    title='Top 10 states with highest increase in new death cases (% increase)',

    width=700,

    height=480,

)

fig.show()
fig = px.choropleth(order,

    locations='State_Code', # Spatial coordinates

    color = 'Face mask requirement', # Data to be color-coded

    locationmode = 'USA-states',

    hover_data = ['State_Name']

)



fig.update_layout(

    geo_scope='usa', # limite map scope to USA

    title='Face Mask requirement across states',

    width=700,

    height=480,

)



fig.show()