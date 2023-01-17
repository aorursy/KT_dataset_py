!pip install -q pywaffle
#importing plotting libraries

import pandas as pd

import numpy as np

import plotly.graph_objs as go

import plotly.express as px

from plotly.subplots import make_subplots

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

from pywaffle import Waffle

import matplotlib.pyplot as plt

init_notebook_mode(connected=False)



#importing modeling libraries

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')

df.drop(['Unnamed: 0.1','Unnamed: 0'], axis = 1, inplace = True)

df.head()
#function to extract the name of the country from the location

def extract_country_name(location):

    country = location.split(',')[-1]

    country = country.strip()

    return country



#dictionary to help in mapping to get consistent and correct Country Names

countries_dict = {

    'Russia' : 'Russian Federation',

    'New Mexico' : 'USA',

    "Yellow Sea": 'China',

    "Shahrud Missile Test Site": "Iran",

    "Pacific Missile Range Facility": 'USA',

    "Barents Sea": 'Russian Federation',

    "Gran Canaria": 'USA'

}
df['Country'] = df['Location'].apply(lambda x: extract_country_name(x))

df['Country'] = df['Country'].replace(countries_dict)
#extracting date-time features

df['Datum'] = pd.to_datetime(df['Datum'])

df['year'] = df['Datum'].apply(lambda datetime: datetime.year)

df['month'] = df['Datum'].apply(lambda datetime: datetime.month)

df['weekday'] = df['Datum'].apply(lambda datetime: datetime.weekday())
#function to get the Launch Vehicle Name from the Details

def getVehicles(detail):

    lv = []

    li = [x.strip() for x in detail.split('|')] #extracting the name of all launch vehicles from the Details section

    for ele in li:

        if('Cosmos' in ele):

            lv.append('Cosmos')

        elif('Vostok' in ele):

            lv.append('Vostok')

        elif('Tsyklon' in ele):

            lv.append('Tsyklon')

        elif('Ariane' in ele):

            lv.append('Ariane')

        elif('Atlas' in ele):

            lv.append('Atlas')

        elif('Soyuz' in ele):

            lv.append('Soyuz')

        elif('Delta' in ele):

            lv.append('Delta')

        elif('Titan' in ele):

            lv.append('Titan')

        elif('Molniya' in ele):

            lv.append('Molniya')

        elif('Zenit' in ele):

            lv.append('Zenit')

        elif('Falcon' in ele):

            lv.append('Falcon')

        elif('Long March' in ele):

            lv.append('Long March')

        elif('PSLV' in ele):

            lv.append('PSLV')

        elif('GSLV' in ele):

            lv.append('GSLV')

        elif('Thor' in ele):

            lv.append('Thor')

        else:

            lv.append('Other')

    return lv

df['Launch Vehicles'] = df['Detail'].apply(lambda x:getVehicles(x))
#creating a waffle Chart using pywaffle

plt.rcParams['figure.figsize'] = (7,12)

data = dict(df['Status Mission'].value_counts(normalize = True) * 100)

fig = plt.figure(

    FigureClass=Waffle, 

    columns=10, 

    values=data, 

    colors=("#3bff3b", "#ff3b3b", "#ffff3b","#ff9d3b"),

    title={'label': 'Status of Space Missions', 'loc': 'center'},

    icons = 'rocket',

    icon_size = 20,

    labels=[f"{k} ({v:.2f}%)" for k, v in data.items()],

    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(data), 'framealpha': 0.3}

)

plt.show()
country_counts = dict(df['Country'].value_counts())

fig = go.Figure(data=[go.Table(

    header=dict(values=['<b>Country Name</b>', '<b>Number of Space Missions</b>'],

                line_color='black',

                fill_color='darkorange',

                align='left',

                font=dict(color='black', size=14)),

    cells=dict(values=[list(country_counts.keys()),

                      list(country_counts.values())],

               line_color='black',

               fill_color='white',

               align='left',

               font=dict(color='black', size=13)))

])



fig.update_layout(width=500, height=450,margin=dict(l=80, r=80, t=25, b=10),

                  title = { 'text' : '<b>Number of Space Missions Per Launch Location</b>', 'x' : 0.95},

                 font_family = 'Fira Code',title_font_color= '#ff0d00')

fig.show()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(df['Status Mission'])

colors = {0 : 'red', 1 : 'Orange', 2 : 'Yellow', 3 : 'Green'}
fig = make_subplots(rows = 4 ,cols = 4,subplot_titles=df['Country'].unique())

for i, country in enumerate(df['Country'].unique()):

    counts = df[df['Country'] == country]['Status Mission'].value_counts(normalize = True) * 100

    color = [colors[x] for x in encoder.transform(counts.index)]

    trace = go.Bar(x = counts.index, y = counts.values, name = country,showlegend=False,marker={'color' : color})

    fig.add_trace(trace, row = (i//4)+1, col = (i%4)+1)

fig.update_layout(template = 'plotly_dark',margin=dict(l=80, r=80, t=50, b=10),

                  title = { 'text' : '<b>Countries and Mission Status</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#cacaca',height = 1000,width = 1100)

for i in range(1,5):

    fig.update_yaxes(title_text = 'Percentage',row = i, col = 1)

fig.show()
fig = px.sunburst(df,path = ['Status Mission','Country'])

fig.update_layout(margin=dict(l=80, r=80, t=25, b=10),

                  title = { 'text' : '<b>Countries and Mission Status</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#8000ff')

fig.show()
successPerc = df[df['Status Mission'] == 'Success'].groupby('Company Name')['Status Mission'].count()

for company in successPerc.index:

    successPerc[company] = (successPerc[company] / len(df[df['Company Name'] == company]))*100

successPerc = successPerc.sort_index()

FailurePerc = df[df['Status Mission'] == 'Failure'].groupby('Company Name')['Status Mission'].count()

for company in FailurePerc.index:

    FailurePerc[company] = (FailurePerc[company] / len(df[df['Company Name'] == company]))*100

FailurePerc = FailurePerc.sort_index()
trace1 = go.Bar(x = successPerc.index, y = successPerc.values, name = 'Success Rate of Companies',opacity=0.7)

trace2 = go.Bar(x = FailurePerc.index, y = FailurePerc.values, name = 'Failure Rate of Companies',opacity=0.7)

fig = go.Figure([trace1,trace2])

fig.update_layout(template = 'plotly_white',margin=dict(l=80, r=80, t=25, b=10),

                  title = {'text' : '<b>Success and Failure Rates of Companies</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#8000ff',width = 1000,yaxis_title = '<b>Percentage</b>',xaxis_title = '<b>Companies</b>',

                 legend=dict(

                    yanchor="top",

                    y=0.99,

                    xanchor="left",

                    x=0.01

))



fig.show()
fig = px.treemap(df,path = ['Status Mission','Country','Company Name'])

fig.update_layout(template = 'ggplot2',margin=dict(l=80, r=80, t=50, b=10),

                  title = { 'text' : '<b>Mission Status,Countries and Companies</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#ff6767')

fig.show()
# creating a single list containing the names of the Launch Vehicles

details = []

for detail in df.Detail.values:

    d = [x.strip() for x in detail.split('|')]

    for ele in d:

        if('Cosmos' in ele):

            details.append('Cosmos')

        elif('Vostok' in ele):

            details.append('Vostok')

        elif('Tsyklon' in ele):

            details.append('Tsyklon')

        elif('Ariane' in ele):

            details.append('Ariane')

        elif('Atlas' in ele):

            details.append('Atlas')

        elif('Soyuz' in ele):

            details.append('Soyuz')

        elif('Delta' in ele):

            details.append('Delta')

        elif('Titan' in ele):

            details.append('Titan')

        elif('Molniya' in ele):

            details.append('Molniya')

        elif('Zenit' in ele):

            details.append('Zenit')

        elif('Falcon' in ele):

            details.append('Falcon')

        elif('Long March' in ele):

            details.append('Long March')

        elif('PSLV' in ele):

            details.append('PSLV')

        elif('GSLV' in ele):

            details.append('GSLV')

        elif('Thor' in ele):

            details.append('Thor')

        else:

            details.append('Other')
counts = dict(pd.Series(details).value_counts(sort = True))

fig = go.Figure(go.Bar(x = list(counts.keys()), y = list(counts.values())))

fig.update_layout(template = 'ggplot2',margin=dict(l=80, r=80, t=50, b=10),

                  title = { 'text' : '<b>Number of Missions in each type of Launch Vehicle</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#ff3434',

                 yaxis_title = '<b>Number of Missions</b>',xaxis_title = '<b>Launch Vehicle</b>',)

fig.show()
fig = make_subplots(rows = 3, cols = 1)

for i, period in enumerate(['year', 'month', 'weekday']):

    data = df[df['Status Mission'] == 'Failure'][period].value_counts().sort_index()

    data = dict((data / df[period].value_counts().sort_index())*100.0)

    mean = sum(data.values()) / len(data)

    if(period == 'year'):

        x = list(data.keys())

    elif(period == 'month'):

        x = ['January', 'February', 'March', 'April', 'May','June', 'July', 'August','September','October', 'November', 'December']

    else:

        x = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']

    trace1 = go.Scatter(x = x, y = list(data.values()),mode = 'lines',text = list(data.keys()),name = f'Failures in each {period}',connectgaps = False)

    trace2 = go.Scatter(x = x, y = [mean]*len(data), mode = 'lines',showlegend=False,name = f'Mean failures over the {period}s',line = {'dash':'dash','color':

                                                                                                                                       'grey'})

    fig.append_trace(trace1, row = i+1, col = 1)

    fig.append_trace(trace2, row = i+1, col = 1)

fig.update_layout(template = 'simple_white',height = 600,

                  title = { 'text' : '<b>Failed Missions as a percentage of total missions in that period</b>', 'x' : 0.5})

for i in range(1,4):

    fig.update_yaxes(title_text = '<b>Percentage</b>',row = i, col = 1)

fig.show()
df[' Rocket'] = df[' Rocket'].apply(lambda x: str(x).replace(',',''))

df[' Rocket'] = df[' Rocket'].astype('float64')

df[' Rocket'] = df[' Rocket'].fillna(0)
costDict = dict(df[df[' Rocket'] > 0].groupby('year')[' Rocket'].mean())

fig = go.Figure(go.Scatter(x = list(costDict.keys()), y = list(costDict.values()), yaxis = 'y2',mode = 'lines',showlegend=False,name = 'Average Mission Cost Over the years'))

fig.update_layout(template = 'plotly_dark',margin=dict(l=80, r=80, t=50, b=10),

                  title = { 'text' : '<b>Average Mission Cost Over the years</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#cacaca',

                 yaxis_title = '<b>Cost of Mission in Million Dollars</b>',xaxis_title = '<b>Year of Launch</b>',)

fig.show()
fig = px.scatter(df[df[' Rocket'].between(1,4999)],x = 'year', y = 'Country', color = 'Status Mission',size = ' Rocket', size_max=30)

fig.update_layout(template = 'simple_white',margin=dict(l=80, r=80, t=50, b=10),

                  title = { 'text' : '<b>Average Mission Cost Over the years For Various Countries</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#00b300')

fig.show()
fig = px.scatter(df[df[' Rocket'].between(1,4999)],x = 'year', y = 'Company Name',color = 'Status Mission',size = ' Rocket',size_max = 30)

fig.update_layout(template = 'simple_white',margin=dict(l=80, r=80, t=50, b=10),

                  title = { 'text' : '<b>Average Mission Cost Over the years For Various Companies</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#00b300',height = 650)

fig.show()
df['Target'] = (~(df['Status Mission'] == 'Success')).astype('int32')
X = df[['Company Name',' Rocket','Country', 'year', 'month', 'weekday']]

encoder = LabelEncoder()

X.loc[:,'Company Name'] = encoder.fit_transform(X['Company Name'])

X.loc[:,'Country'] = encoder.fit_transform(X['Country'])

y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0,stratify = y)
classifier_xgb = xgb.XGBClassifier(random_state = 0, n_jobs = -1,max_depth = 5)

classifier_xgb.fit(X_train,y_train)
print(f'Training Accuracy is {accuracy_score(y_train, classifier_xgb.predict(X_train))*100:.2f}%')

print(f'Testing Accuracy is {accuracy_score(y_test, classifier_xgb.predict(X_test))*100:.2f}%')
feature_importance_xgb = classifier_xgb.get_booster().get_fscore()
trace = go.Bar(x = list(feature_importance_xgb.values()), y = list(feature_importance_xgb.keys()),orientation='h')

fig = go.Figure([trace])

fig.update_layout(template = 'simple_white',margin=dict(l=80, r=80, t=50, b=10),

                  title = { 'text' : '<b>Feature Importance</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#00b300',

                 yaxis_title = '<b>Features</b>',xaxis_title = '<b>F Score</b>')



iplot(fig)
classifier_rf = RandomForestClassifier(random_state = 0, n_jobs = -1,max_depth = 5)

classifier_rf.fit(X_train,y_train)
print(f'Training Accuracy is {accuracy_score(y_train, classifier_rf.predict(X_train))*100:.2f}%')

print(f'Testing Accuracy is {accuracy_score(y_test, classifier_rf.predict(X_test))*100:.2f}%')
trace = go.Bar(x = list(classifier_rf.feature_importances_), y = list(X_train.columns),orientation='h')

fig = go.Figure([trace])

fig.update_layout(template = 'simple_white',margin=dict(l=80, r=80, t=50, b=10),

                  title = { 'text' : '<b>Feature Importance</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#00b300',

                 yaxis_title = '<b>Features</b>',xaxis_title = '<b>Gini Importance</b>')



iplot(fig)