# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Ploting and visualisations 
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px 
from plotly.offline import download_plotlyjs,init_notebook_mode, iplot
import plotly.tools as tls 
import plotly.figure_factory as ff 
py.init_notebook_mode(connected=True)
# ----------------------- #
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')
display(data.info(),data.head())
def missing_plot(dataset,key):
    null_feat = pd.DataFrame(len(dataset[key]) - dataset.isnull().sum(), columns = ['Count'])
    percentage_null = pd.DataFrame((dataset.isnull().sum())/len(dataset[key])*100, columns = ['Count'])
    percentage_null = percentage_null.round(2)

    trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, text = percentage_null['Count'],  textposition = 'auto'
                   ,marker=dict(color = 'darkorange',line=dict(color='#000000',width=1.5)))

    layout = dict(title =  "Missing Values in dataset in percentage%")

    fig = dict(data = [trace], layout=layout)
    py.iplot(fig)

missing_plot(data,'race')
data.dropna(inplace=True)
data['month'] = pd.to_datetime(data['date']).dt.month
data['year'] = pd.to_datetime(data['date']).dt.year
daily_shootouts = data[['date']]
daily_shootouts['kills'] = 1
daily_shootouts=daily_shootouts.groupby('date').sum()
daily_shootouts = daily_shootouts.reset_index()

def line_plot(data,var):
    
    trace = go.Scatter(x=data[var],y=data['kills'],line=dict(color='darkorange',width=2))
    
    layout = dict(title='Deaths per {}'.format(var))
    
    fig = dict(data = [trace], layout=layout)
    py.iplot(fig)
line_plot(daily_shootouts,'date')
daily_shootouts = data[['year']]
daily_shootouts['kills'] = 1
daily_shootouts=daily_shootouts.groupby('year').sum()
daily_shootouts = daily_shootouts.reset_index()

line_plot(daily_shootouts,'year')
daily_shootouts = data[['year','month']]
daily_shootouts['kills'] = 1
fig = px.bar(daily_shootouts, x='month', y='kills',color_discrete_sequence =['darkorange'], 
                facet_col='year')
fig.show()
def target_count(data,column,height):
    trace = go.Bar( x = data[column].value_counts().values.tolist(),
    y = data[column].unique(),
    orientation = 'h',
    text = data[column].value_counts().values.tolist(),
    textfont=dict(size=20),
    textposition = 'auto',
    opacity = 0.5,marker=dict(color='darkorange',
            line=dict(color='#000000',width=1.5))
    )
    layout = (dict(title= "EDA of {} column".format(column),
                  autosize=True,height=height,))
    fig = dict(data = [trace], layout=layout)
    
    py.iplot(fig)

# --------------- donut chart to show there percentage -------------------- # 

def target_pie(data,column):
    trace = go.Pie(labels=data[column].unique(),values=data[column].value_counts(),
                  textfont=dict(size=15),
                   opacity = 0.5,marker=dict(
                   colorssrc='tealrose',line=dict(color='#000000', width=1.5)),
                   hole=0.6)
                  
    layout = dict(title="Dounat chart to see %age ")
    fig = dict(data=[trace],layout=layout)
    py.iplot(fig)
    
target_count(data,'race',400)
target_pie(data,'race')
def density_plot(var):
    M = data[data['gender'] == "M"] # Data of all the males only 
    F = data[data['gender'] == "F"] # Data of all the females only 
    
    hist_data = [F[var],M[var]]
    labels = ["F","M"]
    color = ['pink','skyblue']
    fig = ff.create_distplot(hist_data,labels,colors = color,show_hist=True,curve_type='kde')
    fig['layout'].update(title = 'Most affected age group with gender wise segregation')
    
    py.iplot(fig, filename = 'Density Plot')
density_plot('age')
target_count(data,'manner_of_death',300)
target_pie(data,'manner_of_death')
target_count(data,'armed',700)
target_pie(data,'armed')
shootout_by_states = data['state'].value_counts()
shootout_by_states = pd.DataFrame(shootout_by_states)
shootout_by_states=shootout_by_states.reset_index()
fig = px.pie(shootout_by_states, values='state', names='index', color_discrete_sequence=px.colors.qualitative.Vivid,hole=0.5)
fig.show()
daily_shootouts = data[['threat_level','signs_of_mental_illness','armed']]
daily_shootouts['kills'] = 1

daily_shootouts['armed'] = daily_shootouts['armed'].replace(['hammer','hatchet', 'undetermined', 'sword', 'machete', 'box cutter', 'metal object', 'screwdriver', 'lawn mower blade', 'flagpole', 'guns and explosives', 'cordless drill', 'metal pole', 'Taser', 'metal pipe', 'metal hand tool', 'blunt object', 'metal stick', 'sharp object', 'meat cleaver', 'carjack', 'chain', "contractor's level", 'unknown weapon', 'stapler', 'crossbow', 'bean-bag gun', 'baseball bat and fireplace poker', 'straight edge razor', 'gun and knife', 'ax', 'brick', 'baseball bat', 'hand torch', 'chain saw', 'garden tool', 'scissors', 'pole', 'pick-axe', 'flashlight', 'vehicle', 'spear', 'chair', 'pitchfork', 'hatchet and gun', 'rock', 'piece of wood', 'bayonet', 'glass shard', 'motorcycle', 'pepper spray', 'metal rake', 'baton', 'crowbar', 'oar', 'machete and gun', 'air conditioner', 'pole and knife', 'beer bottle', 'pipe', 'baseball bat and bottle', 'fireworks', 'pen', 'chainsaw', 'gun and sword', 'gun and car', 'pellet gun', 'BB gun', 'incendiary device', 'samurai sword', 'bow and arrow', 'gun and vehicle', 'vehicle and gun', 'wrench', 'walking stick', 'barstool', 'grenade', 'BB gun and vehicle', 'wasp spray', 'air pistol', 'baseball bat and knife', 'vehicle and machete', 'ice pick', 'car, knife and mace']
,'other')
fig = px.bar(daily_shootouts, y='armed', x='kills',facet_col='threat_level',facet_row='signs_of_mental_illness',color='armed')
fig.update_traces(marker_color=['rgb(00, 83, 109)'],marker_line_color='darkorange',
                 marker_line_width=1.5, opacity=0.6)
fig.show()

