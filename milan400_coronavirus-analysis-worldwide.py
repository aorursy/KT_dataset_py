#Import required libraries



import pandas as pd

import numpy as np

from pandas_profiling import ProfileReport





from plotly.offline import iplot, init_notebook_mode

import plotly.express as px

import plotly.graph_objects as go



import cufflinks as cf

cf.go_offline()

import os





import matplotlib.pyplot as plt

import seaborn as sns
full_data = pd.read_csv('/kaggle/input/alldatas/full_data.csv')



full_data.head()
full_data = full_data.fillna(0)

full_data_animation = full_data.copy()



full_data.head()
from datetime import datetime



full_data = full_data.astype({

    'location':str, 

    'new_cases':int, 

    'new_deaths':int,

    'total_cases':int,

    'total_deaths':int,

    'weekly_cases':int,

    'weekly_deaths':int,

    'biweekly_cases':int,

    'biweekly_deaths':int

})



full_data.date = pd.to_datetime(full_data.date)
# Extracting required features and processing time



full_data_total = full_data[['date','location','total_cases', 'total_deaths']]



latest_time = pd.to_datetime('2020-10-11 00:00:00')



def return_nepal(date):

    if(date == latest_time):

        return True

    else:

        return False

    

def return_noworld(location):

    if(location!='World'):

        return True

    else:

        return False

    

full_data_total = full_data_total[full_data_total['date'].apply(return_nepal)]

full_data_total = full_data_total[full_data_total['location'].apply(return_noworld)]



full_data_total_table = full_data_total[['location','total_cases', 'total_deaths']]

full_data_total_table = full_data_total_table.sort_values('total_cases', ascending=False)



full_data_total_table = full_data_total_table.reset_index(drop=True)

full_data_total_table.style.background_gradient(cmap='Oranges')
# Visualizing the data



from plotly.subplots import make_subplots



labels = full_data_total['location'].tolist()



values1 = full_data_total['total_cases'].tolist()

values2 = full_data_total['total_deaths'].tolist()



# Create subplots: use 'domain' type for Pie subplot

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])



fig.add_trace(go.Pie(labels=labels, values=values1, name="Total Cases"),

              1, 1)

fig.add_trace(go.Pie(labels=labels, values=values2, name="Total Deaths"),

              1, 2)



# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.4, hoverinfo="label+percent+name", textposition='inside')



fig.update_layout(

    autosize=True,

    width = 1000,

    title_text="CoronaVirus Total Cases and Death Cases per Country",

    # Add annotations in the center of the donut pies.

    annotations=[

                 dict(text='Total Cases', x=0.18, y=0.5, font_size=13, showarrow=False),

                dict(text='Total Deaths', x=0.83, y=0.5, font_size=13, showarrow=False)])

fig.show()
trace1 = [go.Choropleth(

               colorscale = 'Picnic',

               locationmode = 'country names',

               locations = labels,

               text = labels, 

               z = values1,

               )]



layout = dict(title = 'CoronaVirus Total Cases',

                  geo = dict(

                      showframe = True,

                      showocean = True,

                      showlakes = True,

                      showcoastlines = True,

                      projection = dict(

                          type = 'hammer'

        )))





projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 

               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 

               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 

               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]



buttons = [dict(args = ['geo.projection.type', y],

           label = y, method = 'relayout') for y in projections]

annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 

                    xref='paper', xanchor='right', showarrow=False )])





# Update Layout Object



layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])

layout[ 'annotations' ] = annot





fig = go.Figure(data = trace1, layout = layout)

iplot(fig)

full_data_total_sort = full_data_total.sort_values('total_cases', ascending=False)

full_data_total_10 = full_data_total_sort[:10]



full_data_total_10.iplot(x='location', y=['total_cases', 'total_deaths'], kind='bar', xTitle='Country', yTitle='Number of Cases', title='Top 10 Country with CoronaVirus Cases and Death Cases')
full_data_india = full_data[full_data['location'].apply(lambda x: True if x=='India' else False)]

full_data_Bhutan = full_data[full_data['location'].apply(lambda x: True if x=='Bhutan' else False)]

full_data_Bangladesh = full_data[full_data['location'].apply(lambda x: True if x=='Bangladesh' else False)]

full_data_Afghanistan = full_data[full_data['location'].apply(lambda x: True if x=='Afghanistan' else False)]

full_data_SriLanka = full_data[full_data['location'].apply(lambda x: True if x=='Sri Lanka' else False)]

full_data_Nepal = full_data[full_data['location'].apply(lambda x: True if x=='Nepal' else False)]

full_data_Pakistan = full_data[full_data['location'].apply(lambda x: True if x=='Pakistan' else False)]

full_data_Maldives = full_data[full_data['location'].apply(lambda x: True if x=='Maldives' else False)]





# Visualizing the time Series data 

fig = go.Figure()

fig.add_trace(go.Scatter(x=full_data_india['date'], y=full_data_india['total_cases'],name='India'))

fig.add_trace(go.Scatter(x=full_data_Bhutan['date'], y=full_data_Bhutan['total_cases'],name='Bhutan'))

fig.add_trace(go.Scatter(x=full_data_Bangladesh['date'], y=full_data_Bangladesh['total_cases'],name='Bangladesh'))

fig.add_trace(go.Scatter(x=full_data_Afghanistan['date'], y=full_data_Afghanistan['total_cases'],name='Afghanistan'))

fig.add_trace(go.Scatter(x=full_data_SriLanka['date'], y=full_data_SriLanka['total_cases'],name='Sri Lanka'))

fig.add_trace(go.Scatter(x=full_data_Nepal['date'], y=full_data_Nepal['total_cases'],name='Nepal'))

fig.add_trace(go.Scatter(x=full_data_Pakistan['date'], y=full_data_Pakistan['total_cases'],name='Nepal'))

fig.add_trace(go.Scatter(x=full_data_Pakistan['date'], y=full_data_Pakistan['total_cases'],name='Nepal'))





# Set x-axis title

fig.update_xaxes(title_text="Year")

fig.update_yaxes(title_text="Number of Cases")

fig.update_layout(title_text='Total Cases per Country')

fig.show()
# Visualizing the time Series data 

fig = go.Figure()

fig.add_trace(go.Scatter(x=full_data_india['date'], y=full_data_india['total_deaths'],name='India'))

fig.add_trace(go.Scatter(x=full_data_Bhutan['date'], y=full_data_Bhutan['total_deaths'],name='Bhutan'))

fig.add_trace(go.Scatter(x=full_data_Bangladesh['date'], y=full_data_Bangladesh['total_deaths'],name='Bangladesh'))

fig.add_trace(go.Scatter(x=full_data_Afghanistan['date'], y=full_data_Afghanistan['total_deaths'],name='Afghanistan'))

fig.add_trace(go.Scatter(x=full_data_SriLanka['date'], y=full_data_SriLanka['total_deaths'],name='Sri Lanka'))

fig.add_trace(go.Scatter(x=full_data_Nepal['date'], y=full_data_Nepal['total_deaths'],name='Nepal'))

fig.add_trace(go.Scatter(x=full_data_Pakistan['date'], y=full_data_Pakistan['total_deaths'],name='Nepal'))

fig.add_trace(go.Scatter(x=full_data_Pakistan['date'], y=full_data_Pakistan['total_deaths'],name='Nepal'))





# Set x-axis title

fig.update_xaxes(title_text="Year")

fig.update_yaxes(title_text="Number of Cases")

fig.update_layout(title_text='Death Cases per Country')

fig.show()
# Extracting only data of Nepal

def return_south(location):

    if(location in ['Nepal','India', 'Bhutan', 'Bangladesh', 'Afghanistan', 'Sri Lanka','Maldives','Pakistan']):

        return True

    else:

        return False



full_data_total_south = full_data_total[full_data_total['location'].apply(return_south)]

full_data_total_south_present = full_data_total_south[full_data_total_south['date'].apply(return_nepal)]



full_data_total_south_present_sort = full_data_total_south_present.sort_values('total_cases', ascending=False)

full_data_total_south_present_sort.iplot(x='location', y=['total_cases', 'total_deaths'], kind='bar', xTitle='Country', yTitle='Number of Cases', title='South Asia CoronaVirus Cases and Death Cases')
full_data_Algeria = full_data[full_data['location'].apply(lambda x: True if x=='Algeria' else False)]

full_data_Egypt = full_data[full_data['location'].apply(lambda x: True if x=='Egypt' else False)]

full_data_Libya = full_data[full_data['location'].apply(lambda x: True if x=='Libya' else False)]

full_data_Morocco = full_data[full_data['location'].apply(lambda x: True if x=='Morocco' else False)]

full_data_Sudan = full_data[full_data['location'].apply(lambda x: True if x=='Sudan' else False)]

full_data_Tunisia = full_data[full_data['location'].apply(lambda x: True if x=='Tunisia' else False)]

full_data_WesternSahara= full_data[full_data['location'].apply(lambda x: True if x=='Western Sahara' else False)]





# Visualizing the time Series data 

fig = go.Figure()

fig.add_trace(go.Scatter(x=full_data_Algeria['date'], y=full_data_Algeria['total_cases'],name='Algeria'))

fig.add_trace(go.Scatter(x=full_data_Egypt['date'], y=full_data_Egypt['total_cases'],name='Egypt'))

fig.add_trace(go.Scatter(x=full_data_Libya['date'], y=full_data_Libya['total_cases'],name='Libya'))

fig.add_trace(go.Scatter(x=full_data_Morocco['date'], y=full_data_Morocco['total_cases'],name='Morocco'))

fig.add_trace(go.Scatter(x=full_data_Sudan['date'], y=full_data_Sudan['total_cases'],name='Sudan'))

fig.add_trace(go.Scatter(x=full_data_Tunisia['date'], y=full_data_Tunisia['total_cases'],name='Tunisia'))

fig.add_trace(go.Scatter(x=full_data_WesternSahara['date'], y=full_data_WesternSahara['total_cases'],name='Western Sahara'))



# Set x-axis title

fig.update_xaxes(title_text="Year")

fig.update_yaxes(title_text="Number of Cases")

fig.update_layout(title_text='Total Cases per Country')

fig.show()
# Visualizing the time Series data 

fig = go.Figure()

fig.add_trace(go.Scatter(x=full_data_Algeria['date'], y=full_data_Algeria['total_deaths'],name='Algeria'))

fig.add_trace(go.Scatter(x=full_data_Egypt['date'], y=full_data_Egypt['total_deaths'],name='Egypt'))

fig.add_trace(go.Scatter(x=full_data_Libya['date'], y=full_data_Libya['total_deaths'],name='Libya'))

fig.add_trace(go.Scatter(x=full_data_Morocco['date'], y=full_data_Morocco['total_deaths'],name='Morocco'))

fig.add_trace(go.Scatter(x=full_data_Sudan['date'], y=full_data_Sudan['total_deaths'],name='Sudan'))

fig.add_trace(go.Scatter(x=full_data_Tunisia['date'], y=full_data_Tunisia['total_deaths'],name='Tunisia'))

fig.add_trace(go.Scatter(x=full_data_WesternSahara['date'], y=full_data_WesternSahara['total_deaths'],name='Western Sahara'))



# Set x-axis title

fig.update_xaxes(title_text="Year")

fig.update_yaxes(title_text="Number of Cases")

fig.update_layout(title_text='Total Cases per Country')

fig.show()
# Extracting only data of Nepal

def return_north(location):

    if(location in ['Algeria', 'Egypt', 'Libya', 'Morocco', 'Sudan', 'Tunisia', 'Western Sahara']):

        return True

    else:

        return False



full_data_total_africa = full_data_total[full_data_total['location'].apply(return_north)]



full_data_total_africa_present_sort = full_data_total_africa.sort_values('total_cases', ascending=False)

full_data_total_africa_present_sort.iplot(x='location', y=['total_cases', 'total_deaths'], kind='bar', xTitle='Country', yTitle='Number of Cases', title='North Africa CoronaVirus Cases and Death Cases')
full_data_Estonia = full_data[full_data['location'].apply(lambda x: True if x=='Estonia' else False)]

full_data_Latvia = full_data[full_data['location'].apply(lambda x: True if x=='Latvia' else False)]

full_data_Lithuania= full_data[full_data['location'].apply(lambda x: True if x=='Lithuania' else False)]

full_data_Denmark= full_data[full_data['location'].apply(lambda x: True if x=='Denmark' else False)]

full_data_Finland = full_data[full_data['location'].apply(lambda x: True if x=='Finland' else False)]

full_data_Iceland = full_data[full_data['location'].apply(lambda x: True if x=='Iceland' else False)]

full_data_Norway= full_data[full_data['location'].apply(lambda x: True if x=='Norway' else False)]

full_data_Sweden = full_data[full_data['location'].apply(lambda x: True if x=='Sweden' else False)]





# Visualizing the time Series data 

fig = go.Figure()

fig.add_trace(go.Scatter(x=full_data_Estonia['date'], y=full_data_Estonia['total_cases'],name='Estonia'))

fig.add_trace(go.Scatter(x=full_data_Latvia['date'], y=full_data_Latvia['total_cases'],name='Latvia'))

fig.add_trace(go.Scatter(x=full_data_Lithuania['date'], y=full_data_Lithuania['total_cases'],name='Lithuania'))

fig.add_trace(go.Scatter(x=full_data_Denmark['date'], y=full_data_Denmark['total_cases'],name='Denmark'))

fig.add_trace(go.Scatter(x=full_data_Finland['date'], y=full_data_Finland['total_cases'],name='Finland'))

fig.add_trace(go.Scatter(x=full_data_Iceland['date'], y=full_data_Iceland['total_cases'],name='Iceland'))

fig.add_trace(go.Scatter(x=full_data_Norway['date'], y=full_data_Norway['total_cases'],name='Norway'))

fig.add_trace(go.Scatter(x=full_data_Sweden['date'], y=full_data_Sweden['total_cases'],name='Sweden'))



# Set x-axis title

fig.update_xaxes(title_text="Year")

fig.update_yaxes(title_text="Number of Cases")

fig.update_layout(title_text='Total Cases per Country')

fig.show()
# Visualizing the time Series data 

fig = go.Figure()

fig.add_trace(go.Scatter(x=full_data_Estonia['date'], y=full_data_Estonia['total_deaths'],name='Estonia'))

fig.add_trace(go.Scatter(x=full_data_Latvia['date'], y=full_data_Latvia['total_deaths'],name='Latvia'))

fig.add_trace(go.Scatter(x=full_data_Lithuania['date'], y=full_data_Lithuania['total_deaths'],name='Lithuania'))

fig.add_trace(go.Scatter(x=full_data_Denmark['date'], y=full_data_Denmark['total_deaths'],name='Denmark'))

fig.add_trace(go.Scatter(x=full_data_Finland['date'], y=full_data_Finland['total_deaths'],name='Finland'))

fig.add_trace(go.Scatter(x=full_data_Iceland['date'], y=full_data_Iceland['total_deaths'],name='Iceland'))

fig.add_trace(go.Scatter(x=full_data_Norway['date'], y=full_data_Norway['total_deaths'],name='Norway'))

fig.add_trace(go.Scatter(x=full_data_Sweden['date'], y=full_data_Sweden['total_deaths'],name='Sweden'))



# Set x-axis title

fig.update_xaxes(title_text="Year")

fig.update_yaxes(title_text="Number of Cases")

fig.update_layout(title_text='Death Cases per Country')

fig.show()
# Extracting only data of Nepal

def return_north_europe(location):

    if(location in ['Estonia', 'Latvia', 'Lithuania', 'Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden']):

        return True

    else:

        return False



full_data_total_northeurope = full_data_total[full_data_total['location'].apply(return_north_europe)]



full_data_total_northeurope_present_sort = full_data_total_northeurope.sort_values('total_cases', ascending=False)

full_data_total_northeurope_present_sort.iplot(x='location', y=['total_cases', 'total_deaths'], kind='bar', xTitle='Country', yTitle='Number of Cases', title='North European CoronaVirus Cases and Death Cases')
# Extracting only data of Nepal

def return_nepal(location):

    if(location in ['Nepal']):

        return True

    else:

        return False



full_data_nepal = full_data[full_data.location.apply(return_nepal)]



# Visualizing the time Series data 

fig = go.Figure()

fig.add_trace(go.Scatter(x=full_data_nepal['date'], y=full_data_nepal['total_cases'], fill='tozeroy',name='Total Cases'))

fig.add_trace(go.Scatter(x=full_data_nepal['date'], y=full_data_nepal['total_deaths'], fill='tozeroy',name='Total Death'))





# Set x-axis title

fig.update_xaxes(title_text="Year")

fig.update_yaxes(title_text="Number of Cases")

fig.show()
import plotly.express as px 



#New CoronaVirus Cases 

fig = px.box(full_data_nepal, y='new_cases', title='CoronaVirus New Cases Per Day')

fig.show()
#New CoronaVirus Death Cases 

fig = px.box(full_data_nepal, y='new_deaths', title='CoronaVirus New Death Cases Per Day')

fig.show()
full_data_nepal_process = full_data_nepal[['date', 'new_cases', 'new_deaths', 'total_cases',

       'weekly_cases', 'biweekly_cases'

       ]]
full_data_nepal_process['date'] = full_data_nepal_process['date'].dt.strftime('%y%m%d')

full_data_nepal_process['date'] = full_data_nepal_process['date'].astype(int)



plt.rcParams['figure.figsize'] = (15, 12)

sns.heatmap(full_data_nepal_process.corr(), cmap='gray', annot=True)

plt.show()
full_data_nepal_process['date'].head()



X = full_data_nepal_process[['date', 'total_cases', 'new_cases',

        'weekly_cases', 'biweekly_cases'

       ]].to_numpy()





Y = full_data_nepal_process[['new_deaths']].to_numpy()



from sklearn.model_selection import train_test_split



#make the x for train and test (also called validation data) 

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.9,random_state=42, shuffle=True)
from sklearn.neighbors import KNeighborsRegressor

from joblib import dump, load



def return_error(x_train,x_test,y_train):

    model = KNeighborsRegressor(algorithm='ball_tree', leaf_size=50, metric='minkowski',

                    metric_params=None, n_jobs=-1, n_neighbors=14, p=2,

                    weights='uniform')

    model.fit(x_train, y_train)

    dump(model, 'regressor.joblib')

    

    predict = model.predict(x_test)

    return predict
y_train = y_train.ravel()

predict = return_error(x_train,x_test,y_train)
from sklearn.metrics import r2_score



#for total cases

r2_score(y_test, predict)
from sklearn.metrics import mean_squared_error



#for total cases

mean_squared_error(y_test, predict)
import numpy as np

import matplotlib.pyplot as plt



hist1 = y_test

hist2 = predict



from scipy.stats import norm



sns.distplot(hist1,color="b",hist=False, label='Actual Death')

sns.distplot(hist2,hist=False,color='red', label='Predicted Death')