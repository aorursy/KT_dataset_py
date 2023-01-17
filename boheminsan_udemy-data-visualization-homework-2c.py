# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#df = pd.read_csv('../input/earthquake.csv')

# Any results you write to the current directory are saved as output.
df11 = pd.read_csv('../input/earthquake.csv')
df11=df11.loc[df11['country'] == 'turkey']
df11['country'].loc[df11['country']=='turkey']='Turkey'
df11.sort_values(by=['richter'],ascending=False).head(10)
dfdmy=pd.DataFrame()
df1=[]
mf1=[]
yf1=[]
for i in df11.date:
    a,b,c=i.split('.')
    yf1.append(a)
    mf1.append(b)
    df1.append(c)
dfdmy['day']=df1
dfdmy['month']=mf1
dfdmy['year']=yf1
df11=pd.concat([df11,dfdmy],axis=1)
df11=df11.drop(['id','xm','md','mw','ms','mb','dist'], axis=1)
df11.dropna(inplace=True)
df=df11.nlargest(1000, 'richter')
data = [ dict(
        type = 'scattergeo',
        locationmode = 'country names',
        lon = df['long'],
        lat = df['lat'],
        text = df['city'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = False,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = 'Bluered',
            cmin = 0,
            color = df['richter'],
            cmax = df['richter'].max(),
            colorbar=dict(
                title="Highest Earthquakes by Years"
            )
        ))]

layout = dict(
        title = 'Most Harmful Earthquakes on Turkey',
        #colorbar = True,
        geo = dict(
            scope='asia',
            projection=dict(type='mercator'),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(55, 217, 217)",
            countrycolor="rgb(31, 58, 147)",
            countrywidth = 0.5,
            subunitwidth = 0.5,
            lonaxis = dict(range= [25.5, 45.5]),
            lataxis = dict(range= [35.5, 42.5]),
            showocean=True,
            showlakes=True
        ),
    )

fig = dict( data=data, layout=layout )
iplot(fig)
df11.dropna(inplace=True)
df11.year=df11.year.astype(int)
df11.year.min()
df11=df11.loc[df11['richter'] >= 4]
df11=df11.sort_values(by=['year'],ascending=True)
df11.info()
years = [each for each in list(df11.year.unique())]  # str unique years
types=['Eathquake']
# make figure
figure = {
    'data': [],
    'layout': {},
    'frames': []
}

figure['layout']['geo'] = dict(
            scope='asia',
            projection=dict(type='mercator'),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(55, 217, 217)",
            countrycolor="rgb(31, 58, 147)",
            countrywidth = 1,
            subunitwidth = 1,
            lonaxis = dict(range= [25.5, 45.5]),
            lataxis = dict(range= [35.5, 42.5]),
            showocean=True,
            showlakes=True
        )
figure['layout']['hovermode'] = 'closest'
figure['layout']['sliders'] = {
    'args': [
        'transition', {
            'duration': 400,
            'easing': 'cubic-in-out'
        }
    ],
    'initialValue': df11.year.min(),
    'plotlycommand': 'animate',
    'values': years,
    'visible': True
}
figure['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]

sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Year:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

# make data
year = 1950

data_dict = dict(
    type='scattergeo',
    lon = df11['long'],
    lat = df11['lat'],
    hoverinfo = 'text',
    text = df11['area'],
    mode = 'markers',
    marker=dict(
        sizemode = 'area',
        sizeref = 1,
        size= 10 ,
        line = dict(width=1,color = "white"),
        #color = 'blue',
        opacity = 0.7),
)
figure['data'].append(data_dict)

# make frames
for year in years:
    frame = {'data': [], 'name': str(year)}
    dataset_by_year = df11.loc[df11['year'] == year]

    data_dict = dict(
                type='scattergeo',
                lon = dataset_by_year['long'],
                lat = dataset_by_year['lat'],
                hoverinfo = 'text',
                text = dataset_by_year['area'],
                mode = 'markers',
                marker=dict(
                    sizemode = 'area',
                    sizeref = 1,
                    size= 10 ,
                    line = dict(width=1,color = "white"),
                    color = 'blue',
                    opacity = 0.7),
            )
    frame['data'].append(data_dict)

    figure['frames'].append(frame)
    slider_step = {'args': [
        [year],
        {'frame': {'duration': 300, 'redraw': False},
         'mode': 'immediate',
       'transition': {'duration': 300}}
     ],
     'label': str(year),
     'method': 'animate'}
    sliders_dict['steps'].append(slider_step)


figure["layout"]["autosize"]= True
figure["layout"]["title"] = "Earthquake"       

figure['layout']['sliders'] = [sliders_dict]

iplot(figure)