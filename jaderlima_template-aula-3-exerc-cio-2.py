import pandas as pd
url = "https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/Pokemon.csv"
df_pokemon = pd.read_csv(url, delimiter=',')


import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)

dr = (df_pokemon.groupby(by=['Type 1','Legendary']).sum()['Attack']).reset_index().rename(columns={'index':'Type 1'})
legendary = (df_pokemon[df_pokemon['Legendary'] == True].groupby(by=['Type 1','Legendary']).sum()['Attack']).reset_index().rename(columns={'index':'Type 1','Attack':'Attack Legendary'})
normal = (df_pokemon[df_pokemon['Legendary'] == False].groupby(by=['Type 1','Legendary']).sum()['Attack']).reset_index().rename(columns={'index':'Type 1','Attack':'Attack Normal'})
legendary.drop('Legendary', axis=1, inplace=True)
normal.drop('Legendary', axis=1, inplace=True)
d = pd.merge(normal,legendary ,how='left', on='Type 1')
d.fillna(0, inplace=True)
random_x = d[d['Attack Legendary'].notnull()]['Type 1']
random_y =  d[d['Attack Legendary'].notnull()]['Attack Legendary']

trace1 = go.Scatter(
    x = random_x,
    y = random_y,
    mode = 'lines',
   name = 'Lendarios',
)

random_x1 = d['Type 1']
random_y1 = d['Attack Normal']

trace2 = go.Scatter(
    x = random_x1,
    y = random_y1,
    mode = 'lines',
     name = 'Normal',
)

data = [trace1, trace2]

plotly.offline.iplot(data, filename='basic-line')


%matplotlib inline
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

df_barras = (df_pokemon[['Type 1','HP','Attack','Defense']].groupby(by=['Type 1'])).mean().reset_index().rename(columns={'index':'Type 1'})

x = df_barras['Type 1']
y = df_barras['HP']
y2 = df_barras['Attack']
y3 = df_barras['Defense']

trace1 = go.Bar(
    x= x,
    y=y,
    text=y,
    name='HP',
    textposition = 'auto',
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)

trace2 = go.Bar(
    x=x,
    y=y2,
    name='Attack',
    text=y2,
    textposition = 'auto',
    marker=dict(
        color='rgb(58,200,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)

trace3 = go.Bar(
    x=x,
    y=y3,
    name='Defense',
    text=y3,
    textposition = 'auto',
    marker=dict(
        color='rgb(10,100,155)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)

data = [trace1,trace2,trace3]
plotly.offline.iplot(data, filename='grouped-bar-direct-labels')
#py.iplot(data, filename='grouped-bar-direct-labels')
%matplotlib inline
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



x = df_pokemon['Attack'].values
data = [go.Histogram(x=x)]



layout = go.Layout(
    xaxis=dict(
        title='',
        zeroline=False,
        showgrid=False,
         showline=False,
    )
)

fig = go.Figure(data=data, layout=layout)


plotly.offline.iplot(fig)
%matplotlib inline
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)


Fire_HP = df_pokemon[df_pokemon['Type 1'] == 'Fire'].mean()['HP']
Fire_Attack = df_pokemon[df_pokemon['Type 1'] == 'Fire'].mean()['Attack']
Fire_Defense = df_pokemon[df_pokemon['Type 1'] == 'Fire'].mean()['Defense']
Fire_SpAtk = df_pokemon[df_pokemon['Type 1'] == 'Fire'].mean()['Sp. Atk']
Fire_SpDef = df_pokemon[df_pokemon['Type 1'] == 'Fire'].mean()['Sp. Def']
Fire_Speed = df_pokemon[df_pokemon['Type 1'] == 'Fire'].mean()['Speed']
Fire_Stage = df_pokemon[df_pokemon['Type 1'] == 'Fire'].mean()['Stage']

Water_HP = df_pokemon[df_pokemon['Type 1'] == 'Water'].mean()['HP']
Water_Attack = df_pokemon[df_pokemon['Type 1'] == 'Water'].mean()['Attack']
Water_Defense = df_pokemon[df_pokemon['Type 1'] == 'Water'].mean()['Defense']
Water_SpAtk = df_pokemon[df_pokemon['Type 1'] == 'Water'].mean()['Sp. Atk']
Water_SpDef = df_pokemon[df_pokemon['Type 1'] == 'Water'].mean()['Sp. Def']
Water_Speed = df_pokemon[df_pokemon['Type 1'] == 'Water'].mean()['Speed']
Water_Stage = df_pokemon[df_pokemon['Type 1'] == 'Water'].mean()['Stage']


data = [
    go.Scatterpolar(
      r = [Fire_HP, Fire_Attack, Fire_Defense, Fire_SpAtk, Fire_SpDef, Fire_Speed,Fire_Speed , Fire_Stage],
      theta = ['HP','Attack','Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Stage'],
      fill = 'toself',
      name = 'Fire Pokemons'
    ),
    go.Scatterpolar(
      r = [Water_HP, Water_Attack, Water_Defense, Water_SpAtk, Water_SpDef, Water_Speed,Water_Speed , Water_Stage],
      theta =  ['HP','Attack','Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Stage'],
      fill = 'toself',
      name = 'Water Pokemons'
    )
]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 100]
    )
  ),
  showlegend = False
)

fig = go.Figure(data=data, layout=layout)
#py.iplot(fig, filename = "radar/multiple")
plotly.offline.iplot(fig, filename = "radar")
%matplotlib inline
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



trace0 = go.Box( 
  y= df_pokemon[df_pokemon['Legendary'] == False]['Attack'] ,
    name='Attack',
     boxmean='sd',
    marker=dict(
        color='rgb(8, 81, 156)'
        #color='#1E88E5'
    )
)
trace1 = go.Box(
    y=df_pokemon[df_pokemon['Legendary'] == False]['Defense'],  
    name='Defense',
     boxmean='sd',
    marker=dict(
        color='#D32F2F'
    )
)

data = [trace0, trace1]
layout = go.Layout(
   yaxis=dict(
        title='',
        zeroline=False,
        showgrid=False,
         showline=True,
    ),   
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename = "box plot")
%matplotlib inline
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

df_type1 =  df_pokemon[( ( df_pokemon['Type 1'] == 'Fire') | (df_pokemon['Type 1'] == 'Water')  | (df_pokemon['Type 1'] == 'Ice')  | (df_pokemon['Type 1'] == 'Grass') ) ] 

#for i in range(0,len(pd.unique(df['day']))):
data = []


trace = {
        "type": 'violin',
        "x": 'Fire',
        "y": df_type1[ df_type1['Type 1'] == 'Fire']['Attack'],
        "name": 'Fire',
        "fillcolor": '#ff9900',
         "line": {
            "color": 'black'
        },
        "box": {
            "visible": True
        },
        "meanline": {
            "visible": True
        }
    }
data.append(trace)

trace = {
        "type": 'violin',
        "x": 'Water',
        "y": df_type1[ df_type1['Type 1'] == 'Water']['Attack'],
        "name": 'Water',
        "fillcolor": '#00ccff',
     "line": {
            "color": 'black'
        },
        "box": {
            "visible": True
        },
        "meanline": {
            "visible": True
        }
    }
data.append(trace)

trace = {
        "type": 'violin',
        "x": 'Ice',
        "y": df_type1[ df_type1['Type 1'] == 'Ice']['Attack'],
        "name": 'Ice',
         "fillcolor": '#F5F5F5',
     "line": {
            "color": 'black'
        },
        "box": {
            "visible": True
        },
        "meanline": {
            "visible": True
        }
    }
data.append(trace)

trace = {
        "type": 'violin',
        "x": 'Grass',
        "y": df_type1[ df_type1['Type 1'] == 'Grass']['Attack'],
        "name": 'Grass',
    "fillcolor": '#00cc99',
     "line": {
            "color": 'black'
        },
        "box": {
            "visible": True
        },
        "meanline": {
            "visible": True
        }
    }
data.append(trace)

        
fig = {
    "data": data,
    "layout" : {
        "title": "",
        "yaxis": {
            "zeroline": False,
        }
    }
}

plotly.offline.iplot(fig, filename='violin/multiple', validate = False)
