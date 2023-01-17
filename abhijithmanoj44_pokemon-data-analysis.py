import pandas as pd   

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True) 

pokedata = pd.read_csv("../input/pokemon-all-cleaned/pokemon_cleaned.csv")
pokedata.head()
pokedata.loc[[pokedata['Speed'].idxmax()]]
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True) 

a = pokedata[pokedata["Name"] == "Blastoise"]
b = pokedata[pokedata["Name"] == "Charizard"]
c = pokedata[pokedata["Name"] == "Mewtwo"]

data = [
    go.Scatterpolar(
        name = a.Name.values[0],
        r = [a['HP'].values[0],a['Attack'].values[0],a['Defense'].values[0],a['Sp. Atk'].values[0],a['Sp. Def'].values[0],a['Speed'].values[0],a["HP"].values[0]],
        theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
        fill = 'toself',
        line =  dict(
                color = 'cyan'
            )
        ),
    go.Scatterpolar(
            name = b.Name.values[0],
            r = [b['HP'].values[0],b['Attack'].values[0],b['Defense'].values[0],b['Sp. Atk'].values[0],b['Sp. Def'].values[0],b['Speed'].values[0],b["HP"].values[0]],
            theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
            fill = 'toself',
            line =  dict(
                color = 'orange'
            )
    ),
      go.Scatterpolar(
            name = c.Name.values[0],
            r = [c['HP'].values[0],c['Attack'].values[0],c['Defense'].values[0],c['Sp. Atk'].values[0],c['Sp. Def'].values[0],c['Speed'].values[0],c["HP"].values[0]],
            theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
            fill = 'toself',
            line =  dict(
                color = 'pink'
            )
        )        

]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 200]
    )
  ),
  showlegend = True,
  title = "{} vs {} vs {} Stats Comparison".format(a.Name.values[0], b.Name.values[0], c.Name.values[0])
)

fig = go.Figure(data=data, layout=layout)

fig.layout.images = [dict(
        source="https://rawgit.com/guramera/images/master/blastoise.jpg",
        xref="paper", yref="paper",
        x=0.05, y=-0.15,
        sizex=0.6, sizey=0.6,
        xanchor="center", yanchor="bottom"
      ),
        dict(
        source="https://rawgit.com/guramera/images/master/Charizard.png",
        xref="paper", yref="paper",
        x=0.2, y=-0.15,
        sizex=0.6, sizey=0.6,
        xanchor="center", yanchor="bottom"
      ) ,
                    dict(
        source="https://i.imgur.com/lyTvmPP.png",
        xref="paper", yref="paper",
        x=1, y=-0.15,
        sizex=0.6, sizey=0.6,
        xanchor="center", yanchor="bottom"
      )  
                    ]


iplot(fig, filename = "Pokemon stats comparison")