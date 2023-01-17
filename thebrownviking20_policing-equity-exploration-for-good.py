# Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import os
print(os.listdir("../input"))
# Displaying available data files and directories
def dirlist(parent,child,count=0):
    if os.path.isdir(parent+child+"/"):
        p_list = os.listdir(parent+child+"/")
        p_text = parent+child+"/"
        if len(p_list)>0:
                count = count + 1
                for val,child in enumerate(p_list):
                    print("{}-{}".format(" "*(count*4),child))
                    dirlist(p_text,child,count)
        else:
            pass
    
    
for child in os.listdir("../input/cpe-data/"):
    print("{}".format(child))
    dirlist("../input/cpe-data/",child)
        
#Let's look at the available department files
[f for f in os.listdir("../input/cpe-data/") if f.startswith("Dept")]
#Let's start with Dept_23-00089
os.listdir("../input/cpe-data/Dept_23-00089/23-00089_ACS_data/")
#First, we will look at race,sex and age data
one_df = pd.read_csv("../input/cpe-data/Dept_23-00089/23-00089_ACS_data/23-00089_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv")
one_df.info()
one_df.describe()
data = [go.Histogram(x=one_df["HC01_VC04"],
                     name='Male',
                     opacity=0.7,
                     marker=dict(
                        color='rgb(158,202,225)',
                        line=dict(
                            color='rgb(8,48,107)',
                            width=1.5,
                        )
                    )),
        go.Histogram(x=one_df["HC01_VC05"],
                     name='Female',
                     opacity=0.7,
                     marker=dict(
                        color='rgb(255,254,115)',
                        line=dict(
                            color='rgb(255,233,93)',
                            width=1.5,
                        )
                    )),
        go.Histogram(x=one_df["HC01_VC03"],
                     name='Total',
                     opacity=0.7,
                     marker=dict(
                        color='rgb(0,255,174)',
                        line=dict(
                            color='rgb(51,158,53)',
                            width=1.5,
                        )
                    ))]

updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=list([
            dict(label = 'Male',
                 method = 'update',
                 args = [{'visible': [True, False, False]},
                         {'title': 'Male'}]),
            dict(label = 'Female',
                 method = 'update',
                 args = [{'visible': [False, True, False]},
                         {'title': 'Female'}]),
            dict(label = 'Total',
                 method = 'update',
                 args = [{'visible': [False,False,True]},
                         {'title': 'Total'}]),
            dict(label = 'Male & Female',
                 method = 'update',
                 args = [{'visible': [True, True, False]},
                         {'title': 'Male & Female'}]),
             dict(label = 'Male & Total',
                 method = 'update',
                 args = [{'visible': [True, False, True]},
                         {'title': 'Male & Total'}]),
             dict(label = 'Female & Total',
                 method = 'update',
                 args = [{'visible': [False, True, True]},
                         {'title': 'Female & Total'}]),
             dict(label = 'All',
                 method = 'update',
                 args = [{'visible': [True, True, True]},
                         {'title': 'All'}])
        ]),
    )
])
layout = go.Layout(barmode='overlay')
layout['title'] = 'Population Distribution'
layout['showlegend'] = True
layout['updatemenus'] = updatemenus

fig = dict(data=data, layout=layout)
iplot(fig, filename='update_button')