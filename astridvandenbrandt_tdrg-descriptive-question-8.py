# import additional packages



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py

import plotly.graph_objs as go

import seaborn as sns

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

sns.set_style('whitegrid')

%matplotlib inline

init_notebook_mode()

import matplotlib.colors as colors

import matplotlib.cm as cm

import matplotlib.patches as mpatches

from mpl_toolkits.basemap import Basemap



from subprocess import check_output

from datetime import datetime

print(check_output(["ls", "../input"]).decode("utf8"))
# Import clean data



data_terrorism = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', usecols=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,30,32,34,36,38,39,40,42,44,46,47,48,50,52,54,55,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,80,81,83,84,85,87,89,91,93,95,97,98,100,101,102,103,104,105,107,108,109,110,111,112,113,114,115,116,117,118,119,120,122,124,125,126,127,128,130,131,132,133,134])

data_terrorism.info()
data_terrorism.dropna(subset=['nkill'])

data_terrorism.dropna(subset=['nwound'])

data_terrorism.head()

data_terrorism.info()
nr_weaptype1 = np.asarray(data_terrorism.groupby('weaptype1').weaptype1.count())

nkill_weaptype1 = np.asarray(data_terrorism.groupby('weaptype1').nkill.sum())

average_nkill_weap = np.divide(nkill_weaptype1, nr_weaptype1) 



weaptype1_names = np.array(['Biological','Chemical','Radiological','Firearms','Explosives/Bombs/Dynamite','Fake weapons','Indenciary','Melee','Vehicle','Sabotage Equipment','Other','Unkown'])

#hoe kan dit 12 zijn?--> 4:Nuclear komt niet voor 

total_deaths_weap = sum(nkill_weaptype1)

average_nkill_weap_2 = np.divide(nkill_weaptype1, total_deaths_weap) 

average_nkill_weap_total = average_nkill_weap_2*100



nwound_weaptype1 = np.asarray(data_terrorism.groupby('weaptype1').nwound.sum())

average_nwound_weap = np.divide(nwound_weaptype1, nr_weaptype1) 



total_wounded_weap = sum(nwound_weaptype1)

average_nwound_weap_2 = np.divide(nwound_weaptype1, total_wounded_weap) 

average_nwound_weap_total = average_nwound_weap_2*100



total_nkill_nwound_weap = total_deaths_weap + total_wounded_weap

nkill_nwound_weap = [x + y for x, y in zip(nkill_weaptype1, nwound_weaptype1)]

share_nkill_nwound_weap = np.divide(nkill_nwound_weap,total_nkill_nwound_weap)

share_percent_nkill_nwound_weap = share_nkill_nwound_weap*100
# make donutcharts

fig = {

  "data": [

    {

      "values": average_nkill_weap_total,

      "labels": weaptype1_names

        ,

    "text":"fatalities",

      "textposition":"inside",

      "domain": {"x": [0, .3]},

      "name": "",

      "hoverinfo":"label+percent+name",

          "hole": .4,

      "type": "pie"

    },     

      {

      "values": average_nwound_weap_total,

      "labels": weaptype1_names

        ,

    "text":"ijuries",

      "textposition":"inside",

      "domain": {"x": [.35, .65]},

      "name": "",

      "hoverinfo":"label+percent+name",

          "hole": .4,

      "type": "pie"

    },

       {

      "values": share_percent_nkill_nwound_weap,

      "labels": weaptype1_names

        ,

    "text":"fatalities + ijuries",

      "textposition":"inside",

      "domain": {"x": [.70, 1]},

      "name": "",

      "hoverinfo":"label+percent+name",

          "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Share of killed and wounded people per weapon type", "showlegend":False,

        "annotations": [

            {

                "font": {

                    "size": 10

                },

                "showarrow": False,

                "text": "Killed",

                "x": 0.13,

                "y": 0.5

            },

             {

                "font": {

                    "size": 10

                },

                "showarrow": False,

                "text": "Wounded",

                "x": 0.5,

                "y": 0.5

            },

            {

                "font": {

                    "size": 10

                },

                "showarrow": False,

                "text": "Casualties",

                "x": 0.89,

                "y": 0.5

            }

        ]

    }

}

iplot(fig, filename='donut')