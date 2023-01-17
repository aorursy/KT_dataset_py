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





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_terrorism = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', usecols=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,30,32,34,36,38,39,40,42,44,46,47,48,50,52,54,55,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,80,81,83,85,87,89,91,93,95,97,98,100,101,102,103,104,105,107,108,109,110,111,112,113,114,115,116,117,118,119,120,122,124,125,126,127,128,130,131,132,133,134])

data_terrorism.head()
##graph 1

nr_targtype1 = np.asarray(data_terrorism.groupby('targtype1').targtype1.count())

nkill_targtype1 = np.asarray(data_terrorism.groupby('targtype1').nkill.sum())

average_nkill = np.divide(nkill_targtype1, nr_targtype1) 



targtype1_names = np.array(["Business","Gevernment","Police","Military","Abortion Related","Airports & Aircrafts", "Government (diplomaitc)", "Educational Institution", "Food or water supply","Journalists & Media","Maritime","NGO","Other","Private Citizens & property","Religious figures/institutions","Telecommunication","Terrorist/non-state militias","Tourists","Transportation","Unknown","Utilities","Violent politican parties"])

print(targtype1_names)



total_deaths = sum(nkill_targtype1)

average_nkill2 = np.divide(nkill_targtype1, total_deaths) 

average_nkill_kills = average_nkill2*100



nwound_targtype1 = np.asarray(data_terrorism.groupby('targtype1').nwound.sum())

average_nwound = np.divide(nwound_targtype1, nr_targtype1) 



total_wounded = sum(nwound_targtype1)

average_nwound2 = np.divide(nwound_targtype1, total_wounded) 

average_nwound_wounded = average_nwound2*100



propextent_targtype1 = np.asarray(data_terrorism.groupby('targtype1').propextent.sum())

average_propextent = np.divide(propextent_targtype1, nr_targtype1) 



total_property = sum(propextent_targtype1)

average_propextent2 = np.divide(propextent_targtype1, total_property) 

average_propextent_propextent = average_propextent2*100



# Create dataframe

targtype1_data = pd.DataFrame({"targtype1_names":targtype1_names,"nr_targtype1":nr_targtype1,"nkill_targtype1":nkill_targtype1,"average_nkill":average_nkill, "average_nkill_kills":average_nkill_kills,"average_nwound":average_nwound,"average_nwound_wounded":average_nwound_wounded,"average_propextent":average_propextent,"average_propextent_propextent":average_propextent_propextent})

targtype1_data.head()
#sort the dataframe from large to small

sorted_targtype1_data = targtype1_data.sort_values(by='average_nkill', ascending=0)

sorted1_targtype1_data = targtype1_data.sort_values(by='average_nkill_kills', ascending=0)

sorted2_targtype1_data = targtype1_data.sort_values(by='average_nwound', ascending=0)

sorted3_targtype1_data = targtype1_data.sort_values(by='average_nwound_wounded', ascending=0)

sorted4_targtype1_data = targtype1_data.sort_values(by='average_propextent', ascending=0)

sorted5_targtype1_data = targtype1_data.sort_values(by='average_propextent_propextent', ascending=0)



#Making barplots

ax = sns.barplot(y='targtype1_names',x='average_nkill', data=sorted_targtype1_data, color="#00035b", palette="Reds_r")

ax.set_xlabel("Average number deaths per target", size=10, alpha=1)

ax.set_ylabel("Targettype Names", size=10, alpha=1)

ax.set(xlim=(0, 5))

ax.set_title("The average number of deaths per attack given the target type", fontsize=12, alpha=1)

ax.tick_params(labelsize=10,labelcolor="black")

plt.show()



ax = sns.barplot(y='targtype1_names',x='average_nwound', data=sorted2_targtype1_data, color="#00035b", palette="Blues_r")

ax.set_xlabel("Average number of wounded people per attack", size=10, alpha=1)

ax.set_ylabel("Target type names", size=10, alpha=1)

ax.set(xlim=(0, 7))

ax.set_title("The average number of wounded people per attack given the target type", fontsize=12, alpha=1)

ax.tick_params(labelsize=10,labelcolor="black")

plt.show()



ax = sns.barplot(y='targtype1_names',x='average_propextent', data=sorted4_targtype1_data, color="#00035b", palette="Greens_r")

ax.set_xlabel("Average property damage per attack", size=10, alpha=1)

ax.set_ylabel("Targettype names", size=10, alpha=1)

ax.set(xlim=(0, 3))

ax.set_title("The average extent of property damage per target type", fontsize=12, alpha=1)

ax.tick_params(labelsize=10,labelcolor="black")

plt.show()



#Making donut charts

fig = {

  "data": [

    {

      "values": average_nkill_kills,

      "labels": targtype1_names

        ,

    "text":"Property Damage",

      "textposition":"inside",

      "domain": {"x": [0, .30]},

      "name": "",

      "hoverinfo":"label+percent+name",

          "hole": .4,

      "type": "pie"

    },     

      {

      "values": average_nwound_wounded,

      "labels": targtype1_names

        ,

    "text":"nkill",

      "textposition":"inside",

      "domain": {"x": [.35, .65]},

      "name": "",

      "hoverinfo":"label+percent+name",

          "hole": .4,

      "type": "pie"

    },     

    {

      "values": average_propextent_propextent,

      "labels":  targtype1_names

        ,

      "text":"Nwound",

      "textposition":"inside",

      "domain": {"x": [.70, 1]},

      "name": "",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Share of kills wounded and property per Target type ",

        "showlegend" : False,

        "annotations": [

            {

                "font": {

                    "size": 10

                },

                "showarrow": False,

                "text": "Killed",

                "x": 0.1275,

                "y": 0.5

            },

             {

                "font": {

                    "size": 10

                },

                "showarrow": False,

                "text": "Wounded",

                "x": 0.50,

                "y": 0.5

            },

            {

                "font": {

                    "size": 10

                },

                "showarrow": False,

                "text": "Property",

                "x": 0.885,

                "y": 0.5

            },

        ]

    }

}

iplot(fig, filename='donut')
###attacktype 

nr_attacktype1 = np.asarray(data_terrorism.groupby('attacktype1').attacktype1.count())

nkill_attacktype1 = np.asarray(data_terrorism.groupby('attacktype1').nkill.sum())

average_nkill = np.divide(nkill_attacktype1, nr_attacktype1) 



attacktype1_names = np.array(['Assassination','Armed Attack','Bombing/Explosion','Hijacking','Hostage Taking barricade incident','Hostage Taking kidnapping','Facility/Infrastructure Attack','Unarmed Assault','Unknown'])

print(attacktype1_names)



total_deaths = sum(nkill_attacktype1)

average_nkill2 = np.divide(nkill_attacktype1, total_deaths) 

average_nkill_kills = average_nkill2*100



nwound_attacktype1 = np.asarray(data_terrorism.groupby('attacktype1').nwound.sum())

average_nwound = np.divide(nwound_attacktype1, nr_attacktype1) 



total_wounded = sum(nwound_attacktype1)

average_nwound2 = np.divide(nwound_attacktype1, total_wounded) 

average_nwound_wounded = average_nwound2*100



propextent_attacktype1 = np.asarray(data_terrorism.groupby('attacktype1').propextent.sum())

average_propextent = np.divide(propextent_attacktype1, nr_attacktype1) 

total_property = sum(propextent_attacktype1)

average_propextent2 = np.divide(propextent_attacktype1, total_property) 

average_propextent_propextent = average_propextent2*100

attacktype_data = pd.DataFrame({"attacktype1_names":attacktype1_names,"nr_attacktype1":nr_attacktype1,"nkill_attacktype1":nkill_attacktype1,"average_nkill":average_nkill, "average_nkill_kills":average_nkill_kills,"average_nwound":average_nwound,"average_nwound_wounded":average_nwound_wounded,"average_propextent":average_propextent,"average_propextent_propextent":average_propextent_propextent})

attacktype_data.head()
#sort the dataframes from large to small

sorted_attacktype_data = attacktype_data.sort_values(by='average_nkill', ascending=0)

sorted1_attacktype_data = attacktype_data.sort_values(by='average_nkill_kills', ascending=0)

sorted2_attacktype_data = attacktype_data.sort_values(by='average_nwound', ascending=0)

sorted3_attacktype_data = attacktype_data.sort_values(by='average_nwound_wounded', ascending=0)

sorted4_attacktype_data = attacktype_data.sort_values(by='average_propextent', ascending=0)

sorted5_attacktype_data = attacktype_data.sort_values(by='average_propextent_propextent', ascending=0)
##make barplots 



ax = sns.barplot(y='attacktype1_names',x='average_nkill_kills', data=sorted1_attacktype_data, color="#00035b", palette="Reds_r")

ax.set_xlabel("Mortality rate(?) per attacktype", size=10, alpha=1)

ax.set_ylabel("Attacktype names", size=10, alpha=1)

ax.set(xlim=(0, 100))

ax.set_title("Deaths per attack type compared to total number of killed people (in %)", fontsize=12, alpha=1)

ax.tick_params(labelsize=10,labelcolor="black")

plt.show()



ax = sns.barplot(y='attacktype1_names',x='average_nwound', data=sorted2_attacktype_data, color="#00035b", palette="Blues_r")

ax.set_xlabel("Average number of wounded people per attack", size=10, alpha=1)

ax.set_ylabel("Attacktype names", size=10, alpha=1)

ax.set(xlim=(0, 30))

ax.set_title("The average number of wounded people per attack given the attack type", fontsize=12, alpha=1)

ax.tick_params(labelsize=10,labelcolor="black")

plt.show()



ax = sns.barplot(y='attacktype1_names',x='average_propextent', data=sorted4_attacktype_data, color="#00035b", palette="Greens_r")

ax.set_xlabel("Average property damage per attack", size=10, alpha=1)

ax.set_ylabel("Attacktype names", size=10, alpha=1)

ax.set(xlim=(0, 4))

ax.set_title("The average extent of property damage per attack type", fontsize=12, alpha=1)

ax.tick_params(labelsize=10,labelcolor="black")

plt.show()



# make donutcharts



fig = {

  "data": [

    {

      "values": average_nkill_kills,

      "labels": attacktype1_names

        ,

    "text":"Property Damage",

      "textposition":"inside",

      "domain": {"x": [0, .30]},

      "name": "",

      "hoverinfo":"label+percent+name",

          "hole": .4,

      "type": "pie"

    },     

      {

      "values": average_nwound_wounded,

      "labels": attacktype1_names

        ,

    "text":"nkill",

      "textposition":"inside",

      "domain": {"x": [.35, .65]},

      "name": "",

      "hoverinfo":"label+percent+name",

          "hole": .4,

      "type": "pie"

    },     

    {

      "values": average_propextent_propextent,

      "labels":  attacktype1_names

        ,

      "text":"Nwound",

      "textposition":"inside",

      "domain": {"x": [.70, 1]},

      "name": "",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Share of kills, wounded and property damage per attack type", "showlegend":False,

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

                "x": 0.50,

                "y": 0.5

            },

            {

                "font": {

                    "size": 10

                },

                "showarrow": False,

                "text": "Property",

                "x": 0.885,

                "y": 0.5

            }

        ]

    }

}

iplot(fig, filename='donut')




