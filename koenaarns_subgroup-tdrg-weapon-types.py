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
# weapon type used in hystorical context

data_Biological = data_terrorism[(data_terrorism.weaptype1 == 1)]

data_Chemical = data_terrorism[(data_terrorism.weaptype1 == 2)]

data_Radiological = data_terrorism[(data_terrorism.weaptype1 == 3)]

data_Nuclear = data_terrorism[(data_terrorism.weaptype1 == 4)]

data_Firearms = data_terrorism[(data_terrorism.weaptype1 == 5)]

data_Explosives = data_terrorism[(data_terrorism.weaptype1 == 6)]

data_Fake = data_terrorism[(data_terrorism.weaptype1 == 7)]

data_Incendiary = data_terrorism[(data_terrorism.weaptype1 == 8)]

data_Melee = data_terrorism[(data_terrorism.weaptype1 == 9)]

data_Vehicle = data_terrorism[(data_terrorism.weaptype1 == 10)]

data_Sabotage = data_terrorism[(data_terrorism.weaptype1 == 11)]

data_Other = data_terrorism[(data_terrorism.weaptype1 == 12)]



peryear_Biological = np.asarray(data_Biological.groupby('iyear').iyear.count())

peryear_Chemical = np.asarray(data_Chemical.groupby('iyear').iyear.count())

peryear_Radiological = np.asarray(data_Radiological.groupby('iyear').iyear.count())

peryear_Nuclear = np.asarray(data_Nuclear.groupby('iyear').iyear.count())

peryear_firearms = np.asarray(data_Firearms.groupby('iyear').iyear.count())

peryear_Explosives = np.asarray(data_Explosives.groupby('iyear').iyear.count())

peryear_fake = np.asarray(data_Fake.groupby('iyear').iyear.count())

peryear_Incendiary = np.asarray(data_Incendiary.groupby('iyear').iyear.count())

peryear_Melee = np.asarray(data_Melee.groupby('iyear').iyear.count())

peryear_Vehicle = np.asarray(data_Vehicle.groupby('iyear').iyear.count())

peryear_Sabotage = np.asarray(data_Sabotage.groupby('iyear').iyear.count())

peryear_Other = np.asarray(data_Other.groupby('iyear').iyear.count())
# Group data per year for the world

terror_peryear_world = np.asarray(data_terrorism.groupby('iyear').iyear.count())

terror_years = np.arange(1970, 2016)



# Plot graph

trace1 = go.Scatter(                             

         x = terror_years,

         y = peryear_Biological,

         mode = 'lines',

         line = dict(

             color = 'rgb(140, 140, 45)',

             width = 3),

        name = 'Biological '

         )

trace2 = go.Scatter(                             

         x = terror_years,

         y = peryear_Chemical,

         mode = 'lines',

         line = dict(

             color = 'rgb(240, 40, 45)',

             width = 3),

        name = 'Chemical '

         )

trace3 = go.Scatter(                             

         x = terror_years,

         y = peryear_Radiological,

         mode = 'lines',

         line = dict(

             color = 'rgb(120, 120,120)',

             width = 3),

        name = 'Radiological'

         )

trace4 = go.Scatter(                             

         x = terror_years,

         y = peryear_Nuclear,

         mode = 'lines',

         line = dict(

             color = 'rgb(0, 50, 72)',

             width = 3),

        name = 'Nuclear'

         )

trace5 = go.Scatter(                             

         x = terror_years,

         y = peryear_firearms,

         mode = 'lines',

         line = dict(

             color = 'rgb(27, 135 , 78)',

             width = 3),

        name = 'firearms'

         )

trace6 = go.Scatter(                             

         x = terror_years,

         y = peryear_Explosives,

         mode = 'lines',

         line = dict(

             color = 'rgb(230, 230, 230)',

             width = 3),

        name = 'Explosives'

         )

trace7 = go.Scatter(                             

         x = terror_years,

         y = peryear_fake,

         mode = 'lines',

         line = dict(

             color = 'rgb(238, 133, 26)',

             width = 3),

        name = 'fake weapons'

         )



trace8 = go.Scatter(                             

         x = terror_years,

         y = peryear_Incendiary,

         mode = 'lines',

         line = dict(

             color = 'rgb(238, 133, 26)',

             width = 3),

        name = 'Incendiary'

         )



trace9 = go.Scatter(                             

         x = terror_years,

         y = peryear_Melee,

         mode = 'lines',

         line = dict(

             color = 'rgb(238, 133, 26)',

             width = 3),

        name = 'Melee'

         )



trace10 = go.Scatter(                             

         x = terror_years,

         y = peryear_Vehicle,

         mode = 'lines',

         line = dict(

             color = 'rgb(238, 133, 26)',

             width = 3),

        name = 'Vehicle'

         )



trace11 = go.Scatter(                             

         x = terror_years,

         y = peryear_Sabotage,

         mode = 'lines',

         line = dict(

             color = 'rgb(238, 133, 26)',

             width = 3),

        name = 'Sabotage'

         )



trace12 = go.Scatter(                             

         x = terror_years,

         y = peryear_Other,

         mode = 'lines',

         line = dict(

             color = 'rgb(238, 133, 26)',

             width = 3),

        name = 'Other'

         )



layout = go.Layout(

         title = 'Weapeon type Used (1970-2015)',

         xaxis = dict(

             rangeslider = dict(thickness = 0.05),

             showline = True,

             showgrid = False

         ),

         yaxis = dict(

             range = [0.1, 7500],

             showline = True,

             showgrid = False)

         )



data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12]



figure = dict(data = data, layout = layout)

iplot(figure)
ax = sns.regplot(x="weaptype1", y="nperps", data=data_terrorism)
data_terrorism[~(data_terrorism.nperps < 1)]

ax = sns.regplot(x="weaptype1", y="nperps", data=data_terrorism, x_estimator=np.mean, logx=True, truncate=True)
# terrorist attack weapons grouped in categories

data_terrorism['iday'][data_terrorism.iday == 0] = 1



weapon_codes = []



for attack in data_terrorism ['weaptype1'].values:

    if attack in ['6', '11']:

        weapon_codes.append(1)

    elif attack == '8':

        weapon_codes.append(2)

    elif attack in ['5', '7']:

        weapon_codes.append(3)

    elif attack == '9':

        weapon_codes.append(5)

    elif attack == '1':

        weapon_codes.append(6)

    elif attack in ['2', '3']:

        weapon_codes.append(7)

    elif attack == '10':

        weapon_codes.append(8)

    else:

        weapon_codes.append(4)



data_terrorism['weaptype1'] = weapon_codes

weapon_categories = ['Explosives', 'Flammables', 'Firearms', 'Miscellaneous',

                     'Knives', 'Bacteria/Viruses', 'Chemicals', 'Vehicles']



# terrorist attacks by weapon

weapon_count = np.asarray(data_terrorism.groupby('weaptype1').weaptype1.count())

weapon_percent = np.round(weapon_count / sum(weapon_count) * 100, 2)



# terrorist attack fatalities by weapon

weapon_fatality = np.asarray(data_terrorism.groupby('weaptype1')['nkill'].sum())

weapon_yaxis = np.asarray([1.93, 1.02, 2.28, 0.875, 0.945, 0.83, 0.835, 3.2])



# terrorist attack injuries by weapon

weapon_injury = np.asarray(data_terrorism.groupby('weaptype1')['nwound'].sum())

weapon_xaxis = np.log10(weapon_injury)







weapon_fatality[6] = 7

    

data = [go.Scatter(

        x = weapon_injury,

        y = weapon_fatality,

        text = weapon_text,

        mode = 'markers',

        hoverinfo = 'text',

        marker = dict(

            size = (weapon_count + 50) / 10,

            opacity = 0.9,

            color = 'rgb(240, 140, 45)')

        )]



layout = go.Layout(

         title = 'Terrorist Attacks by Weapon in the world (1970-2015)',

         xaxis = dict(

             title = 'Injuries',

             type = 'log',

             range = [0.45, 3.51],

             tickmode = 'auto',

             nticks = 4,

             showline = True,

             showgrid = False

         ),

         yaxis = dict(

             title = 'Fatalities',

             type = 'log',

             range = [0.65, 3.33],

             tickmode = 'auto',

             nticks = 3,

             showline = True,

             showgrid = False)

         )



annotations = []

for i in range(0, 8):

    annotations.append(dict(x=weapon_xaxis[i], y=weapon_yaxis[i],

                            xanchor='middle', yanchor='top',

                            text=weapon_categories[i], showarrow=False))

layout['annotations'] = annotations



figure = dict(data = data, layout = layout)

iplot(figure)