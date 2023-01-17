import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode
plt.style.use('ggplot')
gtd = pd.read_csv("../input/globalterrorismdb_0616dist.csv",encoding = "ISO-8859-1",low_memory=False)
#select columns related to locations

gt = gtd[['iyear','country','country_txt','region','region_txt','provstate','city','latitude','longitude']]
gt.head()
#select rows where country is 'United States'

gtUSA = gt[gt['country_txt']=='United States']
#check how many rows we have for 'United States'

gtUSA.info()
#check rows with null values 

gtUSA[gtUSA.isnull().any(axis = 1)]
#only one row has null values; let's drop it

gtUSA = gtUSA.dropna(axis = 0)
# generate a dataframe with numbers of attacks in each state

gtUSA_perstate = pd.DataFrame({'State':gtUSA['provstate'].value_counts().index, 'Counts':gtUSA['provstate'].value_counts().values})
gtUSA_perstate
# Convert state names to their abbreviations.

states_dict = {

         'Alaska':'AK',

         'Alabama':'AL',

         'Arkansas':'AR',

         'Arizona':'AZ',

         'California':'CA',

         'Colorado':'CO',

         'Connecticut':'CT',

         'District of Columbia':'DC',

         'Delaware':'DE',

         'Florida':'FL',

         'Georgia':'GA',

         'Hawaii':'HI',

         'Iowa':'IA',

         'Idaho':'ID',

         'Illinois':'IL',

         'Indiana':'IN',

         'Kansas':'KS',

         'Kentucky':'KY',

         'Louisiana':'LA',

         'Massachusetts':'MA',

         'Maryland':'MD',

         'Maine':'ME',

         'Michigan':'MI',

         'Minnesota':'MN',

         'Missouri':'MO',

         'Mississippi':'MS',

         'Montana':'MT',

         'North Carolina':'NC',

         'North Dakota':'ND',

         'Nebraska':'NE',

         'New Hampshire':'NH',

         'New Jersey':'NJ',

         'New Mexico':'NM',

         'Nevada':'NV',

         'New York':'NY',

         'Ohio':'OH',

         'Oklahoma':'OK',

         'Oregon':'OR',

         'Pennsylvania':'PA',

         'Puerto Rico':'PR',

         'Rhode Island':'RI',

         'South Carolina':'SC',

         'South Dakota':'SD',

         'Tennessee':'TN',

         'Texas':'TX',

         'Utah':'UT',

         'Virginia':'VA',

         'Vermont':'VT',

         'Washington':'WA',

         'Wisconsin':'WI',

         'West Virginia':'WV',

         'Wyoming':'WY'

};
gtUSA_perstate['State'].replace(states_dict,inplace = True);
# a bar plot to show terrorism in each state

fig = plt.figure(figsize=(8,4));

ax = fig.add_subplot(1,1,1);

gtUSA_perstate.plot(kind = 'bar',ax = ax);

ax.set_xticklabels(gtUSA_perstate['State'],size = 8);
# show terrorism in each state on a map

init_notebook_mode()

# plotly code for choropleth map

scale = [[0, 'rgb(229, 239, 245)'],[1, 'rgb(1, 97, 156)']]

data = [ dict(

        type = 'choropleth',

        colorscale = scale,

        autocolorscale = False,

        showscale = False,

        locations = gtUSA_perstate['State'],

        z = gtUSA_perstate['Counts'],

        locationmode = 'USA-states',

        marker = dict(

            line = dict (

                color = 'rgb(255, 255, 255)',

                width = 2

            ) ),

        ) ]

layout = dict(

        title = 'Terrorism in United States (1970-2015)',

        geo = dict(

            scope = 'usa',

            projection = dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)',

            countrycolor = 'rgb(255, 255, 255)')        

             )

 

figure = dict(data=data, layout=layout)

iplot(figure)