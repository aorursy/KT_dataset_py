import numpy as np

import pandas as pd



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()
df = pd.read_csv('../input/bachelor-contestants.csv')
print('There are {:,} unique contestants, {:,} of whom have appeared more than once.'.format(df['Name'].nunique(), len([x for x in df['Name'].value_counts() if x > 1])))
states = {'Alabama':'AL','Alaska':'AK','American Samoa':'AS','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT','Delaware':'DE','District Of Columbia':'DC','D.C':'DC', 'D.C.':'DC','Federated States Of Micronesia':'FM','Florida':'FL','Georgia':'GA','Guam':'GU','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Marshall Islands':'MH','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Northern Mariana Islands':'MP','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Palau':'PW','Pennsylvania':'PA','Puerto Rico':'PR','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virgin Islands':'VI','Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'}
# add a column with the abbreviated states

def get_state(name):

    parts = name.split(",")

    if len(parts) == 2:

        if len(parts[1].strip()) == 2:

            return parts[1].strip()

        elif parts[1].strip() in states:

            return states[parts[1].strip()]

        else:

            return ''

    else:

        return ''



df['state'] = df['Hometown'].map(get_state)

df.head()
print('Of the {} unique contestants, {} are not from the US.'.format(df['Name'].nunique(), len(df[df['state'] == ''])))
df[df['state'] == '']['Hometown']
perstate = df[df['state'] != '']['state'].value_counts().to_dict()



data = [dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = 'Blues',

        reversescale = True,

        locations = list(perstate.keys()),

        locationmode = 'USA-states',

        text = list(perstate.values()),

        z = list(perstate.values()),

        marker = dict(

            line = dict(

                color = 'rgb(255, 255, 255)',

                width = 2)

            ),

        )]



layout = dict(

         title = 'Bachelor contestants by State',

         geo = dict(

             scope = 'usa',

             projection = dict(type = 'albers usa'),

             countrycolor = 'rgb(255, 255, 255)',

             showlakes = True,

             lakecolor = 'rgb(255, 255, 255)')

         )



figure = dict(data = data, layout = layout)

iplot(figure)