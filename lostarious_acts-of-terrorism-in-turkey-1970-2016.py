from __future__ import division

import plotly as py

from plotly import tools

from plotly.offline import iplot, init_notebook_mode,plot

import pandas as pd

from collections import Counter

init_notebook_mode()

data_cols = [1,2,3,8,12,13,14,29,35,58,98,101]

df = pd.read_csv('../input/globalterrorismdb_0617dist.csv',error_bad_lines=False,encoding='ISO-8859-1',usecols=data_cols,na_values="Unknown")

data = df.rename( columns={ 'iyear':'year', 'imonth':'month', 'iday':'day',

             'country_txt':'country','attacktype1_txt':"attacktype",

                                  'targtype1_txt':'target',

              'nkill':'fatalities', 'nwound':'injuries'})

tr_dat = pd.DataFrame(columns = data.columns.values )

sizes = []

colors = []

tr_dat['text']=""

cnt = 0

tr_dat.is_copy = False

for i,row in data.iterrows():

    if row['country'] == "Turkey":

        a = data.iloc[i]

        tr_dat.loc[cnt] = a

        tr_dat['text'][cnt] = str(a).replace("\n", "<br>")

        sizes.append((row['fatalities']+row['injuries'])/7)

        if tr_dat['fatalities'][cnt] == 'NaN' or tr_dat['fatalities'][cnt] == 0:

            colors.append("rgb(0,255,0)")

        else:

            colors.append("rgb(0,0,0)")

        cnt+=1

data = [ dict(

        type = 'scattergeo',



        lon = tr_dat['longitude'],

        lat = tr_dat['latitude'],

        text = tr_dat['text'],

        mode = 'markers',

        marker = dict(

            size = sizes,

            sizemin = 3,

            color = colors,

            opacity = 0.8,

            reversescale = True,

            autocolorscale = False,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            ),

        ))]

layout = dict(

        title = 'Terrorist attacks in Turkey(1970-2016)',

        width = 1000,

        height = 800,

        geo = dict(

            resolution = 50,

            lonaxis = dict( range= [26, 45]),

            lataxis = dict( range= [36, 42] ),

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.5,

            subunitwidth = 0.5,

            showcoastlines = True,

            showcountries = True,

           

        ),

    )

fig = dict( data=data, layout=layout )

iplot( fig,)
tattack = len(tr_dat['fatalities'])

cmncty = Counter(tr_dat['city'])

cmntrgt = Counter(tr_dat['target'])

cmngrp = Counter(tr_dat['gname'])

cmntype = Counter(tr_dat['attacktype'])

import plotly.graph_objs as go

data = [

    go.Bar(

        x = [x[0] for x in cmngrp.most_common(10)],

        y = [x[1] for x in cmngrp.most_common(10)],

         marker=dict(

        color='rgb(158,202,225)',

        line=dict(

            color='rgb(8,48,107)',

            width=1.5,

        ),

        ),

        text = ["%{:.2f} of total attacks.".format((x[1]/tattack)*100) for x in cmngrp.most_common(10) ],

        ),

        

    ]

layout = go.Layout(

title = "Terrorist groups that made the most number of attacks.",

)

fig = dict(data = data,layout = layout)

iplot(fig)

    
total = tr_dat['fatalities'].sum(skipna=True)+tr_dat['injuries'].sum(skipna=True)

print("Total number of attacks: {}\nTotal number of fatalities and injuries: {}".format(tattack,total))
data = [{

            'labels': [x[0] for x in cmntype.most_common(10)],

            'values': [x[1] for x in cmntype.most_common(10)],

            'type': 'pie',

            'name': 'Attack Types',

            'domain': {'x': [0, .48],

                       'y': [.51, 1]},

            'hoverinfo':'label+percent+name',

            'textinfo':'Total:'+'value'

},

{

            'labels': [x[0] for x in cmncty.most_common(10)],

            'values': [x[1] for x in cmncty.most_common(10)],

            'type': 'pie',

            'domain': {'x': [.52, 1],

                       'y': [.51, 1]},

            'name': 'Cities',

            'hoverinfo':'label+percent+name+value+text',

            'textinfo':'Total: '+'value'

}

]

layout = dict(title = "Most common attack types and cities that were attacked.",

              showlegend = False,

             width = 800,

             height = 800)

fig = dict(data = data,layout = layout)

iplot(fig)
yearly = Counter(tr_dat['year'])

data = [

go.Scatter(

    x = [x for x in range(1970,2017)],

    y = [yearly[x] for x in range(1970,2017)],

    mode = 'lines+markers',

    

)]

layout = go.Layout(title = "Attacks Per Year")

fig=go.Figure(data= data,layout=layout)

iplot(fig)  