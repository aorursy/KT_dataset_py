# This Python 3 environment comes with many helpful analytics libraries installedimport numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# import warnings library

import warnings        

# ignore filters

warnings.filterwarnings("ignore") # if there is a warning after some codes, this will avoid us to see them.

plt.style.use('ggplot') # style of plots. ggplot is one of the most used style, I also like it.

# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("/kaggle/input/us-counties-covid-19-dataset/us-counties.csv")
dataset.info()
dataset.drop(columns="fips",inplace=True)
dataset.info()
dataset.describe()


data=dataset.groupby("state").sum().reset_index()



dat=[dict(

    type="choropleth",

    locations=['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE',"DC", 'FL', 'GA',"GU", 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND',"NM", 'OH', 'OK', 'OR', 'PA',"PR", 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',"VI", 'VA', 'WA','WV', 'WI', 'WY'],

    locationmode='USA-states',

    z=data["cases"],

    text=data["state"],

    colorscale = [[0,"rgb(5, 10, 172)"],[0.85,"rgb(40, 60, 190)"],[0.9,"rgb(70, 100, 245)"],

            [0.94,"rgb(90, 120, 245)"],[0.97,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

    autocolorscale=False,

    reversescale=True,

    marker=dict(

        line=dict(

            width=0.5,

            color='rgba(100,100,100)',

            ),

        ),

    colorbar=dict(

        #autotick = False,

        #tickprefix="",

        title="Total Cases",

        )  

    )]



layout=dict(

    title={

        'text': "Total Cases of States",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    geo=dict(

        showframe=False,

        showcoastlines=True,

        projection=dict(

            type="albers usa"

            ),

        scope="usa"

        )

    

    )





fig=go.Figure(data=dat,layout=layout)

iplot(fig)
f = plt.figure(figsize=(10,5))

f.add_subplot(111)



plt.axes(axisbelow=True)

plt.barh(data.sort_values('cases')["state"][-10:],data.sort_values('cases')["cases"].values[-10:],color="darkcyan")

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Confirmed Cases",fontsize=18)

plt.title("Top 10 States (Confirmed Cases)",fontsize=20)

plt.grid(alpha=0.3)

f = plt.figure(figsize=(10,5))

f.add_subplot(111)



plt.axes(axisbelow=True)

plt.barh(data.sort_values('deaths')["state"][-10:],data.sort_values("deaths")["deaths"].values[-10:],color="crimson")

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Death Cases",fontsize=18)

plt.title("Top 10 States (Death Cases)",fontsize=20)

plt.grid(alpha=0.3)
a=data.sort_values("deaths")

sizes=a["cases"][-6:]

labels=a["state"][-6:]

explode=[0,0,0,0,0,0]

colors=["orange","red","blue","green","yellow","violet"]

plt.figure(figsize=(7,7))

plt.pie(sizes,explode=[0.1]*6,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Death Rate Top 6 States',color = 'blue',fontsize = 15)
a=[]

for x in dataset["date"]:

    a.append(x.split("-")[1])

dataset["Month"]=a   

data=dataset.groupby("Month").sum().reset_index()

data.replace("01","January",inplace=True)

data.replace("02","February",inplace=True)

data.replace("03","March",inplace=True)

data.replace("04","April",inplace=True)

data.replace("05","Mai",inplace=True)

data.replace("06","June",inplace=True)

cases=[each for each in data.cases]

deaths=[each for each in data.deaths]





pie_list=[each for each in data.cases]

labels=data.Month

fig={

     "data":[

         {

         "values":pie_list,

         "labels":labels,

         "domain":{"x":[0,0.5]},

         "name":"Cases per Month",

         "hoverinfo":"label+percent+name",

         "hole":.3,

         "type":"pie"},],

     

     "layout":{

          "title":"Cases per Month",

          "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Number of Cases",

                "x": 0.20,

                "y": 1.19

             },]

     }

    }

iplot(fig)            

from ipywidgets import widgets



gd = dataset.groupby(["date","state"])

# Function for updating barplot with new values

def update_bar(change):

    bar_out.clear_output()

    try:

        cases = gd.get_group((dates_sl.value, state_dr.value))["cases"].sum()

        deaths = gd.get_group((dates_sl.value, state_dr.value))["deaths"].sum()

    except:

        cases = 0

        deaths = 0

    with bar_out:

        bar_out.clear_output()

        sns.barplot(data = pd.DataFrame({

            "cases":[cases],

            "deaths":[deaths]

        }),palette = ["blue","red"])

        plt.show()



# Dates Slider

dates = np.sort(dataset["date"].unique())

dates_sl = widgets.SelectionSlider(options = dates,continuous_update=False)

dates_sl.observe(update_bar, names = "value")



# States Dropdown

state_dr = widgets.Dropdown(options = np.sort(dataset["state"].unique()))

state_dr.observe(update_bar, names = "value")



bar_out = widgets.Output()



bar_container = widgets.VBox([

    widgets.HBox([state_dr, dates_sl]),

    bar_out

])



bar_container
a=[]

for x in dataset["date"]:

    a.append(x.split("-")[1])

dataset["Month"]=a   

data=dataset.groupby(["Month","state"]).sum().reset_index()

data.replace("01","January",inplace=True)

data.replace("02","February",inplace=True)

data.replace("03","March",inplace=True)

data.replace("04","April",inplace=True)

data.replace("05","Mai",inplace=True)

data.replace("06","June",inplace=True)







Months = [str(each) for each in list(data.Month.unique())] # str unique years

# make list of types

custom_colors = {

    'cases': 'rgb(189, 2, 21)'

}

# make figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}



figure['layout']['geo'] = dict(

        showframe=False,

        showcoastlines=True,

        projection=dict(

            type="albers usa"

            ),

        scope="usa"

        )

figure['layout']['hovermode'] = 'closest'

figure['layout']['sliders'] = {

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': 'January',

    'plotlycommand': 'animate',

    'values': Months,

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

Month = "January"



dataset_by_month = data[data['Month'] == Month]

data_dict = dict(

    type="choropleth",

    locations=['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE',"DC", 'FL', 'GA',"GU", 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND',"NM", 'OH', 'OK', 'OR', 'PA',"PR", 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',"VI", 'VA', 'WA','WV', 'WI', 'WY'],

    locationmode='USA-states',

    z=data["cases"],

    colorscale = [[0,"rgb(5, 10, 172)"],[0.30,"rgb(40, 60, 190)"],[0.60,"rgb(70, 100, 245)"],

            [0.80,"rgb(90, 120, 245)"],[0.97,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

    autocolorscale=False,

    reversescale=True,

    marker=dict(

        line=dict(

            width=0.5,

            color='rgba(100,100,100)',

            ),

        ),

    colorbar=dict(

        #autotick = False,

        #tickprefix="",

        title="Total Cases",

        )  

    )

figure['data'].append(data_dict)

    

# make frames

for Month in Months:

        frame = {'data': [], 'name': str(Month)}

        dataset_by_year = data[data['Month'] == Month]

        data_dict = dict(

            type="choropleth",

            locations=['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE',"DC", 'FL', 'GA',"GU", 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND',"NM", 'OH', 'OK', 'OR', 'PA',"PR", 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',"VI", 'VA', 'WA','WV', 'WI', 'WY'],

            locationmode='USA-states',

            z=dataset_by_year["cases"],

            colorscale = [[0,"rgb(5, 10, 172)"],[0.30,"rgb(40, 60, 190)"],[0.60,"rgb(70, 100, 245)"],

                    [0.80,"rgb(90, 120, 245)"],[0.97,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

            autocolorscale=False,

            reversescale=True,

            marker=dict(

                line=dict(

                    width=0.5,

                    color='rgba(100,100,100)',

                    ),

                ),

            colorbar=dict(

                #autotick = False,

                #tickprefix="",

                title="Total Cases",

                )  

            )

        frame['data'].append(data_dict)

        figure['frames'].append(frame)

        slider_step = {'args': [

            [Month],

            {'frame': {'duration': 500, 'redraw': True},

             'mode': 'immediate',

           'transition': {'duration': 300}}

         ],

         'label': Month,

         'method': 'animate'}

        sliders_dict['steps'].append(slider_step)





figure["layout"]["autosize"]= True

figure["layout"]["title"] = "Animation Map Plot"       



figure['layout']['sliders'] = [sliders_dict]



iplot(figure)
data = dataset.copy()

def cnvrt_month(num):

    return {

        1 : 'Jan',2:'Feb',

        3 : 'Mar',4 : 'Apr',5 : 'May',

        6 : 'Jun',7 : 'Jul',8 : 'Aug',

        9 : 'Sep', 10 : 'Oct',11 : 'Nov',

        12 : 'Dec'

    }[num]

data["Month"] = pd.DatetimeIndex(data["date"]).month

data["Month"] = data["Month"].apply(cnvrt_month)

months = {

        1 : 'Jan',2:'Feb',

        3 : 'Mar',4 : 'Apr',5 : 'May',

        6 : 'Jun'#,7 : 'Jul',8 : 'Aug',

        #9 : 'Sep', 10 : 'Oct',11 : 'Nov',

        #12 : 'Dec'

    }



frame_dict = {}

for k,v in months.items():

    df = data[data["Month"] == v].groupby("state")["cases"].sum().nlargest(n = 5)

    df = df.reset_index()

    frame_dict[k] = df.sort_values(by = ["cases"])
fig = go.Figure(

    data = [

            go.Bar(

                x = frame_dict[1]["cases"], y = frame_dict[1]["state"],orientation = "h",

                text = frame_dict[1]["cases"], textposition = "inside",

                insidetextanchor = "middle"

            )

    ],

    layout = go.Layout(

        xaxis = dict(range=[0,100],autorange = True),

        yaxis = dict(range = [-0.5,5.5],autorange = True),

        title = dict(text = "Top 5 states for total cases in Jan",xanchor = "left"),

        updatemenus=[dict(

            type="buttons",

            buttons=[dict(label="Play",

                          method="animate",

                          args=[None,

                          {"frame": {"duration": 1250, "redraw": True},

                          "transition": {"duration":250}}]

            )]

        )]

    ),

    frames = [

              go.Frame(

                  data = [go.Bar(

                            x = frame_dict[k]["cases"], y = frame_dict[k]["state"],orientation = "h",

                            text = frame_dict[k]["cases"], textposition = "inside",

                            insidetextanchor = "middle"

                          )

                  ],

                  layout = go.Layout(

                            xaxis = dict(range=[14,frame_dict[k]["cases"][0]+frame_dict[k]["cases"][3]],autorange = True),

                            yaxis = dict(range = [-0.5,5],autorange = False),

                            title = dict(text = "Top 5 states for total cases in "+months[k],xanchor = "left"),

                          )

              )

              for k,v in frame_dict.items() 

    ]



)

fig.show()