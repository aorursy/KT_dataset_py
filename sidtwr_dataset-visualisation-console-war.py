from plotly.offline import init_notebook_mode, iplot

from IPython.display import display, HTML



import pandas as pd



init_notebook_mode(connected=True)



dataset = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")

print(list(dataset.columns.values))



dataset['Critic_Score'] = dataset['Critic_Score'].astype(float)

mean = dataset['Critic_Score'].mean()

dataset['Critic_Score'].fillna(mean, inplace=True)



dataset['User_Score'] = dataset['User_Score'].astype(float)

mean = dataset['User_Score'].mean()

dataset['User_Score'].fillna(mean, inplace=True)



dataset['Global_Sales'] = dataset['Global_Sales'].apply(lambda x: x*10000000)



start_year = 1980

end_year = 2018

years = []

for i in range(end_year-start_year):

    years.append(str(start_year+i))



# make list of Publishers

Publishers = ["Nintendo","Activision","Capcom","Crystal Dynamics","Microsoft Game Studios","Sony Computer Entertainment",

              "Sony Computer Entertainment America","Sony Computer Entertainment Europe","Ubisoft",

              "Crystal Dynamics", "Electronic Arts", "Sega","Take-Two Interactive"]



figure = {

    'data': [],

    'layout': {},

    'frames': []

}



# fill in most of layout

figure['layout']['xaxis'] = {'range': [0, 120], 'title': 'Critic_Score'}

figure['layout']['yaxis'] = {'range': [0, 11], 'title': 'User_Score'}

figure['layout']['hovermode'] = 'closest'

figure['layout']['sliders'] = {

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1980',

    'plotlycommand': 'animate',

    'values': years,

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

year = 1980

for Publisher in Publishers:

    dataset_by_year = dataset[dataset['Year_of_Release'] == year]

    dataset_by_year_and_cont = dataset_by_year[dataset_by_year['Publisher'] == Publisher]



    data_dict = {

        'x': list(dataset_by_year_and_cont['Critic_Score']),

        'y': list(dataset_by_year_and_cont['User_Score']),

        'mode': 'markers',

        'text': list(dataset_by_year_and_cont['Name']),

        'marker': {

            'sizemode': 'area',

            'sizeref': 200000,

            'size': list(dataset_by_year_and_cont['Global_Sales'])

        },

        'name': Publisher

    }

    figure['data'].append(data_dict)





# make frames

for year in years:

    frame = {'data': [], 'name': str(year)}

    for Publisher in Publishers:

        dataset_by_year = dataset[dataset['Year_of_Release'] == int(year)]

        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['Publisher'] == Publisher]



        data_dict = {

            'x': list(dataset_by_year_and_cont['Critic_Score']),

            'y': list(dataset_by_year_and_cont['User_Score']),

            'mode': 'markers',

            'text': list(dataset_by_year_and_cont['Name']),

            'marker': {

                'sizemode': 'area',

                'sizeref': 200000,

                'size': list(dataset_by_year_and_cont['Global_Sales'])

            },

            'name': Publisher

        }

        frame['data'].append(data_dict)



    figure['frames'].append(frame)

    slider_step = {'args': [

        [year],

        {'frame': {'duration': 300, 'redraw': False},

         'mode': 'immediate',

         'transition': {'duration': 300}}

    ],

        'label': year,

        'method': 'animate'}

    sliders_dict['steps'].append(slider_step)



figure['layout']['sliders'] = [sliders_dict]



iplot(figure)

import plotly

import plotly.graph_objs as go

import numpy as np

import pandas as pd



plotly.tools.set_credentials_file(username='sid.tiwari4', api_key='4JG0q4BGteFYGWaBm19q')



dataset = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")

print(list(dataset.columns.values))



df = dataset.groupby(['Year_of_Release', 'Platform'])["Name"].agg("nunique")

df = df.reset_index()

print(df.head())



Sony = ["PS4", "PS3", "PS2", "PSP", "PSV", "PS"]

Microsoft = ["XB", "X360", "XOne"]

PC = ["PC"]

Nintendo = ["NES","SNES","N64","NGC","Wii","WiiU","GB","GBA"]



years = df.Year_of_Release.unique()



for year in years:

    specific_year = df[df["Year_of_Release"] == year]

    platform = list(specific_year.Platform)

    not_platform = ((set(Sony) | set(Microsoft) | set(PC)) | set(Nintendo)) - set(platform)

    if not_platform:

        for element in not_platform:

            add = pd.DataFrame({"Year_of_Release": [year],

                                "Platform": [element],

                                "Name": [str(0)]

                                })

            df = df.append(add, ignore_index=True)



# Nintendo --------------------------------------------------------------------------



data1 = df[df["Platform"] == "NES"].sort_values(by=["Year_of_Release"])

data1['Name'] = pd.to_numeric(data1["Name"])

traces_NES = (go.Scatter(

    x=data1.Year_of_Release,

    y=data1.Name,

    name="Nintendo_NES",

    line=dict(color='#ff6666', width=1),

    opacity=0.8))



data2 = df[df["Platform"] == "SNES"].sort_values(by=["Year_of_Release"])

data2['Name'] = pd.to_numeric(data2["Name"])

traces_SNES = (go.Scatter(

    x=data2.Year_of_Release,

    y=data2.Name,

    name="Nintendo_SNES",

    line=dict(color='#ff6666', width=1),

    opacity=0.8))



data3 = df[df["Platform"] == "N64"].sort_values(by=["Year_of_Release"])

data3['Name'] = pd.to_numeric(data3["Name"])

traces_N64 = (go.Scatter(

    x=data3.Year_of_Release,

    y=data3.Name,

    name="Nintendo_N64",

    line=dict(color='#ff6666', width=1),

    opacity=0.8))



data4 = df[df["Platform"] == "NGC"].sort_values(by=["Year_of_Release"])

data4['Name'] = pd.to_numeric(data4["Name"])

traces_NGC = (go.Scatter(

    x=data4.Year_of_Release,

    y=data4.Name,

    name="Nintendo_NGC",

    line=dict(color='#ff6666', width=1),

    opacity=0.8))



data5 = df[df["Platform"] == "Wii"].sort_values(by=["Year_of_Release"])

data5['Name'] = pd.to_numeric(data5["Name"])

traces_Wii = (go.Scatter(

    x=data5.Year_of_Release,

    y=data5.Name,

    name="Nintendo_Wii",

    line=dict(color='#ff0000', width=4),

    opacity=0.8))



data6 = df[df["Platform"] == "WiiU"].sort_values(by=["Year_of_Release"])

data6['Name'] = pd.to_numeric(data6["Name"])

traces_WiiU = (go.Scatter(

    x=data6.Year_of_Release,

    y=data6.Name,

    name="Nintendo_WiiU",

    line=dict(color='#ff0000', width=4),

    opacity=0.8))



data7 = df[df["Platform"] == "GB"].sort_values(by=["Year_of_Release"])

data7['Name'] = pd.to_numeric(data7["Name"])

traces_GB = (go.Scatter(

    x=data7.Year_of_Release,

    y=data7.Name,

    name="Nintendo_GameBoy",

    line=dict(color='#ff3333', width=3),

    opacity=0.8))



data8 = df[df["Platform"] == "GBA"].sort_values(by=["Year_of_Release"])

data8['Name'] = pd.to_numeric(data8["Name"])

traces_GBA = (go.Scatter(

    x=data8.Year_of_Release,

    y=data8.Name,

    name="Nintendo_GameBoyAdvance",

    line=dict(color='#ff3333', width=3),

    opacity=0.8))



Nintendo = pd.DataFrame(

    {

        "Year_of_Release": list(data1.Year_of_Release),

        "Num_of_titles": list(np.asarray(data1.Name) + np.asarray(data2.Name) + np.asarray(data3.Name) + np.asarray(data4.Name)

                              + np.asarray(data5.Name) + np.asarray(data6.Name)+ np.asarray(data7.Name) + np.asarray(data8.Name))

    }

)



traces_Nintendo = (go.Scatter(

    x=Nintendo.Year_of_Release,

    y=Nintendo.Num_of_titles,

    name="Nintendo",

    line=dict(color='#cc0000', width=4, dash='dash'),

    opacity=0.8))



print(df[df["Platform"] == "PS2"])



# Sony Computer Entertainment -------------------------------------------------------

data1 = df[df["Platform"] == "PS4"].sort_values(by=["Year_of_Release"])

data1['Name'] = pd.to_numeric(data1["Name"])

traces_PS4 = (go.Scatter(

    x=data1.Year_of_Release,

    y=data1.Name,

    name="Sony_PS4",

    line=dict(color='#002080', width=5),

    opacity=0.8))



data2 = df[df["Platform"] == "PS3"].sort_values(by=["Year_of_Release"])

data2['Name'] = pd.to_numeric(data2["Name"])

traces_PS3 = (go.Scatter(

    x=data2.Year_of_Release,

    y=data2.Name,

    name="Sony_PS3",

    line=dict(color='#002db3', width=4),

    opacity=0.8))



data3 = df[df["Platform"] == "PS2"].sort_values(by=["Year_of_Release"])

data3['Name'] = pd.to_numeric(data3["Name"])

traces_PS2 = (go.Scatter(

    x=data3.Year_of_Release,

    y=data3.Name,

    name="Sony_PS2",

    line=dict(color='#0039e6', width=3),

    opacity=0.8))



data4 = df[df["Platform"] == "PSP"].sort_values(by=["Year_of_Release"])

data4['Name'] = pd.to_numeric(data4["Name"])

traces_PSP = (go.Scatter(

    x=data4.Year_of_Release,

    y=data4.Name,

    name="Sony_PSP",

    line=dict(color='#1a53ff', width=2),

    opacity=0.8))



data5 = df[df["Platform"] == "PS"].sort_values(by=["Year_of_Release"])

data5['Name'] = pd.to_numeric(data5["Name"])

traces_PS = (go.Scatter(

    x=data5.Year_of_Release,

    y=data5.Name,

    name="Sony_PS",

    line=dict(color='#4d79ff', width=1),

    opacity=0.8))



sony = pd.DataFrame(

    {

        "Year_of_Release": list(data1.Year_of_Release),

        "Num_of_titles": list(np.asarray(data1.Name) + np.asarray(data2.Name) + np.asarray(data3.Name) + np.asarray(data4.Name) + np.asarray(data5.Name))

    }

)



traces_sony = (go.Scatter(

    x=sony.Year_of_Release,

    y=sony.Num_of_titles,

    name="Sony_Playstaion",

    line=dict(color='#002080', width=4, dash='dash'),

    opacity=0.8))



# Microsoft ------------------------------------------------------------------

data1 = df[df["Platform"] == "XB"].sort_values(by=["Year_of_Release"])

data1['Name'] = pd.to_numeric(data1["Name"])

traces_XB = (go.Scatter(

    x=data1.Year_of_Release,

    y=data1.Name,

    name="Microsoft_XB",

    line=dict(color='#00e600', width=2),

    opacity=0.8))



data2 = df[df["Platform"] == "X360"].sort_values(by=["Year_of_Release"])

data2['Name'] = pd.to_numeric(data2["Name"])

traces_X360 = (go.Scatter(

    x=data2.Year_of_Release,

    y=data2.Name,

    name="Microsoft_X360",

    line=dict(color=' #00b300', width=3),

    opacity=0.8))



data3 = df[df["Platform"] == "XOne"].sort_values(by=["Year_of_Release"])

data3['Name'] = pd.to_numeric(data3["Name"])

traces_XOne = (go.Scatter(

    x=data3.Year_of_Release,

    y=data3.Name,

    name="Microsoft_XOne",

    line=dict(color='#008000', width=4),

    opacity=0.8))



microsoft = pd.DataFrame(

    {

        "Year_of_Release": list(data1.Year_of_Release),

        "Num_of_titles": list(np.asarray(data1.Name) + np.asarray(data2.Name) + np.asarray(data3.Name))

    }

)



traces_microsoft = (go.Scatter(

    x=microsoft.Year_of_Release,

    y=microsoft.Num_of_titles,

    name="Microsoft_Xbox",

    line=dict(color='#008000', width=4, dash='dash'),

    opacity=0.8))



# PC --------------------------------------------------------------------------

data = df[df["Platform"] == "PC"].sort_values(by=["Year_of_Release"])

traces_PC = (go.Scatter(

    x=data.Year_of_Release,

    y=data.Name,

    name="PC",

    line=dict(color='#666666', width=4, dash='dash'),

    opacity=0.8))



data = [traces_PS4, traces_PS3, traces_PS2, traces_PSP, traces_PS, traces_sony,

        traces_XB, traces_X360, traces_XOne, traces_microsoft

    , traces_PC

    , traces_NES, traces_SNES, traces_N64, traces_NGC, traces_Wii, traces_WiiU, traces_GB, traces_GBA,traces_Nintendo]



layout = dict(

    title='Time Series with Rangeslider',

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=6,

                     label='6m',

                     step='month',

                     stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(

            visible=True

        ),

        type='date'

    )

)



fig = dict(data=data, layout=layout)

plotly.plotly.iplot(fig, filename="Time Series with Rangeslider")
import plotly.plotly as py

import plotly

import plotly.graph_objs as go

import numpy as np

import pandas as pd



plotly.tools.set_credentials_file(username='sid.tiwari4', api_key='4JG0q4BGteFYGWaBm19q')



dataset = pd.read_csv("../input/PS4_GamesSales.csv",encoding = "cp1252")

print(dataset.head())



some_values = ["Sony Computer Entertainment", "Sony Interactive Entertainment"]

df = dataset[dataset["Publisher"].isin(some_values)]

df = df.sort_values(by=['Global']).tail(12)

print(df)



x = list(df.Game)

y = list(df["Japan"])

y2 = list(df["Europe"])

y3 = list(df["North America"])

y4 = list(df["Global"])



trace1 = go.Bar(

    x=x,

    y=y,

    text="Japan",

    textposition = 'auto',

    marker=dict(

        color='rgb(77, 77, 255)',

        line=dict(

            color='rgb(77, 77, 255)',

            width=1.5),

        ),

    opacity=0.6

)



trace2 = go.Bar(

    x=x,

    y=y2,

    text="Europe",

    textposition = 'auto',

    marker=dict(

        color='rgb(26, 26, 255)',

        line=dict(

            color='rgb(26, 26, 255)',

            width=1.5),

        ),

    opacity=0.6

)



trace3 = go.Bar(

    x=x,

    y=y3,

    text="North America",

    textposition = 'auto',

    marker=dict(

        color='rgb(0, 0, 230)',

        line=dict(

            color='rgb(0, 0, 230)',

            width=1.5),

        ),

    opacity=0.6

)



trace4 = go.Bar(

    x=x,

    y=y4,

    text="Global",

    textposition = 'auto',

    marker=dict(

        color='rgb(0, 0, 179)',

        line=dict(

            color='rgb(0, 0, 179)',

            width=1.5),

        ),

    opacity=0.6

)



data = [trace1,trace2,trace3,trace4]



py.iplot(data, filename='grouped-bar-direct-labels')
import plotly.plotly as py

import plotly

import plotly.graph_objs as go

import numpy as np

import pandas as pd



plotly.tools.set_credentials_file(username='sid.tiwari4', api_key='4JG0q4BGteFYGWaBm19q')



dataset = pd.read_csv("../input/XboxOne_GameSales.csv",encoding = "cp1252")

print(dataset.head())



some_values = ["Microsoft Studios", "Microsoft Game Studios"]

df = dataset[dataset["Publisher"].isin(some_values)]

df = df.sort_values(by=['Global']).tail(12)

print(df)



x = list(df.Game)

y = list(df["Japan"])

y2 = list(df["Europe"])

y3 = list(df["North America"])

y4 = list(df["Global"])



trace1 = go.Bar(

    x=x,

    y=y,

    text="Japan",

    textposition = 'auto',

    marker=dict(

        color='rgb(26, 255, 26)',

        line=dict(

            color='rgb(26, 255, 26)',

            width=1.5),

        ),

    opacity=0.6

)



trace2 = go.Bar(

    x=x,

    y=y2,

    text="Europe",

    textposition = 'auto',

    marker=dict(

        color='rgb(0, 230, 0)',

        line=dict(

            color='rgb(0, 230, 0)',

            width=1.5),

        ),

    opacity=0.6

)



trace3 = go.Bar(

    x=x,

    y=y3,

    text="North America",

    textposition = 'auto',

    marker=dict(

        color='rgb(0, 179, 0)',

        line=dict(

            color='rgb(0, 179, 0)',

            width=1.5),

        ),

    opacity=0.6

)



trace4 = go.Bar(

    x=x,

    y=y4,

    text="Global",

    textposition = 'auto',

    marker=dict(

        color='rgb(0, 128, 0)',

        line=dict(

            color='rgb(0, 128, 0)',

            width=1.5),

        ),

    opacity=0.6

)



data = [trace1,trace2,trace3,trace4]



py.iplot(data, filename='grouped-bar-direct-labels')
