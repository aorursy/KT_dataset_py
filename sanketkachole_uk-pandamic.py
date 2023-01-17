# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import os.path, time, datetime
DataFile = "/kaggle/input/uk-daily-confirmed-cases/UKDailyConfirmedCases.csv"



Data = pd.read_csv(DataFile, encoding = "iso-8859-1", dayfirst=True, parse_dates=['DateVal'])

Data.head(200)

Data = Data.fillna(0)
Data.head(5)
UKDailyScatter = go.Figure()



UKDailyScatter.add_trace(go.Scatter(x=Data.DateVal, y=Data["CMODateCount"], name="New Cases", line_color="green"))

UKDailyScatter.add_trace(go.Scatter(x=Data.DateVal, y=Data["DailyDeaths"], name="Daily Deaths", line_color="crimson"))





UKDailyScatter.update_layout(title_text=" Public Health England COVID-19 UK daily deaths and new cases from 29th April death figures include hospital and elsewhere."

                            ,barmode="stack", xaxis_rangeslider_visible=True , 

     annotations=

        [dict(

            x="2020-03-23",

            y=967,

            xref="x",

            yref="y",

            text="Lockdown Begins",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

        ), 

             dict(

            x="2020-04-16",

            y=4618,

            xref="x",

            yref="y",

            text="Lockdown Extended",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

),

         dict(

            x="2020-05-13",

            y=3403,

            xref="x",

            yref="y",

            text="1st lockdown easing",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

        ),

        dict(

            x="2020-06-01",

            y=1540,

            xref="x",

            yref="y",

            text="2nd lockdown easing",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

        )

        ])









        

UKDailyScatter.show()

UKDailyScatter = go.Figure()

UKDailyScatter.add_trace(go.Scatter(x=Data.DateVal, y=Data["CumCases"], name="New Cases", line_color="green"))

UKDailyScatter.add_trace(go.Scatter(x=Data.DateVal, y=Data["DailyDeaths"], name="Daily Deaths", line_color="crimson"))



UKDailyScatter.update_layout(title_text=" Cumulative Cases"

                            ,barmode="stack", xaxis_rangeslider_visible= False, 



     annotations=

        [dict(

            x="2020-03-23",

            y=967,

            xref="x",

            yref="y",

            text="Lockdown Begins",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

        ),

         dict(

            x="2020-04-16",

            y=103930,

            xref="x",

            yref="y",

            text="Increased Twice a day",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

         ),

         dict(

            x="2020-05-24",

            y=261184,

            xref="x",

            yref="y",

            text="Increased Twice a week",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

         )

             

])

        

UKDailyScatter.show()

UKDailyScatter = go.Figure()

UKDailyScatter.add_trace(go.Scatter(x=Data.DateVal, y=Data["DailyDeaths"], name="New Cases", line_color="green"))





UKDailyScatter.update_layout(title_text=" Daily Deaths "

                            ,barmode="stack", xaxis_rangeslider_visible= False ,

                             

    annotations=

        [dict(

            x="2020-03-27",

            y=181,

            xref="x",

            yref="y",

            text="Boris Johnson tested Positive",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

        ),

         dict(

            x="2020-04-09",

            y=931,

            xref="x",

            yref="y",

            text="Boris Johnson in ICU",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

        )

        ]

)







        

UKDailyScatter.show()
UKDailyScatter = go.Figure()

UKDailyScatter.add_trace(go.Scatter(x=Data.DateVal, y=Data["IncreasePercent"], name="New Cases", line_color="green"))





UKDailyScatter.update_layout(title_text=" Percentage Increase"

                            ,barmode="stack", xaxis_rangeslider_visible= False ,

                             

      annotations=                       

         [dict(

            x="2020-02-05",

            y=0,

            xref="x",

            yref="y",

            text="First Transmission of Covid - 19",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

         ),

         

         dict(

            x="2020-03-15",

            y=31,

            xref="x",

            yref="y",

            text="over-70s asked to self-isolate",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100)

          ,

         

         dict(

            x="2020-06-19",

            y=0,

            xref="x",

            yref="y",

            text="Sudden fall",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100)

         ]

)







        

UKDailyScatter.show()
UKDailyScatter = go.Figure()

UKDailyScatter.add_trace(go.Scatter(x=Data.DateVal, y=Data["DeathPercent"], name="New Cases", line_color="green"))





UKDailyScatter.update_layout(title_text=" Death-Percentage Increase"

                            ,barmode="stack", xaxis_rangeslider_visible= False 

)







        

UKDailyScatter.show()
UKDailyScatter = go.Figure()

UKDailyScatter.add_trace(go.Scatter(x=Data.DateVal, y=Data["CumCases7DayAvg"], name="New Cases", line_color="green"))





UKDailyScatter.update_layout(title_text=" Cumulative Cases 7-DayAvg"

                            ,barmode="stack", xaxis_rangeslider_visible= False ,

                             

            annotations=

        [dict(

            x="2020-03-23",

            y=4168,

            xref="x",

            yref="y",

            text="All pubs, cafes, restaurants, bars and gyms closed",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

        ),

         dict(

            x="2020-04-07",

            y=42561,

            xref="x",

            yref="y",

            text="police’s new enforcement powers",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

        ),

         

         dict(

            x="2020-05-10",

            y=205600,

            xref="x",

            yref="y",

            text="Three-step “conditional” ease-plan  begin",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

        )

         

        ] )







        

UKDailyScatter.show()
UKDailyScatter = go.Figure()

UKDailyScatter.add_trace(go.Scatter(x=Data.DateVal, y=Data["CumDeaths7DayAvg"], name="New Cases", line_color="green"))





UKDailyScatter.update_layout(title_text=" Cumulative Deaths 7-DayAvg"

                            ,barmode="stack", xaxis_rangeslider_visible= False ,

    annotations=

        [dict(

            x="2020-05-05",

            y=27874,

            xref="x",

            yref="y",

            text="Number of people die surpassing Italy to become the highest toll in Europe",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

        )]

)







        

UKDailyScatter.show()
UKDailyScatter = go.Figure()

UKDailyScatter.add_trace(go.Scatter(x=Data.DateVal, y=Data["DailyDeath7DayAvg"], name="New Cases", line_color="green"))





UKDailyScatter.update_layout(title_text=" Daily Death 7-DayAvg "

                            ,barmode="stack", xaxis_rangeslider_visible= False ,

                             

       annotations=

        [dict(

            x="2020-04-13",

            y=837,

            xref="x",

            yref="y",

            text="Daily Death 7-Day avg Peak",

            showarrow=True,

            align="center",

            arrowhead=7,

            arrowsize=1,

            arrowwidth=2,

            borderpad=6,

            ax=0,

            ay=-100

        )]

)







        

UKDailyScatter.show()
UKDailyScatter = go.Figure()

UKDailyScatter.add_trace(go.Scatter(x=Data.DateVal, y=Data["CMODateCount7DayAvg"], name="New Cases", line_color="green"))





UKDailyScatter.update_layout(title_text=" CMO Date Count 7-DayAvg "

                            ,barmode="stack", xaxis_rangeslider_visible= False 

)







        

UKDailyScatter.show()
import plotly.graph_objects as go

fig = go.Figure(

    data=[

        go.Bar(x=Data.DateVal, y=Data["CumCases"]),

        go.Bar(x=Data.DateVal, y=Data["CumDeaths"]),

       

    ])

fig.update_layout(barmode='stack', title_text='Stacked bar-plot for New cases and daily deaths')

fig.show()





Data1 = pd.read_excel("/kaggle/input/uk-regional-death-count/Figure_1__London_had_the_highest_proportion_of_deaths_involving_the_coronavirus_(COVID-19)_between_March_and_May.xls")
Data1.head(20)
import plotly.graph_objects as go







fig = go.Figure(data=[go.Pie(labels=Data1["Region"], values=Data1["Death - Covid 19"])])

fig.update_layout(title_text="Region wise Covid-19 Deaths from 1st Mar - 31st May")

fig.show()


fig = go.Figure()

fig.add_trace(go.Bar(

    y=Data1["Region"],

    x= Data1["Death - Covid 19"],

    name='Covid 19',

    orientation='h',

    marker=dict(

        color='rgba(246, 78, 139, 1.0)',

        line=dict(color='rgba(246, 78, 139, 1.0)', width=3)

    )

))

fig.add_trace(go.Bar(

    y=Data1["Region"],

    x=Data1["Deaths - Non - covid 19"],

    name='Non-covid 19',

    orientation='h',

    marker=dict(

        color='rgba(58, 71, 80, 1.0)',

        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)

    )

))



fig.update_layout(barmode='stack')

fig.show()
