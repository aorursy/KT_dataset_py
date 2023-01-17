# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import plotly_express as px
df = pd.read_csv('/kaggle/input/covid19-confirmed-cases-as-of-5th-april/covid-confirmed-cases-since-100th-case.csv')
df.head()
df.columns = df.columns.str.strip().str.lower()
df.date = pd.to_datetime(df.date)
#daily comparison of pak v india

px.line((df.loc[(df.entity=='India') | (df.entity=='Pakistan')]), 'date','total confirmed cases (cases)',color='entity',color_discrete_sequence=['red','darkgreen'],

       title='Total Confirmed Covid19 Cases')
px.line((df.loc[(df.entity=='United States') | (df.entity=='India')| (df.entity=='Pakistan') | (df.entity=='China') | (df.entity=='Germany')]), 'days since the 100th total confirmed case (days)','total confirmed cases (cases)',color='entity',

       hover_data=['date'], labels={'entity':'Country', 'days since the 100th total confirmed case (days)': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases'},

       title='<b>Trend after the 100th case of Covid19 is confirmed. Data updated till: 3rd April.</b><br>The date on which the 100th case was confirmed is different for each country.<br>However, the tally is started on that day making it easier for a comparison of the rate of infection.',

       template='plotly_white')
#-- us is nuts

px.line((df.loc[(df.entity=='United States') | (df.entity=='United Kingdom')| (df.entity=='Pakistan') | (df.entity=='China') | (df.entity=='Germany')]), 'days since the 100th total confirmed case (days)','total confirmed cases (cases)',color='entity',

       hover_data=['date'], labels={'entity':'Country', 'days since the 100th total confirmed case (days)': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases'},

       title='<b>Trend after the 100th case of Covid19 is confirmed. Data updated till: 3rd April.</b><br>Pakistan sab say alag, sab say juda.',

       template='plotly_white')
#-- India jaisa







px.line((df.loc[(df.entity=='India')| (df.entity=='Pakistan')|(df.entity=='Sweden')]), 'days since the 100th total confirmed case (days)','total confirmed cases (cases)',color='entity',

       hover_data=['date'], labels={'entity':'Country', 'days since the 100th total confirmed case (days)': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases'},

       title='<b>Trend after the 100th case of Covid19 is confirmed. Data updated till: 3rd April.</b><br>Khair India jaisa.<br>Over 6k tou easy by 13th April.',

        color_discrete_sequence=["indianred", "darkgreen", "lightskyblue", "goldenrod", "magnta"],

       template='plotly_white')
px.scatter(df.loc[(df.entity=='Pakistan')&(df['days since the 100th total confirmed case (days)'].notna())],'days since the 100th total confirmed case (days)','total confirmed cases (cases)',trendline='ols',

          template='plotly_white',title='Pakistan covid19 cases somehow maintaining a strong linear relationship.<br>y=124x+33, where x=days after 100th confirmed case. Rsquared=0.967.',

          labels={'entity':'Country', 'days since the 100th total confirmed case (days)': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases'},

          hover_data=['date'],

          )
pak_good = df.loc[df['days since the 100th total confirmed case (days)']== df.loc[df.entity=='Pakistan']['days since the 100th total confirmed case (days)'].max()].reset_index(drop=True)
# color_discrete_sequence=["indianred", "darkgreen", "lightskyblue", "goldenrod", "magnta"]

pak_good = pak_good.sort_values('total confirmed cases (cases)',ascending=False)

pak_good = pak_good.reset_index(drop=True)

coloring=['darkslategrey',] * pak_good.entity.nunique()

coloring[pak_good.loc[pak_good['entity']=='Pakistan'].index.item()] = 'darkgreen'

coloring[pak_good.loc[pak_good['entity']=='India'].index.item()] = 'red'
import plotly.graph_objects as go
xday = df.loc[df.entity=='Pakistan']['days since the 100th total confirmed case (days)'].max()

todaysdate = pd.datetime.now().strftime("%d/%m/%Y")
fig = go.Figure(data=[go.Bar(

    x=pak_good['entity'],

    y=pak_good['total confirmed cases (cases)'],

    marker_color=coloring # marker color can be a single color value or an iterable

)])

fig.update_layout(template='plotly_white',title_text=f"<b>{todaysdate} is Pakistan's {xday} day after the 100th confirmed case.</b><br>The rest of the world at this time were:")
double_time = df.loc[(df.entity=='Pakistan') & (df['days since the 100th total confirmed case (days)'] >= 0)].reset_index(drop=True)



double_time['day'] = double_time.index 
double_time['2days'] = double_time['total confirmed cases (cases)'] *2



double_time['2days'] = double_time['total confirmed cases (cases)'][0] ** double_time.day



double_time['pct_change'] = double_time['total confirmed cases (cases)'].pct_change()
x0 = double_time['total confirmed cases (cases)'][0]



r = round(double_time['pct_change'].mean(),2)





double_time['2days'] = x0 * (1 + r)**double_time.day
fig = px.scatter(double_time,'day','total confirmed cases (cases)',trendline='ols', color='entity',

          template='plotly_white',title=f'Pakistan covid19 cases somehow maintaining a strong linear relationship.<br>OLS. y=125.2x + 27, where x=days after 100th confirmed case. Rsquared=0.97.<br>Transmission rate a little under {r*100}%.',

          labels={'entity':'Actual', 'days since the 100th total confirmed case (days)': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases'},

          hover_data=['date'],

          )

fig.add_trace(

    go.Scatter(

        x=double_time.day,

        y=double_time['2days'],

        mode="lines",

        line=go.scatter.Line(color="gray"), name=f"If rate of transmission is {r*100}%",

        showlegend=True)

)

fig.show()
fig = px.bar(double_time,'date','pct_change',title="Percentage increase of Pakistan's confirmed Covid19 cases per day<br>No idea how we completely skipped days in between.",

      template='plotly_white', labels={'pct_change':'Percentage Change','date':'Date'})



fig.update_layout(

    showlegend=False,

    annotations=[

        dict(

            x='2020-03-24',

            y=0.133,

            xref="x",

            yref="y",

            text="Lockdown announced",

            showarrow=True,

            arrowhead=1,

            ax=45,

            ay=-80

        )

    ]

)

# fig.show()