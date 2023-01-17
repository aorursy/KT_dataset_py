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
import datetime


# Plotly Libraris
import plotly.express as px
import plotly.graph_objects as go


import warnings
warnings.filterwarnings("ignore")
shoot=pd.read_csv('../input/us-police-shootings/shootings.csv')
df=shoot.copy()
df.head()
df.info()
df.describe()
df.shape
#Date
df['date']=pd.to_datetime(shoot['date'])
df['year']=pd.to_datetime(shoot['date']).dt.year
df['month']=pd.to_datetime(shoot['date']).dt.month
df['month_name']=df['date'].dt.strftime('%B')

# Age
df['age_freq']=np.where(df['age']<18,'<18',np.where((df['age']>17)&(df['age']<=30),'18-30',
np.where((df['age']>30)&(df['age']<=40),'31-40',np.where(df['age']>50,'50+',
np.where((df['age']>40)&(df['age']<=50),'41-50',"Not Specified")))))
# Count
df['Count']=1
df.head(5)
df.info()
# 2015 Monthly death report (1)

# Bar Chart - Gradient & Text Position

df_year=df[df['year']==2015]
df_month=df_year['month_name'].value_counts().reset_index().rename(columns={'index':'month_name','month_name':'Count'})

# Sort month

custom_dict ={"January":0,"February":1,"March":2, "April":3,"May":4,"June":5,"July":6,"August":7,"September":8,"October":9,"November":10,"December":11}
df_month['month_name'] = pd.Categorical(df_month['month_name'], categories=sorted(custom_dict, key=custom_dict.get), ordered=True)
df_month=df_month.sort_values('month_name').reset_index(drop=True)



fig = go.Figure(go.Bar(
    x=df_month['month_name'],y=df_month['Count'],
    marker={'color': df_month['Count'], 
    'colorscale': 'Viridis'},  
    text=df_month['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='2015 Monthly death report',xaxis_title="Month",yaxis_title="Number of death",title_x=0.5)
fig.show()
# 2015 Monthly death report (2)

# Basic Line Plot

df_year=df[df['year']==2015]
df_month=df_year['month_name'].value_counts().reset_index().rename(columns={'index':'month_name','month_name':'Count'})


custom_dict ={"January":0,"February":1,"March":2, "April":3,"May":4,"June":5,"July":6,"August":7,"September":8,"October":9,"November":10,"December":11}
df_month['month_name'] = pd.Categorical(df_month['month_name'], categories=sorted(custom_dict, key=custom_dict.get), ordered=True)
df_month=df_month.sort_values('month_name').reset_index(drop=True)

fig = go.Figure(data=go.Scatter(x=df_month['month_name'],
                                y=df_month['Count'],
                                mode='lines')) # hover text goes here
fig.update_layout(title='Monthly Deaths over time',xaxis_title="Date",yaxis_title="Number of Deaths",title_x=0.5)
fig.show()
# 2015 Monthly death report (3)

# Basic Pie

fig = go.Figure([go.Pie(labels=df_month2015['month_name'], values=df_month2015['Count'])])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12,insidetextorientation='radial')

fig.update_layout(title="Month",title_x=0.5)
fig.show()
# Types of line plot

#2015

df_year=df[df['year']==2015]
df_month2015=df_year['month_name'].value_counts().reset_index().rename(columns={'index':'month_name','month_name':'Count'})

# Sort month

custom_dict ={"January":0,"February":1,"March":2, "April":3,"May":4,"June":5,"July":6,"August":7,"September":8,"October":9,"November":10,"December":11}
df_month2015['month_name'] = pd.Categorical(df_month2015['month_name'], categories=sorted(custom_dict, key=custom_dict.get), ordered=True)
df_month2015=df_month2015.sort_values('month_name').reset_index(drop=True)

#2016

df_year=df[df['year']==2016]
df_month2016=df_year['month_name'].value_counts().reset_index().rename(columns={'index':'month_name','month_name':'Count'})

# Sort month

custom_dict ={"January":0,"February":1,"March":2, "April":3,"May":4,"June":5,"July":6,"August":7,"September":8,"October":9,"November":10,"December":11}
df_month2016['month_name'] = pd.Categorical(df_month2016['month_name'], categories=sorted(custom_dict, key=custom_dict.get), ordered=True)
df_month2016=df_month2016.sort_values('month_name').reset_index(drop=True)
df_month2016

#2017

df_year=df[df['year']==2017]
df_month2017=df_year['month_name'].value_counts().reset_index().rename(columns={'index':'month_name','month_name':'Count'})

# Sort month

custom_dict ={"January":0,"February":1,"March":2, "April":3,"May":4,"June":5,"July":6,"August":7,"September":8,"October":9,"November":10,"December":11}
df_month2017['month_name'] = pd.Categorical(df_month2017['month_name'], categories=sorted(custom_dict, key=custom_dict.get), ordered=True)
df_month2017=df_month2017.sort_values('month_name').reset_index(drop=True)
df_month2017

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_month2015['month_name'], y=df_month2015['Count'], name = '2015-Dot',
                         line=dict(color='royalblue', width=4,dash="dot")))

fig.add_trace(go.Scatter(x=df_month2016['month_name'], y=df_month2016['Count'], name = '2016-Dashdot',
                         line=dict(color='green', width=4,dash="dashdot")))

fig.add_trace(go.Scatter(x=df_month2017['month_name'], y=df_month2017['Count'], name = '2017-Dash',
                         line=dict(color='brown', width=4,dash="dash")))
fig.update_layout(title='Monthly Deaths over time different years',xaxis_title="Month",yaxis_title="Number of Deaths",title_x=0.5)
fig.show()
# Simple Bubble Plot

df_manner_of_death=df['manner_of_death'].value_counts().to_frame().reset_index().rename(columns={'index':'manner_of_death','manner_of_death':'Count'})

fig = go.Figure(data=[go.Scatter(
    x=df_manner_of_death['manner_of_death'], y=df_manner_of_death['Count'],
    mode='markers',
    marker=dict(
        size=df_manner_of_death['Count']*0.04))]) # Multiplying by 0.04 to reduce size and stay uniform accross all points

fig.update_layout(title='Manner of Death',xaxis_title="Class",yaxis_title="Number of Deaths",title_x=0.5)
fig.show()
# Bubble Plot with Color gradient

df['age_category']=np.where((df['age']<19),"below 19",
                                 np.where((df['age']>18)&(df['age']<=30),"19-30",
                                    np.where((df['age']>30)&(df['age']<=50),"31-50",
                                                np.where(df['age']>50,"Above 50","NULL"))))

age=df['age_category'].value_counts().to_frame().reset_index().rename(columns={'index':'age_category','age_category':'Count'})


fig = go.Figure(data=[go.Scatter(
    x=age['age_category'], y=age['Count'],
    mode='markers',
    marker=dict(
        color=age['Count'],
        size=age['Count']*0.05,
        showscale=True
    ))])

fig.update_layout(title='Age Frequency ',xaxis_title="Age Category",yaxis_title="Number of Deaths",title_x=0.5)
fig.show()
# Bar Chart - Gradient & Text Position

armed=df['armed'].value_counts()[:12].to_frame().reset_index().rename(columns={'index':'armed','armed':'Count'})

fig = go.Figure(go.Bar(
    x=armed['armed'],y=armed['Count'],
    marker={'color': armed['Count'], 
    'colorscale': 'Viridis'},  
    text=armed['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Top 12 Weapons',xaxis_title="Weapons",yaxis_title="Number of Weapons",title_x=0.5)
fig.show()
# Bar Chart - Stack/Group

df["year"]=pd.to_datetime(shoot['date']).dt.year

ca2_gun=df[(df['state']=='CA')&
           ((df['year']==2015)|(df['year']==2016)|(df['year']==2017)|(df['year']==2018)|(df['year']==2019)|(df['year']==2020))]['year'].value_counts().to_frame().reset_index().rename(columns={'index':'year','year':'count'})
tx2_gun=df[(df['state']=='TX')&
           ((df['year']==2015)|(df['year']==2016)|(df['year']==2017)|(df['year']==2018)|(df['year']==2019)|(df['year']==2020))]['year'].value_counts().to_frame().reset_index().rename(columns={'index':'year','year':'count'})
  

fig = go.Figure()
fig.add_trace(go.Bar(x=ca2_gun['year'],
                y=ca2_gun['count'],
                name='California',
                marker_color='royalblue'
                ))
fig.add_trace(go.Bar(x=tx2_gun['year'],
                y=tx2_gun['count'],
                name='Texas',
                marker_color='violet'
                ))

fig.update_layout(title_text='Past 5 years',xaxis_title="Year",yaxis_title="Number of Death",
                  barmode='stack',title_x=0.5) # by default it is group, else barmode='group'
fig.show()
# Frequency of Race (1)

# 9.Horizontal Bar Chart 

df_category=df['race'].value_counts().reset_index().rename(columns={'index':'race','race':'count'}).sort_values('count',ascending="False")

fig = go.Figure(go.Bar(y=df_category['race'], x=df_category['count'], # Need to revert x and y axis
                      orientation="h")) # default orentation value is "v" - vertical ,we need to change it as orientation="h"
fig.update_layout(title_text=' Race  Frequency ',xaxis_title="Count",yaxis_title="Race",title_x=0.5)
fig.show()
# Frequency of Race (2)

# Pie with custom colors

df_race=df['race'].value_counts().to_frame().reset_index().rename(columns={'index':'race','race':'count'})

colors=['lightcyan','cyan','royalblue','blue','darkblue',"darkcyan"]
fig = go.Figure([go.Pie(labels=df_race['race'], values=df_race['count'])])
fig.update_traces(hoverinfo='label+percent', textinfo='percent+value', textfont_size=15,
                 marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(title="Race Frequency",title_x=0.5)
fig.show()
# Sunburst Gradient

sun_df=df[['gender','manner_of_death','race','Count']].groupby(['gender','manner_of_death','race']).agg('sum').reset_index()

fig = px.sunburst(sun_df, path=['gender','manner_of_death','race'], values='Count',
                  color=sun_df['Count'],
                  color_continuous_scale='orrd') 
fig.update_layout(title="Death distribution by Sex, Manner Of Death and Race",title_x=0.5)
fig.show()
#11.Distribution of Age 

# Basic Box Plot

df_age=df['age']

fig = go.Figure(go.Box(y=df_age,name=" Age")) # to get Horizonal plot change axis :  x=df_age
fig.update_layout(title="Distribution of Age")
fig.show()
#11.1 Distribution of Age With Race 

# Grouped Box Plot

df_ageW=df[df['race']=="White"]['age']
df_ageB=df[df['race']=="Black"]['age']
df_ageH=df[df['race']=="Hispanic"]['age']
df_ageA=df[df['race']=="Asian"]['age']
df_ageN=df[df['race']=="Native"]['age']
df_ageO=df[df['race']=="Other"]['age']

fig = go.Figure()
fig.add_trace(go.Box(y=df_ageW,
                     marker_color="cyan",
                     name="White Age"))
fig.add_trace(go.Box(y=df_ageB,
                     marker_color="darkcyan",
                     name="Black Age" ))
fig.add_trace(go.Box(y=df_ageH,
                     marker_color="royalblue",
                     name="Hispanic Age "))
fig.add_trace(go.Box(y=df_ageA,
                     marker_color="navy",
                     name="Asian Age "))
fig.add_trace(go.Box(y=df_ageN,
                     marker_color="darkblue",
                     name="Native Age "))
fig.add_trace(go.Box(y=df_ageO,
                     marker_color="blue",
                     name="Other Age "))
fig.update_layout(title="Distribution of Age with Race")
fig.show()
#11.2 Distribution of White Age

# Violin Boxplot

df_age=df[df['race']=="White"]['age']

fig = go.Figure(data=go.Violin(y=df_age, box_visible=True, line_color='black',
                               meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,
                               x0='White age'))

fig.update_layout(yaxis_zeroline=False,title="Distribution of White age")
fig.show()
# Basic Histogram

df_age=df['age']

fig = go.Figure(data=[go.Histogram(x=df_age,  # To get Horizontal plot ,change axis - y=df_age
                                  marker_color="green",
                       xbins=dict(
                      start=10, #start range of bin
                      end=100,  #end range of bin
                      size=10   #size of bin
                      ))])
fig.update_layout(title="Age Distribution Of The Deceased",xaxis_title="Age",yaxis_title="Counts",title_x=0.5)
fig.show()
#  Bar Chart - Gradient & Text Position 

df_top10=df.groupby('city')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)
df_top10=df_top10.head(12)

fig = go.Figure(go.Bar(
    x=df_top10['city'],y=df_top10['Count'],
    marker={'color': df_month['Count'], 
    'colorscale': 'Viridis'},  
    text=df_month['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Top 12 City death report',xaxis_title="City",yaxis_title="Number of death",title_x=0.5)
fig.show()
# Pie with custom colors

df_mental=df[df['signs_of_mental_illness']==True]

df_grouped=df_mental.groupby('race')['Count'].sum().reset_index()


df_mental=df[df['signs_of_mental_illness']==True]

df_grouped=df_mental.groupby('race')['Count'].sum().reset_index()

colors=['lightcyan','cyan','royalblue','blue','darkblue',"darkcyan"]
fig = go.Figure([go.Pie(labels=df_grouped['race'], values=df_grouped['Count'])])
fig.update_traces(hoverinfo='label+percent', textinfo='percent+value', textfont_size=15,
                 marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(title="Signs Of Mental Illness Categories",title_x=0.5)
fig.show()
#  Bar Chart - Gradient & Text Position

df_threat_level=df.groupby('threat_level')['Count'].sum().reset_index()


fig = go.Figure(go.Bar(
    x=df_threat_level['threat_level'],y=df_threat_level['Count'],
    marker={'color': df_threat_level['Count'], 
    'colorscale': 'Viridis'},  
    text=armed['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Threat Level',yaxis_title="Number of Threat level",title_x=0.5)
fig.show()
# Facet Bar Chart

df_facet=df[['gender','race','threat_level','age']].groupby(['gender','race','threat_level']).agg('mean').reset_index()

fig = px.bar(df_facet, x="gender", y="age",color="race",barmode="group",
             facet_col="threat_level",
             )
fig.update_layout(title_text='Death Persons AVG Age with Gender,Race,Threat Level',title_x=0.1)
fig.show()
