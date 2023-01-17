import numpy as np 

import pandas as pd

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.subplots import make_subplots



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_19 = pd.read_csv('../input/flight-delay-prediction/Jan_2019_ontime.csv')

df_19.drop('Unnamed: 21',axis=1,inplace=True)

df_19.head()
df_19.info()
plot1 = df_19.groupby('DAY_OF_MONTH')['CANCELLED'].count()

fig = go.Figure()



fig.add_trace(go.Bar(x=plot1.index, y=plot1.values, name='Cancel bar',opacity=0.9,marker_color='#a5afff'))



fig.add_trace(go.Scatter(x=plot1.index, y=plot1.values, line=dict(color='red'), name='Cancel trend'))

fig.update_layout(

    title="Cancelled flights vs day of month",

    xaxis_title="Day of month",

    yaxis_title="Cancel count",

)

fig.show()
plot2 = df_19.groupby('DAY_OF_MONTH')['DIVERTED'].count()

fig2 = go.Figure()



fig2.add_trace(go.Bar(x=plot2.index, y=plot2.values, name='Diverted bar',opacity=0.6,marker_color='#ff0000'))



fig2.add_trace(go.Scatter(x=plot2.index, y=plot2.values, line=dict(color='#4200ff'), name='Diverted trend'))

fig2.update_layout(

    title="Diverted flights vs day of month",

    xaxis_title="Day of month",

    yaxis_title="Diverted count",

)

fig2.show()

import calendar

yy = 2019 

mm = 1    

print(calendar.month(yy, mm))
all(plot1 == plot2)
plot3 = df_19.groupby('ORIGIN')['CANCELLED'].count().sort_values(ascending=False)

#cap to above 400

plot3 = plot3[plot3>400]

fig3 = px.bar(plot3,x=plot3.index,y=plot3.values,color_discrete_sequence=['#6ec8ba'],

             title="Cancelled flights by origin airport",labels={"ORIGIN":"Origin airport","y":"Count"})

fig3.layout.template = 'plotly_white'

fig3.update_xaxes(tickangle=45)

fig3.show()
plot4 = df_19.groupby('DEST')['CANCELLED'].count().sort_values(ascending=False)

#cap to above 400

plot4 = plot4[plot4>400]

fig4 = px.bar(plot4,x=plot4.index,y=plot4.values,color_discrete_sequence=['#1A7065'],

             title="Cancelled flights by destination airport",labels={"DEST":"Destination airport","y":"Count"})

fig4.layout.template = 'plotly_white'

fig4.update_xaxes(tickangle=45)

fig4.show()
plot5 = df_19.groupby(['ORIGIN','DEST'])['CANCELLED'].count().sort_values(ascending=False)

plot5.index = [i+'-'+j for i,j in plot5.index] 

#cap to above 500 for clear visualization

plot5 = plot5[plot5>500]

fig5 = px.bar(plot5,x=plot5.index,y=plot5.values,color_discrete_sequence=['#B93795'],

             title="Cancelled flights by origin-dest",labels={"index":"Origin-Destination","y":"Count"})

fig5.layout.template = 'plotly_white'

fig5.update_xaxes(tickangle=45)

fig5.show()
plot6 = df_19.groupby('OP_CARRIER')['DEP_DEL15'].sum().sort_values()

fig6 = px.pie(names=plot6.index,values=list(map(int,plot6.values)),

              color_discrete_sequence =px.colors.qualitative.T10, hole=0.5, title='Airlines with most delayed flights')

fig6.show()
plot6 = df_19.groupby('OP_CARRIER')['DISTANCE'].mean().sort_values()

fig6 = px.pie(names=plot6.index,values=list(map(int,plot6.values)),

              color_discrete_sequence =px.colors.qualitative.Pastel, hole=0.5, 

              title='Airlines with most long distance flights')

fig6.show()
plt.Figure(figsize=(30,22))

sns.distplot(df_19[df_19.CANCELLED==0.0].DISTANCE, hist=True)

sns.distplot(df_19[df_19.CANCELLED==1.0].DISTANCE, hist=True)

# cancelled orange

# not cancelled blue
cor = df_19.corr().fillna(0)

fig = px.imshow(cor,

            labels=dict(color="Viridis"),

            x=cor.index,

            y=cor.columns)

fig.update_layout(width=800,height=800)

fig.show()