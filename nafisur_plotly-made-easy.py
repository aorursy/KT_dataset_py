import pandas as pd
import numpy as np
import cufflinks as cf
import plotly
plotly.offline.init_notebook_mode()
cf.go_offline()
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')
fig = data.sort_values('Global_Sales', ascending=False)[:10]
fig = fig.pivot_table(index=['Name'], values=['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'], aggfunc=np.sum)
fig = fig.sort_values('Global_Sales', ascending=True)
fig = fig.drop('Global_Sales', axis=1)
fig = fig.iplot(kind='barh', barmode='stack' , asFigure=True) #For plotting with Plotly
fig.layout.margin.l = 350 #left margin distance
fig.layout.xaxis.title='Sales in Million Units'# For setting x label
fig.layout.yaxis.title='Title' # For setting Y label
fig.layout.title = "Top 10 global game sales" # For setting the graph title
plotly.offline.iplot(fig) # Show the graph
data['Critic_Score'].iplot(kind='histogram',opacity=.75,title='Critic Score Distribution')

fig = data.pivot_table(index=['Year_of_Release'], values=['NA_Sales','EU_Sales','JP_Sales','Other_Sales'], 
                       aggfunc=np.sum, dropna=False,).iplot( asFigure=True,xTitle='Year',yTitle='Sales in Million',title='Yearly Sales By region')
plotly.offline.iplot(fig)
fig = (data.pivot_table(index=['Year_of_Release'], values=['Global_Sales'], columns=['Genre'], aggfunc=np.sum, dropna=False,)['Global_Sales']
        .iplot(subplots=True, subplot_titles=True, asFigure=True, fill=True,title='Sales per year by Genre'))
fig.layout.height= 800
fig.layout.showlegend=False 
plotly.offline.iplot(fig)
fig = (data.pivot_table(index=['Year_of_Release'], values=['Global_Sales'], columns=['Genre'], aggfunc=np.sum, dropna=False,)['Global_Sales']
        .iplot(subplots=True, subplot_titles=True, asFigure=True, fill=True,title='Sales per year by Genre'))
fig.layout.height= 800
fig.layout.showlegend=False 
for key in fig.layout.keys():
    if key.startswith('yaxis'):
        fig.layout[key].range = [0, 145]
plotly.offline.iplot(fig)
top_publish=data['Publisher'].value_counts().sort_values(ascending=False)[:10]
top_publish=data[data['Publisher'].isin(top_publish.index)].groupby(['Publisher','Year_of_Release'])['Name'].count().reset_index()
top_publish=top_publish.pivot('Year_of_Release','Publisher','Name')
top_publish[top_publish.columns[:-1]].iplot(kind='heatmap',colorscale='RdYlGn',title='Publisher Games Releases By Years')