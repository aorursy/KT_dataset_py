# Importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = (8.0, 5.0)
df = pd.read_csv('../input/imo_results.csv')

df.head()
print("Shape of dataframe: ", df.shape)
# Checking for null values

def na_columns(df):

    [print(d,":\t ", df[d].isna().sum()) for d in df.columns if df[d].isna().sum() > 0]



na_columns(df)
# Data filling 

df.firstname = df.firstname.fillna('None')

df.lastname = df.lastname.fillna('None')

df.problem1 = df.problem1.fillna(0.0)

df.problem2 = df.problem2.fillna(0.0)

df.problem3 = df.problem3.fillna(0.0)

df.problem4 = df.problem4.fillna(0.0)

df.problem5 = df.problem5.fillna(0.0)

df.problem6 = df.problem6.fillna(0.0)

df.award = df.award.fillna('Participant')
# Data Formatting

df.year = df.year.astype('object')

df.problem1 = df.problem1.astype('int')

df.problem2 = df.problem2.astype('int')

df.problem3 = df.problem3.astype('int')

df.problem4 = df.problem4.astype('int')

df.problem5 = df.problem5.astype('int')

df.problem6 = df.problem6.astype('int')

df['name'] = df[['firstname', 'lastname']].apply(lambda x: '_'.join(x), axis=1)
# Remove columns

df = df.drop(columns=['firstname', 'lastname'])
gold_winners = df[df.award == 'Gold medal']

silver_winners = df[df.award == 'Silver medal']

bronze_winners= df[df.award =="Bronze medal"]
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)
def medal_winners(med1, med2, med3):

    

    gold = med1.groupby(['country']).size().reset_index(name='Count')

    silver = med2.groupby(['country']).size().reset_index(name='Count')

    bronze = med3.groupby(['country']).size().reset_index(name='Count')

    

    gold = gold.sort_values(by='Count', ascending=False)[:5]

    silver = silver.sort_values(by='Count', ascending=False)[:5]

    bronze = bronze.sort_values(by='Count', ascending=False)[:5]

      

    trace1 = go.Bar(x=gold.country, y=gold.Count, name='GOLD')

    trace2 = go.Bar(x=silver.country, y=silver.Count, xaxis='x2', yaxis='y2', name='SILVER')

    trace3 = go.Bar(x=bronze.country, y=bronze.Count, xaxis='x3', yaxis='y3', name='BRONZE')



    dt = [trace1, trace2, trace3]

    layout = go.Layout(

        xaxis=dict(domain=[0, 0.3]),

        xaxis2=dict(domain=[0.33, 0.63]),

        xaxis3=dict(domain=[0.67, 1]),

        yaxis2=dict(anchor='x2'),

        yaxis3=dict(anchor='x3'),

        )

    fig = go.Figure(data=dt, layout=layout)



    fig['layout'].update(title='Top Medal Winning Countries')

    fig['layout']['xaxis'].update(title='Top 5 Gold')

    fig['layout']['xaxis2'].update(title='Top 5 Silver')

    fig['layout']['xaxis3'].update(title='Top 5 Bronze')

    fig['layout']['yaxis'].update(title='Medal Count')



    iplot(fig)
medal_winners(gold_winners,silver_winners,bronze_winners)
# Save top 5 medal winners

gwp = gold_winners['name'].value_counts().sort_values(ascending=True)[-5:]

swp = silver_winners['name'].value_counts().sort_values(ascending=True)[-5:]

bwp = bronze_winners['name'].value_counts().sort_values(ascending=True)[-5:]
gwp = gwp.reset_index()

swp = swp.reset_index()

bwp = bwp.reset_index()
from plotly import tools

def medal_participants(med1, med2, med3):

          

    trace1 = go.Bar(y=med1.iloc[:,0], x=med1.iloc[:,1], name='GOLD', orientation='h')

    trace2 = go.Bar(y=med2.iloc[:,0], x=med2.iloc[:,1], xaxis='x2', yaxis='y2', name='SILVER',  orientation='h')

    trace3 = go.Bar(y=med3.iloc[:,0], x=med3.iloc[:,1], xaxis='x3', yaxis='y3', name='BRONZE',  orientation='h')



    dt = [trace1, trace2, trace3]

    

    layout = go.Layout(

        autosize=False,

        yaxis2=dict(anchor='y2'),

        yaxis3=dict(anchor='y3'),

    )

    

    fig = go.Figure(data=dt,layout=layout)

    fig = tools.make_subplots(rows=3, cols=1, subplot_titles=('Top 5 Gold participants','Top 5 Silver participants','Top 5 Bronze participants'))



    fig.append_trace(trace1, 1, 1)

    fig.append_trace(trace2, 2, 1)

    fig.append_trace(trace2, 3, 1)

    

    fig['layout'].update(height=600, width=800)

    fig['layout']['yaxis'].update(tickangle=-45)

    fig['layout']['yaxis2'].update(tickangle=-45)

    fig['layout']['yaxis3'].update(tickangle=-45)

    fig['layout'].update(title='Top 5 Medal Winning Participants')



    iplot(fig)
medal_participants(gwp,swp,bwp)