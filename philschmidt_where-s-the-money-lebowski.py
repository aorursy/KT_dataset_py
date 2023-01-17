import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')



# subsample

df = pd.read_csv("../input/PS_20174392719_1491204439457_log.csv")#, nrows=int(1e6))

df.head()
df.info()
df.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
df_corr = df[['amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFraud']]



data = [

    go.Heatmap(

        z=df_corr.corr().values,

        x=df_corr.columns.values,

        y=df_corr.columns.values,

        colorscale='Viridis',

        text = True ,

        opacity = 1.0

        

    )

]





layout = go.Layout(

    title='Pearson Correlation of all numeric features',

    #xaxis = dict(ticks='', nticks=36),

    #yaxis = dict(ticks='' ),

    #width = 900, height = 700,

    

)





fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='labelled-heatmap')
df.groupby(['type', 'isFraud']).count()['step']
df = df.sample(int(5e5))
plt.figure(figsize=(12,8))

sns.boxplot(x = 'isFraud', y = 'amount', data = df[df.amount < 1e5])
plt.figure(figsize=(12,8))

sns.boxplot(hue = 'isFraud', x = 'type', y = 'amount', data = df[df.amount < 1e5])
plt.figure(figsize=(12,8))

sns.pairplot(df[['amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFraud']], hue='isFraud')
from scipy.stats import probplot

fig = plt.figure()

ax = fig.add_subplot(111)



probplot(df['amount'], plot=ax)