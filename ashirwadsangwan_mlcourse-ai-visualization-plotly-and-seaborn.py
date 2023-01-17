import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

import matplotlib

matplotlib.rcParams['figure.figsize'] = (15,8)

#Plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



from warnings import filterwarnings as fw

fw('ignore')

%matplotlib inline



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/beauty.csv')

data.head()
plt.figure(figsize=(12,8))

sns.heatmap(data.corr(),cmap='PiYG',linewidths=0.5, annot=True), ;
data.looks.nunique()
plt.figure(figsize=(15,8))

data['looks'].value_counts().plot(kind='bar');

plt.xlabel('Looks',fontsize = 14);

plt.ylabel('#People',fontsize = 14);

plt.title('Distribution of Looks',fontsize = 18);
plt.figure(figsize=(15,8))

sns.distplot(data['wage'], fit = norm);
plt.figure(figsize=(15,8))

plt.plot(data.groupby('educ')['wage'].mean());

plt.title('Average Variation in Wage with Education Level');

plt.xlabel('Education Level');

plt.ylabel('Mean Wage Level');
data.female.value_counts()
plt.figure(figsize = (15,7))

sns.countplot(data.educ ,hue=data.female);
pd.crosstab(data['female'],data['married'])
df = pd.read_csv('../input/telecom_churn.csv')

df.head()
df['Churn'] = df['Churn']*1 # To make the column numeric
df.Churn.value_counts().plot(kind='bar');
plt.figure(figsize=(15,15))

sns.heatmap(df.corr(),cmap='PiYG',annot=True,linewidths=0.5);
trace1 = go.Scatter(

                    x = df['Customer service calls'],

                    y = df['Total day charge'],

                    mode = "lines",

                    name = "Day Charge",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df['Total day calls'])



trace2 = go.Scatter(

                    x = df['Customer service calls'],

                    y = df['Total eve charge'],

                    mode = "lines",

                    name = "Evening Charge",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df['Total eve calls'])



trace3 = go.Scatter(

                    x = df['Customer service calls'],

                    y = df['Total night charge'],

                    mode = "lines",

                    name = "Day Charge",

                    marker = dict(color = 'rgba(255, 128, 10, 0.8)'),

                    text= df['Total night calls'])



data = [trace1, trace2, trace3]



layout = dict(title = 'Charges paid by Customers vs Service Calls',

              xaxis= dict(title= 'Customer Service Calls',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
df['Total minutes']=df['Total day minutes']+df['Total eve minutes']+df['Total night minutes']

df['Total calls'] = df['Total day calls'] + df['Total eve calls'] + df['Total night calls']

trace1 = go.Histogram(

    x=df['Total minutes'],

    opacity=0.75,

    name = "Total Minutes",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

    x=df['Total calls'],

    opacity=0.75,

    name = "Total Calls",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title='Total Minutes vs Total Calls',

                   xaxis=dict(title='Minutes and Calls Distrubution'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df['Total charge'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge']

trace0 = go.Box(

    y=df['Total calls'],

    name = '# Total Calls',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=df['Total minutes'],

    name = '# Total Minutes',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

         

)

trace2 = go.Box(

    y=df['Total charge'],

    name = '# Total Charge',

    marker = dict(

        color = 'rgb(256, 128, 128)',

    )         

)         

data = [trace0, trace1, trace2]

iplot(data)
sns.pairplot(df, hue='Churn');
sns.lmplot(x='Total calls', y='Total charge', hue='Churn', markers='o',

           fit_reg=False, data = df);
_, axes = plt.subplots(1, 3, sharey=True, figsize=(20, 6))

sns.countplot(x = 'Customer service calls', hue = 'Churn', data=df);

sns.countplot(x='International plan', hue='Churn', data=df, ax=axes[0]);

sns.countplot(x='Voice mail plan', hue='Churn', data=df, ax=axes[1]);
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler
dataframe = df.drop(['Churn', 'State'], axis=1)

dataframe['International plan'] = dataframe['International plan'].map({'Yes': 1, 'No': 0})

dataframe['Voice mail plan'] = dataframe['Voice mail plan'].map({'Yes': 1, 'No': 0})



# Scaling for tSNE

scaler = StandardScaler()

scaled_df = scaler.fit_transform(dataframe)
%%time

tsne = TSNE(random_state=17)

tsne_repr = tsne.fit_transform(scaled_df)
plt.figure(figsize=(15,8))

plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1]);
plt.figure(figsize=(15,8))

plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1],

            c=df['Churn'].map({False: 'blue', True: 'orange'}), alpha=.7);
_, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))



for i, name in enumerate(['International plan', 'Voice mail plan']):

    axes[i].scatter(tsne_repr[:, 0], tsne_repr[:, 1], 

                    c=df[name].map({'Yes': 'orange', 'No': 'blue'}), alpha=.5);

    axes[i].set_title(name);
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=df['Total calls'],

    y=df['Total charge'],

    z=df['Churn'],

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255,0,0)',                # set color to an array/list of desired values      

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)