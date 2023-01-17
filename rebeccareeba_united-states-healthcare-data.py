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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



import datetime as dt



#import chart_studio.plotly as py

import plotly.graph_objs as go

import plotly.express as px

import glob



df =pd.read_csv('../input/20082019/2008-2019.csv')
df.head()
heatdata = df.copy()

heatdata = heatdata.drop(['YEAR'],axis = 1)

f,ax = plt.subplots(figsize=(20, 20))

sns.heatmap(heatdata.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
#need to remove string values so creating a seperate dataframe with states and ids

state_id = df.STATE.unique()

#Convert to dataframe and add an id column

statedf = pd.DataFrame({'STATE': state_id})

statedf.insert(0, 'ID', range(1, 1 + len(statedf)))

df = df.merge(statedf, left_on='STATE', right_on='STATE')

df_copy = df.copy()



df_copy = df_copy.drop(['Unnamed: 0','STATE'],axis =1)

#df_copy.head()

df2018 = df.loc[df['YEAR']==2018].sort_values('MEDIAN_INCOME')



a4_dims = (20, 20)



fig, ax = plt.subplots(figsize=a4_dims)

sns.set_style('darkgrid')

sns.barplot(x = 'MEDIAN_INCOME', y = 'STATE', data= df2018, color='b')#,palette="Blues_d")
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

df_2018 = df2018.copy()

df_2018 = df_2018.drop(['STATE'],axis=1)



#df_df_2018copy = df_copy.drop(['STATE'],axis =1)

sns.relplot(x="TOTAL_HEALTH_SPENDING", y="MEDIAN_INCOME", data= df2018, size ="UNINSURED", hue = 'STATE', palette = 'Set3',alpha=.7,sizes=(40,1000),height=10)
layout = go.Layout(

title = 'Transaction Value',

yaxis = dict(title = 'median income'),

xaxis = dict(title = 'Healthcare spending')

)



fig = px.scatter(df2018, x="TOTAL_HEALTH_SPENDING", y="MEDIAN_INCOME", color="UNINSURED",

                 size='UNINSURED', hover_data=['STATE'])

fig.show()
df2018.head()
# Scatter plot of 2018 Median_income

data = [

    {

        'y': df2018.PREVENTABLE_HOSPITALIZATIONS,

        'x': df2018.MEDIAN_INCOME,

        'mode': 'markers',

        'marker': {

            'color': df2018.MEDIAN_INCOME,

            'size': df2018.UNINSURED*2.5,

            'showscale': True

        },

        "text" :  df2018.STATE    

    }

]





layout= go.Layout(

    title= 'Scatter plot of preventable hospitalisations by median income in 2018',

    hovermode= 'closest',

    xaxis= dict(

        title= 'Median Income',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Preventable Hospitalisations',

        ticklen= 5,

        gridwidth= 2,

    ),

    annotations=[dict(

            showarrow=False,

            x=1.05,

            y=-0.215,



            text="Radius of points proportional to uninsured percentage",



            xref="paper",

            yref="paper",

            opacity=0.7

        )],

    showlegend= False

)



fig = go.Figure(data=data, layout=layout)

fig.show()
X = np.array(df_2018)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA()

pca.fit(X_scaled)

len(pca.components_)

print(pca.explained_variance_ratio_)
print('Explained Variance Ratio = ', sum(pca.explained_variance_ratio_[: 2]))

#More than half (~56%) of the total variance comes from only two dimensions (i.e., the first two principal components). Let's visualize the relative contribution of all components.
plt.plot(np.cumsum(pca.explained_variance_ratio_))

#So we need about 10 dimensions to explain ~90% of the total variance.
pca.n_components = 2

X_reduced = pca.fit_transform(X_scaled)

df_X_reduced = pd.DataFrame(X_reduced, index=df2018)

#We will graph only the first 2

data = [

    {

        'y': df_X_reduced[1],

        'x': df_X_reduced[0],

        'mode': 'markers',

        'marker': {

            'sizemode': 'diameter',

            'size': df2018['UNINSURED'],

            'sizeref':df2018['UNINSURED'].max()/50,

            'opacity':0.5

        }, "text" :  df2018.STATE  

    }

]





layout= go.Layout(

    title= '(PCA)',

    hovermode= 'closest',

    xaxis= dict(

        showgrid=False,

        zeroline=False,

        showticklabels=False

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showticklabels=False

    ),

    showlegend= False

)



fig = go.Figure(data=data, layout=layout)

fig.show()
from sklearn.cluster import KMeans

# Let the number of clusters be a parameter, so we can get a feel for an appropriate

# value thereof.

def cluster(n_clusters):

    kmeans = KMeans(n_clusters=n_clusters)

    kmeans.fit(X_reduced)

    Z = kmeans.predict(X_reduced)

    return kmeans, Z



max_clusters = len(df2018)

# n_clusters = max_clusters would be trivial clustering.



inertias = np.zeros(max_clusters)



for i in range(1, max_clusters):

    kmeans, Z = cluster(i)

    inertias[i] = kmeans.inertia_
n_clusters = 3

model, Z = cluster(n_clusters)
data = [

    {

        'y': df_X_reduced[1],

        'x': df_X_reduced[0],

        'mode': 'markers',

        'marker': {

            'sizemode': 'diameter',

            'size': df2018['UNINSURED'],

            'sizeref':df2018['UNINSURED'].max()/50,

            'opacity':0.5,

            'color':Z

        }, "text" :  df2018.STATE  

    }

]





layout= go.Layout(

    title= 'Clustering States using K means',

    hovermode= 'closest',

    xaxis= dict(

        showgrid=False,

        zeroline=False,

        showticklabels=False

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showticklabels=False

    ),

    showlegend= False

)



fig = go.Figure(data=data, layout=layout)

fig.show()
list(df_2018.columns.values)
df_X_scaled = df2018[['HEART_ATTACK', 'DIABETES', 'SMOKING', 'CHOLESTEROL_CHECK', 'OBESITY']]



def cluster_nonpca(n_clusters):

    kmeans = KMeans(n_clusters=n_clusters)

    kmeans.fit(df_X_scaled)

    Z = kmeans.predict(df_X_scaled)

    return kmeans, Z

n_clusters = 3

model, Z = cluster_nonpca(n_clusters)
data = [

    {

        'y': df2018['DIABETES'],

        'x': df2018['HEART_ATTACK'],

        'mode': 'markers',

        'marker': {

            'sizemode': 'diameter',

            'size': df2018['DIABETES'],

            'sizeref':df2018['DIABETES'].max()/40,

            'opacity':0.5,

            'color':Z

        }, "text" :  df2018.STATE  

    }

]





layout= go.Layout(

    title= 'Clustering States using K means',

    hovermode= 'closest',

    xaxis= dict(

        showgrid=True,

        zeroline=False,



        title= 'Heart Attack',

        ticklen= 5,

        gridwidth= 2,

    ),

    yaxis=dict(

        showgrid=True,

        zeroline=False,



        title= 'Cholesterol Check',

        ticklen= 5,

        gridwidth= 2,

    ),

    annotations=[dict(

            showarrow=False,

            x=.005,

            y=-0.215,



            text="K means groups the states accoring to the 5 top predictors for Heathcare spending ",



            xref="paper",

            yref="paper",

            opacity=0.7

        )],

    showlegend= False

)





fig = go.Figure(data=data, layout=layout)

fig.show()
!pip install chart_studio
import chart_studio

from chart_studio.plotly import plot, iplot

trace1 = go.Scatter3d(



    x=df2018['DIABETES'],

    y=df2018['OBESITY'],

    z=df2018['SMOKING'],

    mode='markers',

    marker=dict(

        size=df2018['HEART_ATTACK']*10,

        color = Z,                # set color to an array/list of desired values

        colorscale='Viridis',   # choose a colorscale

        opacity=0.5           # set color to an array/list of desired values      

    ), text =  df2018.STATE 

)



data = [trace1]



layout= go.Layout(

        margin=dict(

        l=20,

        r=20,

        b=20,

        t=50  

    ),

    width=1000,

    height=1000,



    hovermode= 'closest',

    showlegend= False

)



fig = go.Figure(data=data, layout=layout)

fig.update_layout(title = 'State clustering using 2018 data',

                  scene = dict(xaxis=dict(title='Diabetes, % of population'),

                               yaxis=dict(title='Obesity, % of population'),

                               zaxis=dict(title='Smoking, % of population'),

               

                           ))



fig.show()