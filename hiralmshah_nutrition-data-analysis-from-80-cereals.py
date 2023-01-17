# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.offline as py 
py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version
import plotly.graph_objs as go # it's like "plt" of matplot
import plotly.tools as tls # It's useful to we get some tools of plotly
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
cereal_df = pd.read_csv('../input/cereal.csv')
cereal_df.head()

cereal_df['weight'].value_counts()
#calculate all nutritions data in reference with weight.

def nutritions_per_kg(cols):
    for col in cols:
        cereal_df[col] = cereal_df[col]/cereal_df['weight']
    
cols = ['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']
nutritions_per_kg(cols)

cereal_df.corr()
f, ax = plt.subplots(figsize = (18,18))
sns.heatmap(cereal_df.corr(), annot = True, linewidths = 0.5, fmt = '.1f', ax=ax)
plt.show()
cereal_df.columns
#cereal_df.round(1)
cereal_df.round(2)
cereal_df.info()
plt.figure(figsize = (30,10))
sns.set_style('darkgrid')
sns.set(font_scale = 1.5)

plt.subplot(311)

ax = sns.barplot(x='name', y='sugars', data=cereal_df)
ax.set_xticklabels(cereal_df['name'], rotation=90, ha="center")
ax.set(xlabel='Name',ylabel='sugars')
ax.set_title('Sugar level in Cereals')

plt.subplot(312)
ax = sns.barplot(x='name', y='sodium', data=cereal_df)
ax.set_xticklabels(cereal_df['name'], rotation=90, ha="center")
ax.set(xlabel='Name',ylabel='salt')
ax.set_title('Salt level in Cereals')

plt.subplot(313)
ax = sns.barplot(x='name', y='fat', data=cereal_df)
ax.set_xticklabels(cereal_df['name'], rotation=90, ha="center")
ax.set(xlabel='Name',ylabel='Fat')
ax.set_title('Fat level in Cereals')

plt.subplots_adjust(hspace = 2.0, top = 1.5)

plt.show()

#plot1 = sns.barplot()
from plotly.plotly import iplot
from  plotly.offline import plot
import plotly.offline as py 
py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version
import plotly.graph_objs as go # it's like "plt" of matplot
import plotly.tools as tls # It's useful to we get some tools of plotly
import warnings # This library will be used to ignore some warnings

sugar = cereal_df.groupby('name')['sugars'].sum().to_frame().reset_index()
salt = cereal_df.groupby('name')['sodium'].sum().to_frame().reset_index()
fat = cereal_df.groupby('name')['fat'].sum().to_frame().reset_index()

colors = None
trace0 = go.Scatter(x = sugar['name'], y = sugar['sugars'],
               mode = 'lines+markers',
               name = 'sugars')
trace1 = go.Scatter(x=fat['name'], y=fat['fat'],
               mode = 'lines+markers',
               name = 'fat')
#trace2 = go.Scatter(x=salt['name'], y=salt['sodium'],
              # mode = 'lines+markers',
              # name = 'sodium')


layout = dict(title= 'sugar and fat level in each cereal', 
                  font=dict(family='Courier New, monospace', size=9, color='#7f7f7f'),
                  height=400, width=800,)
fig = dict(data=[trace0, trace1], layout = layout)
py.iplot(fig)
#plotly.offline.iplot(fig)
print('Sugar level\n', cereal_df['sugars'].value_counts())
print('Fat level:\n',cereal_df['fat'].value_counts())
print('Sodium or salt level:\n',cereal_df['sodium'].value_counts())

#healthy_cereals = cereal_df.loc[(cereal_df['sugars'] <= 50) & (cereal_df['sodium'] <= 3) 
             #                  & (cereal_df['fat'] <= 30)]
    
cereal_df['healthy'] = np.where((cereal_df['sugars'] <= 5) & (cereal_df['sodium'] <= 0.3) 
                               & (cereal_df['fat'] <= 3)  
                               &(cereal_df['calories'] >= 50), 'yes','no')
cereal_df['healthy'].value_counts()
healthy_cereals = cereal_df.loc[(cereal_df['healthy'] == 'yes') | (cereal_df['rating'] >= 60)]
healthy_cereals
healthy = healthy_cereals.groupby('name')['rating'].sum().to_frame().reset_index()
healthy
trace0 = go.Scatter(
    x = healthy_cereals['name'],
    y = healthy_cereals['sugars'],
    mode = 'lines+markers',
    name = 'sugars'
)
trace1 = go.Scatter(
    x = healthy_cereals['name'],
    y = healthy_cereals['fat'],
    mode = 'lines+markers',
    name = 'fat'
)
trace2 = go.Scatter(
    x = healthy_cereals['name'],
    y = healthy_cereals['sodium'],
    mode = 'lines+markers',
    name = 'protine'
)
trace3 = go.Scatter(
    x = healthy_cereals['name'],
    y = healthy_cereals['fiber'],
    mode = 'lines+markers',
    name = 'fiber'
)
trace4 = go.Scatter(
    x = healthy_cereals['name'],
    y = healthy_cereals['rating'],
    mode = 'lines+markers',
    name = 'rating'
)

trace5 = go.Scatter(
    x = healthy_cereals['name'],
    y = healthy_cereals['carbo'],
    mode = 'lines+markers',
    name = 'carbo',
    xaxis='x2',
    yaxis='y2'
)
trace6 = go.Scatter(
    x = healthy_cereals['name'],
    y = healthy_cereals['potass'],
    mode = 'lines+markers',
    name = 'potass',
    xaxis='x2',
    yaxis='y2'
)
trace7 = go.Scatter(
    x = healthy_cereals['name'],
    y = healthy_cereals['calories'],
    mode = 'lines+markers',
    name = 'calories',
    xaxis='x2',
    yaxis='y2'
)
trace8 = go.Scatter(
    x = healthy_cereals['name'],
    y = healthy_cereals['rating'],
    mode = 'lines+markers',
    name = 'rating',
    xaxis='x2',
    yaxis='y2'
)

layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    yaxis2=dict(
        anchor='x2'
    )
)

data = [trace0, trace1,trace2, trace3, trace4, trace5, trace6,trace7, trace8]
fig = go.Figure(data=data, layout=layout)
#py.iplot(data)
py.iplot(fig)
dropset = ['name', 'healthy']
nutrients = healthy_cereals.drop(dropset,axis=1)

# heatmeap to see the correlation between features 
mask = np.zeros_like(nutrients.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (15,15))
sns.heatmap(nutrients.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 7)
plt.show()