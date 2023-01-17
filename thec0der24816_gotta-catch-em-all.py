
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
pk = pd.read_csv("../input/Pokemon.csv")
pk.head()
pk.info()
pk.describe()
pk.corr()
pk["Type 1"].unique()
pk["Type 2"].unique()
# pkp = sns.countplot()
# plt.show(pkp)
plt.figure(figsize=(18,9))
sns.countplot(pk['Type 1'])
plt.xlabel('Types')
plt.title('Pokemon types')
plt.figure(figsize=(18,9))
sns.countplot(pk['Type 2'])
plt.xlabel('Types')
plt.title('Pokemon types')
labels = 'Water', 'Normal', 'Grass', 'Bug', 'Psychic', 'Fire', 'Electric', 'Rock', 'Other'
sizes = [112, 98, 70, 69, 57, 52, 44, 44, 175]
colors = ['C', 'grey', 'g', 'burlywood', 'b', 'r', 'Y', 'k', 'M']
plt.pie(sizes, labels=labels, colors=colors, startangle=90)
plt.axis('equal')
plt.title("Percentage of Different Types of Pokemon")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(7,7)
plt.show()
plt.figure(figsize=(15,5))
sns.stripplot(x="Type 1", y="HP", data=pk, jitter=True, linewidth= 1)
plt.xlabel('Types')
plt.title('Pokemon types')


LP = pk[pk['Legendary']==True]
print(LP)
pk['Type 2'].fillna(pk['Type 1'], inplace=True) 
print(pk[pk['Type 2'] == np.nan])
# As we can see the dataset has no NA values

sns.jointplot(x='Attack', y='Speed', 
              data=pk, color ='red', kind ='reg', 
              size = 8.0)
plt.show()
sns.jointplot(x='Attack', y='Defense', 
              data=pk, color ='c', kind ='reg', 
              size = 8.0)
plt.show()
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True) 

x = pk[pk["Name"] == "Alakazam"]
data = [go.Scatterpolar(
  r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],
  theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
  fill = 'toself',
)]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 250]
    )
  ),
  showlegend = False,
  title = "Performance of {} in terms of RAW power".format(x.Name.values[0])
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename = "Single Pokemon stats")

iplot([go.Histogram2dContour(x=pk.head(100)['Sp. Atk'], 
                             y=pk.head(100)['Sp. Def'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=pk.head(100)['Sp. Atk'], y=pk.head(100)['Sp. Def'], mode='markers')])





