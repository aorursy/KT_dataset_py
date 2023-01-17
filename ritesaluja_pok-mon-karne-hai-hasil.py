# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import seaborn as sns
import matplotlib.pyplot as plt

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True) 

from PIL import Image #for image mask

#for word cloud
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

from termcolor import colored
# Any results you write to the current directory are saved as output.
pk = pd.read_csv("../input/pokemon/Pokemon.csv")
#pk.head(10)
plt.figure(figsize=(18,9))
sns.countplot(pk['Type 1'],palette=['green','#FFA500','blue','red','#cc7700','purple','#0892d0','#654321','yellow',\
                                   '#830303','pink','grey','#444444','#fcfcfb','#FF8C00','black','#4682b4','skyblue'])
plt.xlabel('Types')
plt.title('Pokemon types')
plt.show()
#how does different attributes of a Pokemon correlate with each other 
# Compute the correlation matrix
corr = pk.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(120, 100, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
#Function to compare the Pokémon powers
def PokeFight(trace1,trace2,trace3,trace4):
    x = pk[pk["Name"] == trace1]
    trace1 = go.Scatterpolar(
      r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],
      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
      fill = 'toself',
      name = trace1
    )
    x = pk[pk["Name"] == trace2]
    trace2 = go.Scatterpolar(
      r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],
      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
      fill = 'toself',
      name = trace2
    )
    x = pk[pk["Name"] == trace3]
    trace3 = go.Scatterpolar(
      r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],
      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
      fill = 'toself',
      name = trace3
    )
    x = pk[pk["Name"] == trace4]
    trace4 = go.Scatterpolar(
      r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],
      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
      fill = 'toself',
      name = trace4
    )

    layout = go.Layout(
      xaxis=dict(
            domain=[0, 0.45]
        ),
        yaxis=dict(
            domain=[0, 0.45]
        ),
        xaxis2=dict(
            domain=[0.55, 1]
        ),
        xaxis3=dict(
            domain=[0, 0.45],
            anchor='y3'
        ),
        xaxis4=dict(
            domain=[0.55, 1],
            anchor='y4'
        ),
        yaxis2=dict(
            domain=[0, 0.45],
            anchor='x2'
        ),
        yaxis3=dict(
            domain=[0.55, 1]
        ),
        yaxis4=dict(
            domain=[0.55, 1],
            anchor='x4'
        ),
      showlegend = True,
      title = "Pokémons' Performance (RAW power)"
    )

    data = [trace1, trace2, trace3, trace4]
    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename = "Pokemon stats")
PokeFight("Squirtle","Pikachu","Bulbasaur","Charmander")
#Data Prep:- summary stats for each type
summarypk = pk.groupby('Type 1').sum()
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = fig.gca(projection='3d')
fig.subplots_adjust(left=0.4, right=2,bottom=2,top=4)
    
x=summarypk.Attack
y=summarypk.Defense
z=summarypk.Speed
Desc = np.array(summarypk.index)
    
a = 'right'
for i in range(18):
    ax.scatter(x[i], y[i], z[i], c='orange', marker='*',s=100)
    ax.text(x[i], y[i], z[i], '%s'%(Desc[i]), color='r',alpha=0.8, fontsize=8,horizontalalignment=a,verticalalignment='bottom', \
           bbox=dict(facecolor='red', alpha=0.12)) 
    if a=='right':
        a = 'left'
    else:
        a = 'right'
        
plt.title("Pokemon Attack, Defense and Speed by Types")
 
ax.set_xlabel('Attack')
ax.set_ylabel('Defense')
ax.set_zlabel('Speed');
ax.w_xaxis.set_pane_color((0.0, 0.99, 1.0,  1.0))
ax.w_yaxis.set_pane_color((0.0, 0.99, 1.0, 1.0))
ax.w_zaxis.set_pane_color((0.1, 1.0, 0.2, 1.0))
    
plt.show()  
sns.jointplot(x='Attack', y='Speed', 
              data=pk, color ='#FF8C00', kind ='reg', 
              size = 8.0)
plt.show()
sns.jointplot(x='Defense', y='Speed', 
              data=pk, color ='grey', kind ='reg', 
              size = 8.0)
plt.show()
sns.jointplot(x='Attack', y='Sp. Atk', 
              data=pk, color ='blue', kind ='reg', 
              size = 8.0)
plt.ylabel("Special Attack",color='red')
plt.xlabel("Attack",color='red')
plt.show()
#Some Popular Type 2 Pokémons'
maskComfy = np.array(Image.open( "../input/beerimage/images.jpg"))

wordcloud = (WordCloud(width=1440, height=1080, relative_scaling=0.5, mask=maskComfy,max_words=1000,background_color='white').generate_from_frequencies(pk["Type 2"].value_counts()))

fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud,interpolation="gaussian")
plt.axis('off')
plt.show()
x = pk[pk["Legendary"]].sort_values(by=['HP'],ascending=False)["Name"].head(1).values
print(colored(x[0],'green'))
#DataPrep
pkattack = pk.groupby("Generation")["Attack"].sum()
pkdefend = pk.groupby("Generation")["Defense"].sum()

#Plotting
fig = plt.figure(figsize=(15,9))
plt.bar(np.array(pkattack.index),np.array(pkattack.values),color='#FF8C00',edgecolor=['black']*6,width = 0.25,align='center',label='Aggresive')
plt.bar(np.array(pkdefend.index)+0.25,np.array(pkdefend.values),color='grey',edgecolor=['grey']*6,width = 0.25,align='center',label='Defensive')
plt.xlabel("Generations",color='blue')
plt.ylabel("Aggresiveness/Defensiveness",color='blue') #measured by Attack points
plt.legend(loc = 'upper right',)
plt.grid(color='g', linestyle='--', linewidth=0.1)
plt.show()
#Data Prep for legendry Pokemon
pklegendary = pk[pk["Legendary"]]
pklegendary = pklegendary[["HP","Attack","Defense","Sp. Atk","Sp. Def"]]

#plotting
fig = plt.figure(figsize= (12,9))
sns.violinplot(data=pklegendary)
plt.ylabel("Power Points",color="grey")
sns.swarmplot(data=pklegendary, color="black", edgecolor="gray")
plt.show()
#DataPrep
datatemp = pk[["Type 1","HP","Attack","Defense","Sp. Atk","Sp. Def","Legendary","Generation"]] #can be used later

fig = plt.figure(figsize= (15,9))
#swarm plot
sns.set(style="darkgrid")
g = sns.swarmplot(x="Type 1",y="Sp. Atk",data=datatemp,hue="Generation", edgecolor=['black']*6)
plt.ylabel("Special Attack")
plt.title("Special Attack of Different Types by Gen")
plt.show()
# Stacked Bar Chart of Pokemon Types and count differentiating based on Legend
fig = plt.figure(figsize= (15,9))
barWidth = 1
r = [i for i in range(18)]
bars1 = pk[pk["Legendary"]].groupby("Type 1")["Type 1"].count().values #initial counts it doesn't take into account pokemons that are not legendary
bars1  = [0, 2, 12,  4,  1, 0, 5,  2,  2,  3,  4,  2,  2,0, 14,  4,  4,  4] #changing Non Legendary types with zero
bars2 = pk[pk["Legendary"]==False].groupby("Type 1")["Type 1"].count().values
names = pk.groupby("Type 1")["Type 1"].count().index
plt.bar(r,bars2 , color='green', edgecolor=['black']*18, width=barWidth,label='Not Legendary') #first Non Legendary Pokemons bar
plt.bar(r, bars1, bottom=bars2, color='blue', edgecolor=['black']*18, width=barWidth,label='Legendary',hatch='..') #Legendary Pokemons' stacked bar
plt.xticks(r, names, fontweight='bold')
plt.ylabel("Counts",color='grey')
plt.xlabel("Primary Types",color='grey')
plt.legend()
plt.show()
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(pk["Type 1"],pk["Type 2"] , rownames=['Primary Type'], colnames=['Secondary Type']).style.background_gradient(cmap=cm)

#function to plot line for each capability of pokémon
def plotline(i):
    a= pk.groupby("Generation")[i].mean().index
    b = pk.groupby("Generation")[i].mean().values
    plt.plot(a,b,ls='dashed')
#plotting
fig = plt.figure(figsize=(12,9))
for i in pk.columns[5:11]:   #go through each required column
    plotline(i)
plt.legend(pk.columns[5:11], loc='upper left')
plt.ylabel("Avg. Points", color = 'grey')
plt.xlabel("Generations", color = 'grey')
plt.title("Pokémon Capabilities by Generations",color='blue')
plt.show()

#Functions for Pokémon Tracker
def pokeatri(trace1):
    cm = sns.light_palette("orange", as_cmap=True)
    pkt = pk.fillna('Unknown')
    display(pkt[pkt["Name"] == trace1].iloc[:,2:-1])


def PokeTracker(trace1):
    pokeatri(trace1)
    x = pk[pk["Name"] == trace1]
    trace1 = go.Scatterpolar(
      r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],
      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
      fill = 'toself',
      name = trace1
    )
    layout = go.Layout(
          xaxis=dict(
            domain=[0, 0.45]
            ),
            yaxis=dict(
            domain=[0, 0.45]
            ),
        
           
          showlegend = True,
          title = "Pokémons' Performance (RAW power)"
    )

    data = [trace1]
    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename = "Pokemon stats")

PokeTracker("Pikachu")
cm = sns.light_palette("orange", as_cmap=True)
pk[['Name','Type 1','Type 2','HP','Total','Legendary']].sort_values(by=['Total'],ascending = False).head(20).sort_values(by=['Legendary','Type 1','Type 2'],ascending = False).reset_index(drop=True).style.background_gradient(cmap=cm)

