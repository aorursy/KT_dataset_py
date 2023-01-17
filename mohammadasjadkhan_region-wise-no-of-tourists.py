import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from plotly import __version__

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

cf.go_offline()
df_region=pd.read_csv("../input/tourists-by-region/international-tourist-arrivals-by-world-region.csv")

df_region.head()
df_region["No of tourist in million"]=df_region[" (arrivals)"]/1000000

df_region.rename(columns={"Entity":"Region"},inplace=True)

df_region.drop(columns=["Code"," (arrivals)"],inplace=True)

df_region.set_index("Year",inplace=True)

df_region.tail()
colors_list = ['red','lightgreen','lightcoral','lightskyblue','yellow']

explode_list = [0.1,0.05,0,0,0.1]

df_region[df_region.index==2018]['No of tourist in million'].plot(kind='pie',

                                                                 figsize=(10,6),autopct='%1.1f%%',pctdistance=1.11,

                                                                 colors=colors_list,explode = explode_list,labels=None,

                                                                 startangle=90,shadow=True)

plt.title('Region-wise tourists distribution for Y2018 ',y=1.12)

plt.legend(df_region[df_region.index==2018].Region,loc = 'upper left')

plt.axis('equal')

plt.show()
fig,ax=plt.subplots(figsize=(4,2.5),dpi=144)

color=plt.cm.Dark2(range(5))

y=df_region[df_region.index==2018]["Region"]

width=df_region[df_region.index==2018]["No of tourist in million"]

ax.barh(y=y,width=width,color=color)

ax.set_title("Region-wise tourists distribution for Y2018")

ax.set_xlabel("tourist in million")

def nice_axes(ax): #so that we don't have set grid,facecolor etc everytime.

    ax.set_facecolor('.8')  # 0 to 1 ->black to white resp

    ax.tick_params(labelsize=8, length=0)

    ax.grid(True, axis='x', color='white')

    ax.set_axisbelow(True)                  # make it false and see change 

    [spine.set_visible(False) for spine in ax.spines.values()]  # make it true and see change

    

nice_axes(ax)

fig



years=list(df_region.index.unique())
from IPython.display import HTML

from matplotlib.animation import FuncAnimation

colors=plt.cm.Dark2(range(5))

def bar_chart(year):

    ax.clear()

    y=df_region[df_region.index==year]["Region"]

    width=df_region[df_region.index==year]["No of tourist in million"]

    ax.barh(y=y,width=width,color=color)



    ax.set_title(year)

    ax.set_xlabel("tourist in million")

    nice_axes(ax)

fig, ax = plt.subplots(figsize=(10,6))

animator = FuncAnimation(fig, bar_chart, frames=years)

HTML(animator.to_jshtml())

    