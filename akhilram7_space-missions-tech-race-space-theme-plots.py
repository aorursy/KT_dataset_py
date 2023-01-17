# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Importing Required Libraries

from pandas.api.types import CategoricalDtype
from plotnine import *
from plotnine.data import mpg
from plotnine import options as op
from plotnine import watermark as wm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

import warnings
import networkx as nx
warnings.filterwarnings('ignore')
space = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')
space['Country'] = np.NaN

for each in range(len(space['Location'])):
    space['Country'][each]  = space['Location'][each].split(',')[-1].strip()
rocket_pd = pd.read_csv('../input/allrocketsfrom1957/all-rockets-from-1957.csv')
#top 20 highest payload missions to GTO

leo = rocket_pd[['Name','Cmp','Payload to GTO']].sort_values(by=['Payload to GTO'], ascending=False).head(20)
op.figure_size=(8,8)
ax = (ggplot(leo)         
 + aes(x='Name', y='Payload to GTO', fill='Cmp')    
 + geom_col(size=20)
 + coord_flip()
 + scale_fill_manual(values = ["#B21236","#F03812","#FE8826","#FEB914","#2C9FA2","#002C2B","#F7E1C0"])
 + labs(title = "Top 20 highest Payload carrying Vehicles to GTO")
 + labs(y = "Payload in Tonnes", x = "Vehicle Name")
 + theme(
    panel_background = element_rect(fill = "black"),    
    plot_background = element_rect(fill = "black", color = "black"),
    legend_background = element_rect(fill = "black"),
    legend_key = element_blank(),
   
    panel_grid = element_line(size = 0.3),
    panel_grid_minor_y = element_blank(),
    panel_grid_major_y = element_blank(),
    
    legend_text = element_text(color = "white"),
    axis_text_x = element_text(color = "white", size = 10),
    axis_text_y = element_text(color = "white", size = 10, hjust = 1, margin={'b': 20, 't':10}),
    axis_title = element_text(color = "white", size = 14, hjust = 1),
    plot_title = element_text(color = "white", face = "bold", size = 16, hjust = 4, margin={'b': 20, 't':10}),
    panel_spacing_x = 1
  )
)

fig = ax.draw()

image = plt.imread('https://raw.githubusercontent.com/AkhilRam7/Covid19Tweets/master/space%20images/gto.png')

fig.figimage(image, xo=0, yo=0, alpha=0.2, norm=None, cmap=None, vmin=None, vmax=None, origin=None, resize=True)

fig.show()
#top 20 highest payload missions to LEO orbit

leo = rocket_pd[['Name','Cmp','Payload to LEO']].sort_values(by=['Payload to LEO'], ascending=False).head(20)
op.figure_size=(8,8)
ax = (ggplot(leo)         
 + aes(x='Name', y='Payload to LEO', fill='Cmp')    
 + geom_col(size=20)
 + coord_flip()
 + scale_fill_brewer(type='qual' ,palette = 2)
 + labs(title = "Top 20 highest Payload carrying Vehicles to LEO")
 + labs(y = "Payload in Tonnes", x = "Vehicle Name")
 + theme(
    panel_background = element_rect(fill = "black"),    
    plot_background = element_rect(fill = "black", color = "black"),
    legend_background = element_rect(fill = "black"),
    legend_key = element_blank(),
   
    panel_grid = element_line(size = 0.3),
    panel_grid_minor_y = element_blank(),
    panel_grid_major_y = element_blank(),
    
    legend_text = element_text(color = "white"),
    axis_text_x = element_text(color = "white", size = 10),
    axis_text_y = element_text(color = "white", size = 10, hjust = 1, margin={'b': 20, 't':10}),
    axis_title = element_text(color = "white", size = 14, hjust = 1),
    plot_title = element_text(color = "white", face = "bold", size = 16, hjust = 4, margin={'b': 20, 't':10}),
    panel_spacing_x = 1
  )
)

fig = ax.draw()

image = plt.imread('https://raw.githubusercontent.com/AkhilRam7/Covid19Tweets/master/space%20images/leo.png')

fig.figimage(image, xo=0, yo=0, alpha=0.3, norm=None, cmap=None, vmin=None, vmax=None, origin=None, resize=True)

fig.show()

fair = rocket_pd[['Name','Cmp','Fairing Diameter','Fairing Height']].dropna()

fair['Fairing Diameter'] = fair['Fairing Diameter'].str.split('m').str.get(0).str.strip().astype(float)
fair['Fairing Height'] = fair['Fairing Height'].str.split('m').str.get(0).str.strip().astype(float)

scat = fair.dropna()['Cmp'].value_counts().rename_axis('Cmp').reset_index(name='Counts')
scat = scat[(scat['Counts'] >= 5)]
companies = scat['Cmp'].tolist()
scat = fair[(fair['Cmp'].isin(companies))].dropna()


op.figure_size=(8, 8)
p = (ggplot(scat)         # defining what data to use
 + aes(y='Fairing Height', x='Fairing Diameter', fill='Cmp')    # defining what variable to use
 + geom_jitter(height=7, width=7, size=7, shape='d') # defining the type of plot to use
 + scale_fill_brewer(type='qual' ,palette = 3)
 + labs(title = "Fairing Height and Fairing Diameter")
 + labs(y = "Fairing Height", x = "Fairing Diameter")
 + theme(
    panel_background = element_rect(fill = "black"),    
    plot_background = element_rect(fill = "black", color = "black"),
    legend_background = element_rect(fill = "black"),
    legend_key = element_blank(),
   
    panel_grid = element_line(size = 0.3),
    panel_grid_minor_y = element_blank(),
    panel_grid_major_y = element_blank(),
    
    legend_text = element_text(color = "white"),
    axis_text_x = element_text(color = "white", size = 10),
    axis_text_y = element_text(color = "white", size = 10),
    axis_title = element_text(color = "white", size = 14, hjust = 1),
    plot_title = element_text(color = "white", face = "bold", size = 16, hjust = 4, margin={'b': 20, 't':10}),
    panel_spacing_x = 1
  )
)

fig = p.draw()

image = plt.imread('https://raw.githubusercontent.com/AkhilRam7/Covid19Tweets/master/space%20images/fairing.png')

fig.figimage(image, xo=0, yo=0, alpha=0.25, norm=None, cmap=None, vmin=None, vmax=None, origin=None, resize=True)

fig.show()
nasa = rocket_pd[['Name','Cmp']]
nasa = nasa[nasa['Cmp'] == 'NASA']

nasa = nx.from_pandas_edgelist(nasa, source='Cmp', target='Name', edge_attr=True,)

plt.rcParams['figure.figsize']  = (12, 12)
plt.style.use('dark_background')
plt.title('NASA Launch Vehicles', fontsize = 30)

layout = nx.spring_layout(nasa)
nx.draw_networkx_nodes(nasa, layout, node_size = 2000, node_color = '#0B3D91')
nx.draw_networkx_edges(nasa, layout, width = 6, alpha = 1, edge_color = '#FC3D21')
nx.draw_networkx_labels(nasa, layout, font_size = 10, font_color='white', font_weight='bold')

plt.grid()
plt.axis('off')

image = plt.imread('https://raw.githubusercontent.com/AkhilRam7/Covid19Tweets/master/space%20images/nas.png')

plt.figimage(image, xo=570, yo=580, alpha=1, norm=None, cmap=None, vmin=None, vmax=None, origin=None, resize=False)

plt.show()

spacex = rocket_pd[['Name','Cmp']]
spacex = spacex[spacex['Cmp'] == 'SpaceX']

spacex = nx.from_pandas_edgelist(spacex, source='Cmp', target='Name', edge_attr=True,)

plt.rcParams['figure.figsize']  = (12, 12)
plt.style.use('dark_background')
plt.title('SpaceX Launch Vehicles', fontsize = 30)

layout = nx.spring_layout(spacex)
nx.draw_networkx_nodes(spacex, layout, node_size = 2000, node_color = '#72A6B5')
nx.draw_networkx_edges(spacex, layout, width = 6, alpha = 1, edge_color = '#E4D493')
nx.draw_networkx_labels(spacex, layout, font_size = 10, font_color='white', font_weight='bold')

plt.grid()
plt.axis('off')

image = plt.imread('https://raw.githubusercontent.com/AkhilRam7/Covid19Tweets/master/space%20images/elon.png')

plt.figimage(image, xo=350, yo=400, alpha=1, norm=None, cmap=None, vmin=None, vmax=None, origin=None, resize=False)

plt.show()

nasa = rocket_pd[['Name','Cmp']]
nasa = nasa[nasa['Cmp'] == 'ISRO']

nasa = nx.from_pandas_edgelist(nasa, source='Cmp', target='Name', edge_attr=True,)

plt.rcParams['figure.figsize']  = (12, 12)
plt.style.use('dark_background')
plt.title('ISRO Launch Vehicles', fontsize = 30)

layout = nx.spring_layout(nasa)
nx.draw_networkx_nodes(nasa, layout, node_size = 2000, node_color = '#F47216')
nx.draw_networkx_edges(nasa, layout, width = 6, alpha = 1, edge_color = '#0E88D3')
nx.draw_networkx_labels(nasa, layout, font_size = 10, font_color='white', font_weight='bold')

plt.grid()
plt.axis('off')

image = plt.imread('https://raw.githubusercontent.com/AkhilRam7/Covid19Tweets/master/space%20images/isro.png')

plt.figimage(image, xo=580, yo=580, alpha=1, norm=None, cmap=None, vmin=None, vmax=None, origin=None, resize=False)

plt.show()

#top 20 company with most launches 

all_launches = space['Company Name'].value_counts().rename_axis('Company Name').reset_index(name='Launches')
all_launches['Country'] = np.NaN
cmp_cnt = {each[0]:each[1] for each in space[['Company Name', 'Country']].values.tolist()} 
for launch in range(len(all_launches)):
    if all_launches['Company Name'][launch] in cmp_cnt.keys():
        all_launches['Country'][launch] = cmp_cnt[all_launches['Company Name'][launch]]

op.figure_size=(8,8)
ax = (ggplot(all_launches.head(20))         
 + aes(x='Company Name', y='Launches', fill='Country')    
 + geom_col()
 + scale_fill_manual(values = ["#B21236","#F03812","#FE8826","#FEB914","#2C9FA2","#002C2B","#F7E1C0","#947EA9","#FE8826","#FEB914","#2C9FA2","#002C2B","#F7E1C0"])
 + labs(title = "Top 20 Space companies with most launches")
 + labs(y = "Launches since 1957", x = "Company Name")
 + theme(
    panel_background = element_rect(fill = "black"),    
    plot_background = element_rect(fill = "black", color = "black"),
    legend_background = element_rect(fill = "black"),
    legend_key = element_blank(),
   
    panel_grid = element_line(size = 0.1),
    panel_grid_minor_y = element_blank(),
    panel_grid_major_y = element_blank(),
    
    legend_text = element_text(color = "white"),
    axis_text_x = element_text(color = "white", size = 10, angle=90),
    axis_text_y = element_text(color = "white", size = 10, hjust = 8, margin={'b': 20, 't':10}),
    axis_title = element_text(color = "white", size = 14, hjust = 4),
    plot_title = element_text(color = "white", face = "bold", size = 16, hjust = 4, margin={'b': 20, 't':10}),
    panel_spacing_x = 1
  )
)

fig = ax.draw()

fig.show()

#company launchs since 1957 

timeseries = space.groupby(['Company Name', 'Datum'])['Status Mission'].count().reset_index()
timeseries['Datum'] = timeseries['Datum'].str.split(' ').str.get(3).str.strip().astype(int)
tcmp = timeseries['Company Name'].value_counts().head(5).rename_axis('Company Name').reset_index(name='Missions')['Company Name'].to_list()

timeseries = timeseries[timeseries['Company Name'].isin(tcmp)]

timeseries = timeseries.groupby(['Company Name','Datum'])['Status Mission'].sum().reset_index()

op.figure_size=(8,8)


ax = (ggplot(timeseries)         
 + aes(x='Datum', y='Status Mission', color='Company Name')    
 + geom_line(size=1.5)
 + scale_color_manual(values = ["#5BC0EB","#FDE74C","#9BC53D","#E55934","#E1DAD0"])
 + labs(title = "Top Companies Launch Timelines")
 + labs(x = "Year", y = "Launches")
 + theme(
    panel_background = element_rect(fill = "black"),    
    plot_background = element_rect(fill = "black", color = "black"),
    legend_background = element_rect(fill = "black"),
    legend_key = element_blank(),
   
    panel_grid = element_line(size = 0.1),
    panel_grid_minor_y = element_blank(),
    panel_grid_major_y = element_blank(),
    
    legend_text = element_text(color = "white"),
    axis_text_x = element_text(color = "white", size = 10),
    axis_text_y = element_text(color = "white", size = 10, hjust = 8, margin={'b': 20, 't':10}),
    axis_title = element_text(color = "white", size = 14, hjust = 4),
    plot_title = element_text(color = "white", face = "bold", size = 16, hjust = 4, margin={'b': 20, 't':10}),
    panel_spacing_x = 1
  )
      
)



fig = ax.draw()

fig.show()

#company Misson Status
all_cmp = space[['Company Name', 'Status Mission']]
tcmp = space['Company Name'].value_counts().head(10).rename_axis('Company Name').reset_index(name='Missions')['Company Name'].to_list()
cmp = all_cmp[all_cmp['Company Name'].isin(tcmp)]

op.figure_size=(8,8)

ax = (ggplot(cmp)         
 + aes(x='Company Name', fill='Status Mission')    
 + geom_bar(size=1,position = "stack")
 + scale_fill_manual(values = ["#FC7F00","#F7993F","#B4E0DF","#79DAEB"])
 + labs(title = "Top Companies Launch Status")
 + labs(x = "Year", y = "Companies")
 + theme(
    panel_background = element_rect(fill = "black"),    
    plot_background = element_rect(fill = "black", color = "black"),
    legend_background = element_rect(fill = "black"),
    legend_key = element_blank(),
   
    panel_grid = element_line(size = 0.1),
    panel_grid_minor_y = element_blank(),
    panel_grid_major_y = element_blank(),
    
    legend_text = element_text(color = "white"),
    axis_text_x = element_text(color = "white", size = 10, angle=90),
    axis_text_y = element_text(color = "white", size = 10, hjust = 8, margin={'b': 20, 't':10}),
    axis_title = element_text(color = "white", size = 14, hjust = 4),
    plot_title = element_text(color = "white", face = "bold", size = 16, hjust = 4, margin={'b': 20, 't':10}),
    panel_spacing_x = 1
  )
      
)

fig = ax.draw()

fig.show()

#Highest cost per launches
cost = space[['Company Name', ' Rocket']].dropna()
cost[' Rocket'] = cost[' Rocket'].str.replace(',','').astype(float)
cost = cost.groupby(['Company Name'])[' Rocket'].sum().reset_index().sort_values(by = ' Rocket', ascending=False).reset_index()[['Company Name',' Rocket']]

op.figure_size=(8, 8)
ax = (ggplot(cost.head(10))         
 + aes( x='Company Name', y=' Rocket', fill='Company Name')    
 + geom_col()
 + scale_fill_brewer(type='qual' ,palette = 3)
 + labs(title = "Most Spent Companies for all Launches")
 + labs(x = "Company", y = "Spent in Millions")
 + theme(
    panel_background = element_rect(fill = "black"),    
    plot_background = element_rect(fill = "black", color = "black"),
    legend_background = element_rect(fill = "black"),
    legend_key = element_blank(),
   
    panel_grid = element_line(size = 0.1),
    panel_grid_minor_y = element_blank(),
    panel_grid_major_y = element_blank(),
    
    legend_text = element_text(color = "white"),
    axis_text_x = element_text(color = "white", size = 10, angle=90),
    axis_text_y = element_text(color = "white", size = 10, hjust = 1, margin={'b': 20, 't':10}),
    axis_title = element_text(color = "white", size = 14, hjust = 1),
    plot_title = element_text(color = "white", face = "bold", size = 16, hjust = 4, margin={'b': 20, 't':10}),
    panel_spacing_x = 1
  )
      
)

fig = ax.draw()
fig.show()
cost = space[['Company Name', ' Rocket']].dropna()
cost[' Rocket'] = cost[' Rocket'].str.replace(',','').astype(float)
cost = cost.sort_values(by = ' Rocket', ascending=False).reset_index()[['Company Name',' Rocket']]

# TO reduce outliners
cost = cost[cost[' Rocket'] < 500]
tcmp = cost['Company Name'].value_counts().head(10).rename_axis('Company Name').reset_index(name='Missions')['Company Name'].to_list()
cost = cost[cost['Company Name'].isin(tcmp)]

op.figure_size=(8, 8)

ax = (ggplot(cost)         
 + aes(x='Company Name', y=' Rocket', fill='Company Name')    
 + geom_boxplot(outlier_colour  = 'white')
 + scale_fill_brewer(type='qual' ,palette = 3)
 + labs(title = "Most Launched Companies vs Distribution of Cost per Launch")
 + labs(x = "Companies", y = "Cost in Million")
 + theme(
    panel_background = element_rect(fill = "black"),    
    plot_background = element_rect(fill = "black", color = "black"),
    legend_background = element_rect(fill = "black"),
    legend_key = element_blank(),
   
    panel_grid = element_line(size = 0.1),
    panel_grid_minor_y = element_blank(),
    panel_grid_major_y = element_blank(),
    
    legend_text = element_text(color = "white"),
    axis_text_x = element_text(color = "white", size = 10, angle=90),
    axis_text_y = element_text(color = "white", size = 10, hjust = 1, margin={'b': 20, 't':10}),
    axis_title = element_text(color = "white", size = 14, hjust = 'center'),
    plot_title = element_text(color = "white", face = "bold", size = 16, hjust = 4, margin={'b': 20, 't':10}),
    panel_spacing_x = 1
  )
      
)
fig = ax.draw()
fig.show()

mission = space[['Company Name','Detail']]
mission['Detail'] = mission['Detail'].str.split('|').str.get(1).str.split('&').str.len().astype(int)
mission = mission.groupby(['Company Name'])['Detail'].sum().reset_index().sort_values(by = 'Detail', ascending=False).reset_index()

op.figure_size=(8, 8)
ax = (ggplot(mission.head(15))         
 + aes( x='Company Name', y='Detail', fill='Company Name')    
 + geom_col()
 + scale_fill_brewer(type='qual' ,palette = 3)
 + labs(title = "Top Companies based on overall space missions")
 + labs(x = "Company Name", y = "Space Missions")
 + theme(
    panel_background = element_rect(fill = "black"),    
    plot_background = element_rect(fill = "black", color = "black"),
    legend_background = element_rect(fill = "black"),
    legend_key = element_blank(),
   
    panel_grid = element_line(size = 0.1),
    panel_grid_minor_y = element_blank(),
    panel_grid_major_y = element_blank(),
    
    legend_text = element_text(color = "white"),
    axis_text_x = element_text(color = "white", size = 10, angle=90),
    axis_text_y = element_text(color = "white", size = 10, hjust = 1, margin={'b': 20, 't':10}),
    axis_title = element_text(color = "white", size = 14, hjust = 1),
    plot_title = element_text(color = "white", face = "bold", size = 16, hjust = 4, margin={'b': 20, 't':10}),
    panel_spacing_x = 1
  )
      
)

fig = ax.draw()
fig.show()