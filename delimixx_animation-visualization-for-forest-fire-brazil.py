import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gp

import io

import base64

from IPython.display import HTML

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.animation as animation

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns

import subprocess

import warnings

import imageio

warnings.filterwarnings(action='ignore')

plt.style.use('fivethirtyeight')
data = pd.read_csv("../input/forest-fires-in-brazil/amazon.csv",encoding="ISO-8859-1")
data.head()
print("year : ", data['year'].unique())
print("state : ", data['state'].unique())
print("month : ", data['month'].unique())
month_map={'Janeiro': 'January', 'Fevereiro': 'February', 'Março': 'March', 'Abril': 'April', 'Maio': 'May',

          'Junho': 'June', 'Julho': 'July', 'Agosto': 'August', 'Setembro': 'September', 'Outubro': 'October',

          'Novembro': 'November', 'Dezembro': 'December'}

#mapping our translated months

data['month']=data['month'].map(month_map)

#checking the month column for the second time after the changes were made

data.month.unique()
# data

plt.figure(figsize = (15,10))

data[['state','number']].groupby(['state']).number.sum().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',15))

plt.axhspan(21.5,22.5 ,facecolor='Blue', alpha=0.4) # hilight space

plt.title('Number of Fire')

plt.show()
#plot

f, ax = plt.subplots(1,2, figsize = (25,15))

data[data['state'] == 'Mato Grosso'][['month','number']].groupby(['month']).number.sum().plot.pie(autopct='%1.1f%%',explode=[0,0,0,0,0.2,0,0,0,0.1,0,0,0], shadow=True,ax=ax[0])

ax[0].set_title('Mato Cross Month Fire', size = 25)

data[['month','number']].groupby(['month']).number.sum().plot.pie(autopct='%1.1f%%',explode=[0,0,0,0,0,0.2,0,0,0,0,0.2,0], shadow=True,ax=ax[1])

ax[1].set_title('ALL Month Fire', size = 25)

plt.show()
ani_d1 = data[['year','number']].groupby(['year']).number.sum().reset_index()
def animate(i):

    data = ani_d1.iloc[:int(i+1)]

    p = sns.barplot(x=data['year'], y=data['number'], data=data, palette="rocket")

    p.tick_params(labelsize=17)

    plt.setp(p.lines,linewidth=4)

    

fig = plt.figure(figsize=(10,6))

plt.xlim()

plt.ylim(np.min(ani_d1)[1], np.max(ani_d1)[1] + 10000)

plt.xlabel('Year',fontsize=20)

plt.title('Number of Fire Per Year',fontsize=20)



ani = matplotlib.animation.FuncAnimation(fig, animate, frames=20, interval=500, repeat=True)

ani.save('plot.gif', writer='imagemagick')

plt.close()



video = io.open('plot.gif', 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
geo_data = gp.read_file('../input/geo-brazil-data/geo_brazil_data.shp')

geo_data = geo_data.rename(columns = {'nome': 'state'})
ani_d2 = data[['year','state','number']].groupby(['year','state']).number.sum().reset_index()

ani_d2_pivot = ani_d2.pivot_table(values='number', index=['state'], columns='year').reset_index()

ani_d2_final = geo_data.set_index('state').join(ani_d2_pivot.set_index('state'))

# ani_d2_pivot = ani_d2.pivot_table(values='number', index=['state'], columns='year')

# ani_d2_final = pd.merge(ani_d2_pivot, geo_data, on='state', how='inner')

# ani_d2_final.head()
filename = []

vmin, vmax = 150, 300



for year in ani_d2_final.columns[1:]:

    

    # create map

    fig = ani_d2_final.plot(column=year, cmap='Oranges', figsize=(8,5), linewidth=0.8, edgecolor='0.8', vmin=vmin, vmax=vmax, legend=True, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    fig.axis('off')

    fig.set_title('Fire In Brazil', fontdict={'fontsize': '15', 'fontweight' : '3'})

        

    fig.annotate(year, xy=(0.1, .225), xycoords='figure fraction', horizontalalignment='left', verticalalignment='top',fontsize=20)

    chart = fig.get_figure()

    chart.savefig(str(year) +'_fire.png', dpi=300)

    filename.append(str(year) +'_fire.png')

    plt.close()
images = []

for filename in filename:

    images.append(imageio.imread(filename))

imageio.mimsave('fire.gif', images)
video = io.open('fire.gif', 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))