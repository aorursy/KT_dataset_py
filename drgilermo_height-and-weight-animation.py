import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import animation

import io

import base64

from IPython.display import HTML

plt.style.use('fivethirtyeight')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
players = pd.read_csv('../input/Players.csv')

seasons = pd.read_csv('../input/Seasons_Stats.csv')

seasons = seasons.drop(seasons.index[len(seasons) - 1])
seasons = seasons[pd.isnull(seasons.Player) == 0]

seasons['height'] = seasons.Player.apply(lambda x: players.height[players.Player == x].values[0])

seasons['weight'] = seasons.Player.apply(lambda x: players.weight[players.Player == x].values[0])
fig = plt.figure(figsize = (10,10))

ax = plt.axes()
plt.style.use('fivethirtyeight')

def animate(year):

    ax.clear()

    ax.set_xlim([60,160])

    ax.set_ylim([160,230])

    ax.set_title(str(int(year)))

    ax.set_xlabel('Weight [kg]')

    ax.set_ylabel('Height [cm]')

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'PG')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'PG')]

    ax.plot(x,y,'o', color = 'r', markersize = 10, alpha = 0.5)

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'SG')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'SG')]

    ax.plot(x,y,'o', color = 'm', markersize = 10, alpha = 0.5)

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'SF')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'SF')]

    ax.plot(x,y,'o', color = 'b', markersize = 10, alpha = 0.5)

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'PF')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'PF')]

    ax.plot(x,y,'o', color = 'g', markersize = 10, alpha = 0.5)

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'C')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'C')]

    ax.plot(x,y,'o', color = 'y', markersize = 10, alpha = 0.5)

    ax.legend(['PG','SG','SF','PF','C'], loc = 1)





ani = animation.FuncAnimation(fig,animate,seasons.Year.unique().tolist(), interval = 500)

ani.save('animation.gif', writer='imagemagick', fps=2)

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
plt.style.use('fivethirtyeight')

def animate(year):

    ax.clear()

    ax.set_xlim([60,160])

    ax.set_ylim([160,230])

    ax.set_title(str(int(year)))

    ax.set_xlabel('Weight [kg]')

    ax.set_ylabel('Height [cm]')

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'PG')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'PG')]

    ax.plot(x,y,'o', color = 'r', markersize = 10, alpha = 0.5)

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'SG')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'SG')]

    ax.plot(x,y,'o', color = 'm', markersize = 10, alpha = 0.5)

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'SF')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'SF')]

    ax.plot(x,y,'o', color = 'b', markersize = 10, alpha = 0.5)

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'PF')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'PF')]

    ax.plot(x,y,'o', color = 'g', markersize = 10, alpha = 0.5)

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'C')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'C')]

    ax.plot(x,y,'o', color = 'y', markersize = 10, alpha = 0.5)

    ax.legend(['PG','SG','SF','PF','C'], loc = 1)





ani = animation.FuncAnimation(fig,animate,np.arange(1950,1985,1), interval = 500)

ani.save('animation.gif', writer='imagemagick', fps=2)

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
plt.style.use('fivethirtyeight')

def animate(year):

    ax.clear()

    ax.set_xlim([60,160])

    ax.set_ylim([160,230])

    ax.set_title(str(int(year)))

    ax.set_xlabel('Weight [kg]')

    ax.set_ylabel('Height [cm]')

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'PG')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'PG')]

    ax.plot(x,y,'o', color = 'r', markersize = 10, alpha = 0.5)

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'SG')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'SG')]

    ax.plot(x,y,'o', color = 'm', markersize = 10, alpha = 0.5)

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'SF')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'SF')]

    ax.plot(x,y,'o', color = 'b', markersize = 10, alpha = 0.5)

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'PF')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'PF')]

    ax.plot(x,y,'o', color = 'g', markersize = 10, alpha = 0.5)

    x = seasons.weight[(seasons.Year == year) & (seasons.Pos == 'C')]

    y = seasons.height[(seasons.Year == year) & (seasons.Pos == 'C')]

    ax.plot(x,y,'o', color = 'y', markersize = 10, alpha = 0.5)

    ax.legend(['PG','SG','SF','PF','C'], loc = 1)





ani = animation.FuncAnimation(fig,animate,np.arange(1990,2018,1), interval = 500)

ani.save('animation.gif', writer='imagemagick', fps=2)

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
data_1 = seasons.height[(seasons.Year > 1950) & (seasons.Year < 1960)]

data_2 = seasons.height[(seasons.Year > 1960) & (seasons.Year < 1970)]    

data_3 = seasons.height[(seasons.Year > 1970) & (seasons.Year < 1980)]    

data_4 = seasons.height[(seasons.Year > 1980) & (seasons.Year < 1990)]    

data_5 = seasons.height[(seasons.Year > 1990) & (seasons.Year < 2000)]   

data_6 = seasons.height[(seasons.Year > 2000) & (seasons.Year < 2010)]    

data_7 = seasons.height[(seasons.Year > 2010) & (seasons.Year < 2020)]   



data = [data_1,data_2,data_3,data_4,data_5,data_6,data_7]

plt.boxplot(data)

plt.xticks([1,2,3,4,5,6,7], ['50s','60s','70s','80s','90s','2000s','2010s'])

plt.ylabel('Height [cm]')

plt.xlabel('Decade')
data_1 = seasons.weight[(seasons.Year > 1950) & (seasons.Year < 1960)]

data_2 = seasons.weight[(seasons.Year > 1960) & (seasons.Year < 1970)]    

data_3 = seasons.weight[(seasons.Year > 1970) & (seasons.Year < 1980)]    

data_4 = seasons.weight[(seasons.Year > 1980) & (seasons.Year < 1990)]    

data_5 = seasons.weight[(seasons.Year > 1990) & (seasons.Year < 2000)]   

data_6 = seasons.weight[(seasons.Year > 2000) & (seasons.Year < 2010)]    

data_7 = seasons.weight[(seasons.Year > 2010) & (seasons.Year < 2020)]   



data = [data_1,data_2,data_3,data_4,data_5,data_6,data_7]

plt.boxplot(data)

plt.xticks([1,2,3,4,5,6,7], ['50s','60s','70s','80s','90s','2000s','2010s'])

plt.ylabel('Weight [kg]')

plt.xlabel('Decade')
data_1 = seasons.Age[(seasons.Year > 1950) & (seasons.Year < 1960)]

data_2 = seasons.Age[(seasons.Year > 1960) & (seasons.Year < 1970)]    

data_3 = seasons.Age[(seasons.Year > 1970) & (seasons.Year < 1980)]    

data_4 = seasons.Age[(seasons.Year > 1980) & (seasons.Year < 1990)]    

data_5 = seasons.Age[(seasons.Year > 1990) & (seasons.Year < 2000)]   

data_6 = seasons.Age[(seasons.Year > 2000) & (seasons.Year < 2010)]    

data_7 = seasons.Age[(seasons.Year > 2010) & (seasons.Year < 2020)]   



data = [data_1,data_2,data_3,data_4,data_5,data_6,data_7]

plt.boxplot(data)

plt.xticks([1,2,3,4,5,6,7], ['50s','60s','70s','80s','90s','2000s','2010s'])

plt.ylabel('Age')

plt.xlabel('Decade')