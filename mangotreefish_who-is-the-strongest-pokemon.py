

import numpy as np 

import pandas as pd 

import seaborn as sns 

from matplotlib import pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

pokemon = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv') #First we read our csv
pokemon.head() #Check the data
pokemon.dtypes #Data types look good
pokemon.isnull().sum() #Check for nulls
pokemon.fillna(value = 'None', inplace=True) #Fill them in
most_powerful = pokemon[pokemon['Legendary']!= True].sort_values('Total' ,ascending = False) #Sort the data and exclude legendaries
most_powerful.nlargest(40,'Total')
pokemon_final = df_without_mega = most_powerful[~most_powerful.Name.str.contains("Mega")] #Dropping rows with "Mega"
pokemon_final[pokemon_final.Name.str.contains("Mega")].sum() #Check to make sure it worked
top_10 = pokemon_final.nlargest(10,'Total') #Define the data set
import seaborn as sns #Happy plotting!

from matplotlib import pyplot as plt



plt.figure(figsize=(20,8))



sns.set(style='whitegrid')



sns.barplot(x="Name", y="Total", data=top_10)
stats_df = pokemon_final.groupby('Type 1').mean() #Create the data set to put into the graphs
stats_df
plt.figure(figsize= (17,8))



sns.barplot(x = 'Total', y='Type 1', data=stats_df.reset_index(),palette='dark')

stats = stats_df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']] #Get stats without 'Total' column

list_types = stats.index.unique().tolist()#Get types into a list

i= 1 #Set variable to distribute subplots

c=0 #set variable to distribute palettes

palette=['BuGn_r','OrRd','copper','YlOrRd','Blues_d','winter']

plt.figure(figsize=(17,17))





for stat in stats: #function to make a chart for each stat

    plt.subplot(3,2,i)

    i=i+1

    sns.barplot(x =stats[stat], y=list_types,palette = palette[c])

    c=c+1

    

    plt.title(str('Mean of ' + stat))
TYPE_LIST = ['Grass','Fire','Water','Bug','Normal','Poison',

            'Electric','Ground','Fairy','Fighting','Psychic',

            'Rock','Ghost','Ice','Dragon','Dark','Steel','Flying']



COLOR_LIST = ['#8ED752', '#F95643', '#53AFFE', '#C3D221', '#BBBDAF', '#AD5CA2', 

              '#F8E64E', '#F0CA42', '#F9AEFE', '#A35449', '#FB61B4', '#CDBD72', 

              '#7673DA', '#66EBFF', '#8B76FF', '#8E6856', '#C3C1D7', '#75A4F9']



# The colors are copied from this script: https://www.kaggle.com/ndrewgele/d/abcsds/pokemon/visualizing-pok-mon-stats-with-seaborn

# The colors look reasonable in this map: For example, Green for Grass, Red for Fire, Blue for Water...

COLOR_MAP = dict(zip(TYPE_LIST, COLOR_LIST))





# A radar chart example: http://datascience.stackexchange.com/questions/6084/how-do-i-create-a-complex-radar-chart

def _scale_data(data, ranges):

    (x1, x2), d = ranges[0], data[0]

    return [(d - y1) / (y2 - y1) * (x2 - x1) + x1 for d, (y1, y2) in zip(data, ranges)]



class RaderChart():

    def __init__(self, fig, variables, ranges, n_ordinate_levels = 6):

        angles = np.arange(0, 360, 360./len(variables))



        axes = [fig.add_axes([0.1,0.1,0.8,0.8],polar = True, label = "axes{}".format(i)) for i in range(len(variables))]

        _, text = axes[0].set_thetagrids(angles, labels = variables)

        

        for txt, angle in zip(text, angles):

            txt.set_rotation(angle - 90)

        for ax in axes[1:]:

            ax.patch.set_visible(False)

            ax.xaxis.set_visible(False)

            ax.grid("off")

        

        for i, ax in enumerate(axes):

            grid = np.linspace(*ranges[i], num = n_ordinate_levels)

            grid_label = [""]+[str(int(x)) for x in grid[1:]]

            ax.set_rgrids(grid, labels = grid_label, angle = angles[i])

            ax.set_ylim(*ranges[i])

        

        self.angle = np.deg2rad(np.r_[angles, angles[0]])

        self.ranges = ranges

        self.ax = axes[0]



    def plot(self, data, *args, **kw):

        sdata = _scale_data(data, self.ranges)

        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)



    def fill(self, data, *args, **kw):

        sdata = _scale_data(data, self.ranges)

        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)



    def legend(self, *args, **kw):

        self.ax.legend(*args, **kw)

        

# select display colors according to Pokemon's Type 1

def select_color(types):

    colors = [None] * len(types)

    used_colors = set()

    for i, t in enumerate(types):

        curr = COLOR_MAP[t]

        if curr not in used_colors:

            colors[i] = curr

            used_colors.add(curr)

    unused_colors = set(COLOR_LIST) - used_colors

    for i, c in enumerate(colors):

        if not c:

            try:

                colors[i] = unused_colors.pop()

            except:

                raise Exception('Attempt to visualize too many pokemons. No more colors available.')

    return colors







df = stats

df = df.reset_index()

# In this order, 

# HP, Defense and Sp. Def will show on left; They represent defense abilities

# Speed, Attack and Sp. Atk will show on right; They represent attack abilities

# Attack and Defense, Sp. Atk and Sp. Def will show on opposite positions

use_attributes = ['Speed', 'Sp. Atk', 'Defense', 'HP', 'Sp. Def', 'Attack']

# choose the pokemons you like

use_pokemons = ['Steel','Dragon']



df_plot = df[df['Type 1'].map(lambda x:x in use_pokemons)==True] #df[df['Name']

use_pokemons = df_plot['Type 1'].values

datas = df_plot[use_attributes].values 

ranges = [[2**-20, df_plot[attr].max()] for attr in use_attributes]

colors = select_color(df_plot['Type 1']) # select colors based on pokemon Type 1 #'Type 1'



fig = plt.figure(figsize=(10, 10))

radar = RaderChart(fig, use_attributes, ranges)

for data, color, pokemon in zip(datas, colors, use_pokemons):

    radar.plot(data, color = color, label = pokemon)

    radar.fill(data, alpha = 0.1, color = color)

    radar.legend(loc = 1, fontsize = 'small')

plt.title('Mean Stats of '+(', '.join(use_pokemons[:-1])+' and '+use_pokemons[-1] if len(use_pokemons)>1 else use_pokemons[0]))

plt.show() 

      