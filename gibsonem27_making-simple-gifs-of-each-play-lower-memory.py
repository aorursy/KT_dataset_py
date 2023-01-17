import matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import *

import pandas as pd

import numpy as np

import os

import imageio

import feather

import seaborn as sns

#Trust me, you'll want a progress bar for the final loop in this notebook
import tqdm
import os
print(os.listdir("../input"))
#create custom dtypes to conserve memory
column_types = {'GameKey': 'int64',
 'PlayID': 'int64',
 'GSISID': 'float32',
 'x': 'float32',
 'y': 'float32',
 'dis': 'float32',
 'o': 'float32',
 'dir': 'float32'}
kag_dir = "../input"

# If you decided to make feather checkpoints for your play data, that works.

# path = os.path.join(kag_dir,'feather_chks','filename.feather')
# play = feather.read_dataframe(path)


#If you're not working with a feather file, you can use the original CSV files

# For this example, I'll only use one of our NGS files. You can concatenate files as needed or run this notebook multiple times to get what you need.

play= pd.read_csv("../input/NGS-2016-pre.csv", nrows = 25000, dtype=column_types)

play.head()
#Let's drop some superfluous columns to save memory

# Obviously, as you merge dataframes, you can drop (or not drop) variables as you need. 
play = play.drop(['Season_Year',
                 'Event'], axis=1)

play.head()
# This cell changes the time column to DTG
play['Time'] = pd.to_datetime(play["Time"])
# sort the dataframe
play.sort_values(by=['Time'], axis=0, inplace=True)

# set the index to be this and don't drop
play.set_index(keys=['Time'], drop=False,inplace=True)

# get a list of names
names=play['Time'].unique().tolist()
print(names)

#get a list of the unique game ids and play ids

games=play['GameKey'].unique().tolist()

print(games)

plays=play['PlayID'].unique().tolist()
#This lets us know which plays and games we can expect
print(plays, games)
play.head()
#Creating a dictionary for our dataframes
list_of_games_dfs = {}
for i in games:
    list_of_games_dfs[i] = play.loc[play.GameKey==i]
#This cell draws the football field
def draw_field(ax):
    #field Outline & Center Line
    field = Rectangle([0,0], width = 120, height = 53.3, fill = False)
    
    #All of the yard lines
    line_10 = ConnectionPatch([10,0], [10,53.3], "data", "data")
    line_20 = ConnectionPatch([20,0], [20,53.3], "data", "data")
    line_30 = ConnectionPatch([30,0], [30,53.3], "data", "data")
    line_40 = ConnectionPatch([40,0], [40,53.3], "data", "data")
    line_50 = ConnectionPatch([50,0], [50,53.3], "data", "data")
    line_60 = ConnectionPatch([60,0], [60,53.3], "data", "data")
    line_70 = ConnectionPatch([70,0], [70,53.3], "data", "data")
    line_80 = ConnectionPatch([80,0], [80,53.3], "data", "data")
    line_90 = ConnectionPatch([90,0], [90,53.3], "data", "data")
    line_100 = ConnectionPatch([100,0], [100,53.3], "data", "data")
    line_110 = ConnectionPatch([110,0], [110,53.3], "data", "data")
    
    #Yardlines
    yardmark_10 = plt.text(18, 50, '10')
    yardmark_20 = plt.text(28, 50, '20')
    yardmark_30 = plt.text(38, 50, '30')
    yardmark_40 = plt.text(48, 50, '40')
    yardmark_50 = plt.text(58, 50, '50')
    yardmark_60 = plt.text(68, 50, '40')
    yardmark_70 = plt.text(78, 50, '30')
    yardmark_80 = plt.text(88, 50, '20')
    yardmark_90 = plt.text(98, 50, '10')
    
    #Endzones
    endzone_1 = plt.text(5, 30, 'Home End Zone', rotation = 90)
    endzone_1 = plt.text(114, 30, 'Visitor End Zone', rotation = 270)
    
    element = [field, line_10, line_20,
               line_30, line_40, line_50, 
               line_60, line_70, line_80,
              line_90, line_100, line_110]
    for i in element:
        ax.add_patch(i)
fig=plt.figure() #set up the figures
fig.set_size_inches(12, 8)
ax=fig.add_subplot(1,1,1)
draw_field(ax) #overlay our different objects on the field

plt.ylim(-2, 82)
plt.xlim(-2, 122)
plt.axis('off')

#Choose which data to draw for the KDE plot
sns.kdeplot(list_of_games_dfs[3].loc[(list_of_games_dfs[3].PlayID==3949)].x, 
    list_of_games_dfs[3].loc[(list_of_games_dfs[3].PlayID==3949)].y,
            color= 'green', shade = "True", n_levels = 100, shade_lowest=False);

#Choose the play to visualize
plt.scatter(list_of_games_dfs[3].loc[(list_of_games_dfs[3].PlayID==3949) & (list_of_games_dfs[3].Time=='2016-08-12 02:26:51.200')].x, 
    list_of_games_dfs[3].loc[(list_of_games_dfs[3].PlayID==3949) & (list_of_games_dfs[3].Time=='2016-08-12 02:26:51.200')].y,
    color="blue")

plt.show()
#cCreate a folder to the images we will create in the next cell.
os.mkdir("../images")
#This is important if you are running through a Jupyter notebook. This will ensure that memory doesn't leak as you are creating PNG images.
matplotlib.interactive(False)

ima_fldr = "../images"

# This loop will create PNG files for every instance of every game in our dataset. 
# Utilizing TQDM here will give display a progress bar that steps by GAMES
for i in tqdm.tqdm(list_of_games_dfs):
    
    for z in list_of_games_dfs[i].PlayID.unique():
        
        times = list_of_games_dfs[i].loc[list_of_games_dfs[i].PlayID==z].Time.unique()
        
        for t in times:
            fig=plt.figure() #set up the figures
            fig.set_size_inches(12, 8)
            ax=fig.add_subplot(1,1,1)
            draw_field(ax) #overlay our different objects on the field

            plt.ylim(-2, 82)
            plt.xlim(-2, 122)
            plt.axis('off')
            
        
            plt.scatter(list_of_games_dfs[i].loc[(list_of_games_dfs[i].PlayID==z) & (list_of_games_dfs[i].Time==t)].x, 
                        list_of_games_dfs[i].loc[(list_of_games_dfs[i].PlayID==z) & (list_of_games_dfs[i].Time==t)].y,
                        color="blue")
            filename=os.path.join(ima_fldr, "game" + str(i), "play" + str(z))
            if not os.path.exists(filename):
                os.makedirs(filename)
            plt.savefig(os.path.join(filename, str(t).replace(" ", "_").replace(":", "_") + ".png"))
            # Clear the current axes.
            plt.cla() 
            # Clear the current figure.
            plt.clf() 
            # Closes all the figure windows.
            plt.close('all')
#This cell will create GIF files for every play in our image folder.
files = os.listdir(ima_fldr)
files = [ima_fldr + "/" + s for s in files]
files

files_2 = {}
for file in files:
    files_2[file] = os.listdir(file)

gif_files = []
for i in files_2:
    for x in files_2[i]:
        gif_files.append(os.path.join(i,x))

for f in tqdm.tqdm(gif_files):
    images = []
    for k in os.listdir(f):
        images.append(imageio.imread(os.path.join(f, k)))
    imageio.mimsave(os.path.join(f, 'movie.gif'), images)