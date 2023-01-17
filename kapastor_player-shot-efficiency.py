import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle    
import matplotlib
import matplotlib.pyplot as plt
color_map = plt.cm.winter
from matplotlib.patches import RegularPolygon
import math 

# Needed for custom colour mapping!
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcolors
c = mcolors.ColorConverter().to_rgb
positive_cm = ListedColormap([c('#e1e5e5'),c('#e78c79'),c('#d63b36')]) # Positive map
negative_cm = ListedColormap([c('#e1e5e5'), c('#a1ceee'),c('#28aee4')]) # Negative map

from PIL import Image
with open('../input/nhl-data/2019FullDataset.pkl', 'rb') as f:
    game_data = pickle.load(f)
# Lets make a dictionary to capture all of the needed data
league_data = {};
league_data['Shot'] = {};league_data['Shot']['x'] = [];league_data['Shot']['y'] = [];
league_data['Goal'] = {};league_data['Goal']['x'] = [];league_data['Goal']['y'] = [];

# We are only looking for shot and goal type events to count toward the SOG %
event_types = ['Shot','Goal']
for data in game_data:
    if 'liveData' not in data: # Make sure the data is valid
        continue
    
    # Capture all of the plays for the game
    plays = data['liveData']['plays']['allPlays']
    for play in plays: # For each play get all of the events and capture the Shot and Goals
        for event in event_types:
            if play['result']['event'] in [event]:
                if 'x' in play['coordinates']:
                    league_data[event]['x'].append(play['coordinates']['x'])
                    league_data[event]['y'].append(play['coordinates']['y'])
# Get the player SOG % 
full_name = 'Auston Matthews'
player_data = {};
player_data['Shot'] = {};player_data['Shot']['x'] = [];player_data['Shot']['y'] = [];
player_data['Goal'] = {};player_data['Goal']['x'] = [];player_data['Goal']['y'] = [];
event_types = ['Shot','Goal']
for data in game_data:
    if 'liveData' not in data:
        continue
    plays = data['liveData']['plays']['allPlays']
    for play in plays:
        if 'players' in play:
            for player in play['players']:
                # Here we do the filtering on who was involved in the play and who took the shot or scored.
                if player['player']['fullName'] in [full_name] and player['playerType'] in ["Shooter","Scorer"]:
                    for event in event_types:
                        if play['result']['event'] in [event]:
                            if 'x' in play['coordinates']:
                                player_data[event]['x'].append(play['coordinates']['x'])
                                player_data[event]['y'].append(play['coordinates']['y'])

player_total_shots = len(player_data['Shot']['x']) + len(player_data['Goal']['x'])
player_goal_pct = len(player_data['Goal']['x'])/player_total_shots
league_total_shots = len(league_data['Shot']['x']) + len(league_data['Goal']['x'])
league_goal_pct = len(league_data['Goal']['x'])/league_total_shots
PL_e_spread = player_goal_pct-league_goal_pct


print("Player Total Shots: " + str(player_total_shots))
print("Player Total Goals: " + str(len(player_data['Goal']['x'])))
print("Player SOG %: " + str(player_goal_pct))


print("League Total Shots: " + str(league_total_shots))
print("League SOG %: " + str(league_goal_pct))

# Get the average spread on shot efficiency
print("Player Vs League SOG% Spread: " + str(PL_e_spread))
xbnds = np.array([-100.,100.0])
ybnds = np.array([-100,100])
extent = [xbnds[0],xbnds[1],ybnds[0],ybnds[1]]
gridsize= 30;mincnt=0
league_x_all_shots = league_data['Shot']['x'] + league_data['Goal']['x'];league_y_all_shots = league_data['Shot']['y'] + league_data['Goal']['y']

# If we need to flip the x coordinate then we need to also flip the y coordinate!
league_x_all_shots_normalized = [];league_y_all_shots_normalized=[]
for i,s in enumerate(league_x_all_shots):
    if league_x_all_shots[i] <0:
        league_x_all_shots_normalized.append(-league_x_all_shots[i])
        league_y_all_shots_normalized.append(-league_y_all_shots[i])
    else:
        league_x_all_shots_normalized.append(league_x_all_shots[i])
        league_y_all_shots_normalized.append(league_y_all_shots[i])
        
# If we need to flip the x coordinate then we need to also flip the y coordinate!
league_x_goal_normalized = [];league_y_goal_normalized=[]
for i,s in enumerate(league_data['Goal']['x']):
    if league_data['Goal']['x'][i] <0:
        league_x_goal_normalized.append(-league_data['Goal']['x'][i])
        league_y_goal_normalized.append(-league_data['Goal']['y'][i])
    else:
        league_x_goal_normalized.append(league_data['Goal']['x'][i])
        league_y_goal_normalized.append(league_data['Goal']['y'][i])

league_hex_data = plt.hexbin(league_x_all_shots_normalized,league_y_all_shots_normalized,gridsize=gridsize,extent=extent,mincnt=mincnt,alpha=0.0)
league_verts = league_hex_data.get_offsets();
league_shot_frequency = league_hex_data.get_array();
league_goal_hex_data = plt.hexbin(league_x_goal_normalized,league_y_goal_normalized,gridsize=gridsize,extent=extent,mincnt=mincnt,alpha=0.0)
league_goal_frequency = league_goal_hex_data.get_array();
# Create a new figure for plotting
fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
fig.patch.set_alpha(0.0)
ax.set_xticklabels(labels = [''], fontsize = 18,alpha = .7,minor=False)
ax.set_yticklabels(labels = [''], fontsize = 18,alpha = .7,minor=False)

# Using pillow to get the rink image and rescale the data base on the image size
I = Image.open('../input/nhl-images/half.png')
ax.imshow(I);width, height = I.size
scalingx=width/100-0.6;scalingy=height/100+0.5;x_trans=33;y_trans=height/2
S = 3.8*scalingx;

# Loop over the locations and draw the hex
for i,v in enumerate(league_verts):
    if league_shot_frequency[i] < 1:continue
        
    scaled_league_shot_frequency = league_shot_frequency[i]/max(league_shot_frequency)
    radius = S*math.sqrt(scaled_league_shot_frequency)
    # Scale the radius to the number of goals made in that area
    hex = RegularPolygon((x_trans+v[0]*scalingx, y_trans-v[1]*scalingy), \
                         numVertices=6, radius=radius, orientation=np.radians(0), \
                          alpha=0.5, edgecolor=None)
    ax.add_patch(hex) 

player_x_all_shots = player_data['Shot']['x'] + player_data['Goal']['x'];player_y_all_shots = player_data['Shot']['y'] + player_data['Goal']['y']

# If we need to flip the x coordinate then we need to also flip the y coordinate!
player_x_all_shots_normalized = [];player_y_all_shots_normalized=[]
for i,s in enumerate(player_x_all_shots):
    if player_x_all_shots[i] <0:
        player_x_all_shots_normalized.append(-player_x_all_shots[i])
        player_y_all_shots_normalized.append(-player_y_all_shots[i])
    else:
        player_x_all_shots_normalized.append(player_x_all_shots[i])
        player_y_all_shots_normalized.append(player_y_all_shots[i])
        
# If we need to flip the x coordinate then we need to also flip the y coordinate!
player_x_goal_normalized = [];player_y_goal_normalized=[]
for i,s in enumerate(player_data['Goal']['x']):
    if player_data['Goal']['x'][i] <0:
        player_x_goal_normalized.append(-player_data['Goal']['x'][i])
        player_y_goal_normalized.append(-player_data['Goal']['y'][i])
    else:
        player_x_goal_normalized.append(player_data['Goal']['x'][i])
        player_y_goal_normalized.append(player_data['Goal']['y'][i])

        
        
        
player_hex_data = plt.hexbin(player_x_all_shots_normalized,player_y_all_shots_normalized,gridsize=gridsize,extent=extent,mincnt=mincnt,alpha=0.0)
player_verts = player_hex_data.get_offsets();
player_shot_frequency = player_hex_data.get_array();
player_goal_hex_data = plt.hexbin(player_x_goal_normalized,player_y_goal_normalized,gridsize=gridsize,extent=extent,mincnt=mincnt,alpha=0.0)
player_goal_frequency = player_goal_hex_data.get_array();

# Create a new figure for plotting
fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
fig.patch.set_alpha(0.0)
ax.set_xticklabels(labels = [''], fontsize = 18,alpha = .7,minor=False)
ax.set_yticklabels(labels = [''], fontsize = 18,alpha = .7,minor=False)

# Using pillow to get the rink image and rescale the data base on the image size
I = Image.open('../input/nhl-images/half.png')
ax.imshow(I);width, height = I.size
scalingx=width/100-0.6;scalingy=height/100+0.5;x_trans=33;y_trans=height/2
S = 3.8*scalingx;

# Loop over the locations and draw the hex
for i,v in enumerate(player_verts):
    if player_shot_frequency[i] < 1:continue
    
    scaled_player_shot_frequency = player_shot_frequency[i]/max(player_shot_frequency)
    radius = S*math.sqrt(scaled_player_shot_frequency)
    hex = RegularPolygon((x_trans+v[0]*scalingx, y_trans-v[1]*scalingy), \
                         numVertices=6, radius=radius, orientation=np.radians(0), \
                          facecolor='#0000FF',alpha=0.1, edgecolor=None)
    ax.add_patch(hex)
    
    
    scaled_player_goal_frequency = player_goal_frequency[i]/max(player_goal_frequency)
    radius = S*math.sqrt(scaled_player_goal_frequency)
    # Scale the radius to the number of goals made in that area
    hex = RegularPolygon((x_trans+v[0]*scalingx, (y_trans-v[1]*scalingy)), \
                         numVertices=6, radius=radius, orientation=np.radians(0), \
                         facecolor='#00FF00', alpha=0.9, edgecolor=None)
    ax.add_patch(hex) 
league_efficiency = []
player_efficiency = []
relative_efficiency = []
for i in range(0,len(league_shot_frequency)):
    if league_shot_frequency[i]<2 or player_shot_frequency[i]<2:
        continue
    league_efficiency.append(league_goal_frequency[i]/league_shot_frequency[i])
    player_efficiency.append(player_goal_frequency[i]/player_shot_frequency[i])
    relative_efficiency.append((player_goal_frequency[i]/player_shot_frequency[i])-(league_goal_frequency[i]/league_shot_frequency[i]))

max_league_efficiency = max(league_efficiency)
max_player_efficiency = max(player_efficiency)
max_relative_efficiency = max(relative_efficiency)
min_relative_efficiency = min(relative_efficiency)

# Create a new figure for plotting
fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
fig.patch.set_alpha(0.0)
ax.set_xticklabels(labels = [''], fontsize = 18,alpha = .7,minor=False)
ax.set_yticklabels(labels = [''], fontsize = 18,alpha = .7,minor=False)

# Using pillow to get the rink image and rescale the data base on the image size
I = Image.open('../input/nhl-images/half.png')
ax.imshow(I);width, height = I.size
scalingx=width/100-0.6;scalingy=height/100+0.5;x_trans=33;y_trans=height/2
S = 3.8*scalingx;

# Loop over the locations and draw the hex
for i,v in enumerate(player_verts):
    if player_shot_frequency[i] < 1:continue
    
    
    scaled_player_shot_frequency = player_shot_frequency[i]/max(player_shot_frequency)
    radius = S*math.sqrt(scaled_player_shot_frequency)
    
    player_efficiency = player_goal_frequency[i]/player_shot_frequency[i]
    league_efficiency = league_goal_frequency[i]/league_shot_frequency[i]
    relative_efficiency = player_efficiency - league_efficiency
  
    if relative_efficiency>0:
        colour = positive_cm(math.pow(relative_efficiency,0.1))
    else:
        colour = negative_cm(math.pow(-relative_efficiency,0.1))
    
    hex = RegularPolygon((x_trans+v[0]*scalingx, y_trans-v[1]*scalingy), \
                         numVertices=6, radius=radius, orientation=np.radians(0), \
                          facecolor=colour,alpha=1, edgecolor=None)
    ax.add_patch(hex)
    
    
fig=plt.figure(figsize=(50,50))
ax = fig.add_subplot(111)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
fig.patch.set_alpha(0.0)
ax.set_xticklabels(labels = [''], fontsize = 18,alpha = .7,minor=False)
ax.set_yticklabels(labels = [''], fontsize = 18,alpha = .7,minor=False)

# Using pillow to get the rink image and rescale the data base on the image size
I = Image.open('../input/nhl-images/half.png')
ax.imshow(I);width, height = I.size
scalingx=width/100-0.6;scalingy=height/100+0.5;x_trans=33;y_trans=height/2
S = 3.8*scalingx;
# Loop over the locations and draw the hex
for i,v in enumerate(player_verts):
    if player_shot_frequency[i] < 4:continue
    
    
    scaled_player_shot_frequency = player_shot_frequency[i]/max(player_shot_frequency)
    radius = S*math.sqrt(scaled_player_shot_frequency)
    
    player_efficiency = player_goal_frequency[i]/player_shot_frequency[i]
    league_efficiency = league_goal_frequency[i]/league_shot_frequency[i]
    relative_efficiency = player_efficiency - league_efficiency
  
    if relative_efficiency>0:
        colour = positive_cm(math.pow(relative_efficiency,0.1))
    else:
        colour = negative_cm(math.pow(-relative_efficiency,0.1))
    
    hex = RegularPolygon((x_trans+v[0]*scalingx, y_trans-v[1]*scalingy), \
                         numVertices=6, radius=radius, orientation=np.radians(0), \
                          facecolor=colour,alpha=1, edgecolor=None)
    ax.add_patch(hex)
    
ax.set_xlim([0,width])
ax.set_ylim([0,height])
for spine in ax.spines.values():
    spine.set_edgecolor('white')
plt.grid(False)
