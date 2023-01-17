
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
from datetime import date, timedelta
import pickle    
import matplotlib
import matplotlib.pyplot as plt
color_map = plt.cm.winter
from matplotlib.patches import RegularPolygon
import math 

plt.style.use('fivethirtyeight')
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcolors


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.ListedColormap('CustomMap', cdict)

c = mcolors.ColorConverter().to_rgb
rvb = ListedColormap([c('#28aee4'), c('#a1ceee'),c('#e1e5e5'),c('#e78c79'),c('#d63b36')])
from PIL import Image

with open('../input/nhl-2019-season-data/2019FullDataset.pkl', 'rb') as f:
    game_data = pickle.load(f)
x_shot = [];x_missed =[];x_blocked=[];x_goal=[];x_hit=[]
y_shot =[];y_missed =[];y_blocked=[];y_goal=[];y_hit=[]
x_shot_slap = [];x_shot_wrist = [];x_shot_snap = []
y_shot_slap = [];y_shot_wrist = [];y_shot_snap = []

x_goal_slap = [];x_goal_wrist = [];x_goal_snap = []
y_goal_slap = [];y_goal_wrist = [];y_goal_snap = []
x_faceoff = []
y_faceoff = []

for data in game_data:
    if 'liveData' not in data:
        continue
    plays = data['liveData']['plays']['allPlays']
    
    for play in plays:
            
        if play['result']['event'] in ['Shot']:
            if 'x' in play['coordinates']:
                x_shot.append(play['coordinates']['x'])
                y_shot.append(play['coordinates']['y'])
            if 'secondaryType' in play['result']:
                if 'Slap Shot' in play['result']['secondaryType']:
                    x_shot_slap.append(play['coordinates']['x'])
                    y_shot_slap.append(play['coordinates']['y'])
                if 'Wrist Shot' in play['result']['secondaryType']:
                    x_shot_wrist.append(play['coordinates']['x'])
                    y_shot_wrist.append(play['coordinates']['y'])  
                if 'Snap Shot' in play['result']['secondaryType']:
                    x_shot_snap.append(play['coordinates']['x'])
                    y_shot_snap.append(play['coordinates']['y'])  
                
                
    for play in plays:
        if play['result']['event'] in ['Blocked Shot']:
            if 'x' in play['coordinates']:
                x_blocked.append(play['coordinates']['x'])
                y_blocked.append(play['coordinates']['y'])
   
    for play in plays:
        if play['result']['event'] in ['Missed Shot']:
            if 'x' in play['coordinates']:
                x_missed.append(play['coordinates']['x'])
                y_missed.append(play['coordinates']['y'])
        
    for play in plays:
        if play['result']['event'] in ['Goal']:
            if 'x' in play['coordinates']:
                x_goal.append(play['coordinates']['x'])
                y_goal.append(play['coordinates']['y'])
                if 'secondaryType' in play['result']:
                    if 'Slap Shot' in play['result']['secondaryType']:
                        x_goal_slap.append(play['coordinates']['x'])
                        y_goal_slap.append(play['coordinates']['y'])
                    if 'Wrist Shot' in play['result']['secondaryType']:
                        x_goal_wrist.append(play['coordinates']['x'])
                        y_goal_wrist.append(play['coordinates']['y'])  
                    if 'Snap Shot' in play['result']['secondaryType']:
                        x_goal_snap.append(play['coordinates']['x'])
                        y_goal_snap.append(play['coordinates']['y'])  
    for play in plays:
        if play['result']['event'] in ['Hit']:
            if 'x' in play['coordinates']:
                x_hit.append(play['coordinates']['x'])
                y_hit.append(play['coordinates']['y'])
    for play in plays:
        if play['result']['event'] in ['Faceoff']:
            if 'x' in play['coordinates']:
                x_faceoff.append(play['coordinates']['x'])
                y_faceoff.append(play['coordinates']['y'])
# Shot Type Analysis
xbnds = np.array([-100.,100.0])
ybnds = np.array([-100,100])
extent = [xbnds[0],xbnds[1],ybnds[0],ybnds[1]]
fig=plt.figure(figsize=(50,50))
ax = fig.add_subplot(111)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
fig.patch.set_alpha(0.0)
ax.set_xticklabels(labels = [''], fontsize = 18,alpha = .7,minor=False)
ax.set_yticklabels(labels = [''], fontsize = 18,alpha = .7,minor=False)
basewidth = 100

# I = Image.open('../input/nhl-images/Half.png')
# wpercent = (basewidth/float(I.size[0]))
# hsize = int((float(I.size[1])*float(wpercent)))
# I = I.resize((basewidth,hsize), Image.ANTIALIAS)

# ax.imshow(I)
# print(I.size)
    
# scalingx = I.size[0]/100. -2.8
# scalingy = I.size[1]/100. + 1.4
# x_trans = 165.
# y_trans = I.size[1]/2. - 10

scalingx=1
scalingy=1
x_trans=0
y_trans=0
print(len(x_shot_wrist))
print(len(x_shot_slap))
print(len(x_shot_snap))

total_shots = len(x_shot_wrist)+len(x_shot_slap)+len(x_shot_snap)
print(total_shots)
total_goal = len(x_goal_wrist)+len(x_goal_slap)+len(x_goal_snap)
print(total_goal)



print(len(x_shot_wrist)/total_shots)
print(len(x_shot_slap)/total_shots)
print(len(x_shot_snap)/total_shots)
S = 2.3
# Wrist Shot
x_shot_wrist2 = [abs(y) for y in x_shot_wrist];y_shot_wrist2 = [-y for y in y_shot_wrist]
image = plt.hexbin(x_shot_wrist2,y_shot_wrist2,cmap='Reds',gridsize=50,extent=extent,mincnt=1,bins='log',alpha=0.0,edgecolors ='white',linewidths=1)
shot_freq = image.get_array()
shot_verts = image.get_offsets()
for i,v in enumerate(shot_verts):
    total_freq = shot_freq[i]/max(shot_freq)
    radius = S*math.sqrt(total_freq)
    if radius < 1.: radius =0
    radius = radius
    hex = RegularPolygon((x_trans+v[0]*scalingx, y_trans+ v[1]*scalingy), numVertices=6, radius=radius, orientation=np.radians(0),facecolor='#f6cd60', alpha=0.8, edgecolor=None)
    ax.add_patch(hex)
    
# Slap Shot
x_shot_slap2 = [abs(y) for y in x_shot_slap];y_shot_slap2 = [-y for y in y_shot_slap]
image = plt.hexbin(x_shot_slap2,y_shot_slap2,cmap='Reds',gridsize=50,extent=extent,mincnt=1,bins='log',alpha=0.0,edgecolors ='white',linewidths=1)
shot_freq = image.get_array()
shot_verts = image.get_offsets()
for i,v in enumerate(shot_verts):
    total_freq = shot_freq[i]/max(shot_freq)
    radius = S*math.sqrt(total_freq)
    if radius < 1.: radius =0
    radius = radius
    hex = RegularPolygon((x_trans+v[0]*scalingx, y_trans+ v[1]*scalingy), numVertices=6, radius=radius, orientation=np.radians(0),facecolor='#d63b36', alpha=0.8, edgecolor=None)
    ax.add_patch(hex)
    
# Snap Shot
x_shot_snap2 = [abs(y) for y in x_shot_snap];y_shot_snap2 = [-y for y in y_shot_snap]
image = plt.hexbin(x_shot_snap2,y_shot_snap2,cmap='Reds',gridsize=50,extent=extent,mincnt=1,bins='log',alpha=0.0,edgecolors ='white',linewidths=1)
shot_freq = image.get_array()
shot_verts = image.get_offsets()
for i,v in enumerate(shot_verts):
    total_freq = shot_freq[i]/max(shot_freq)
    radius = S*math.sqrt(total_freq)
    if radius < 1.: radius =0
    radius = radius
    hex = RegularPolygon((x_trans+v[0]*scalingx, y_trans+ v[1]*scalingy), numVertices=6, radius=radius, orientation=np.radians(0),facecolor='#28aee4', alpha=0.8, edgecolor=None)
    ax.add_patch(hex) 
    
    
# ANCHOR
x_faceoff2 = [abs(y) for y in x_faceoff];y_faceoff2 = [-y for y in y_faceoff]
image = plt.hexbin(x_faceoff2,y_faceoff2,cmap='Reds',gridsize=50,extent=extent,mincnt=1,bins='log',alpha=0.0,edgecolors ='white',linewidths=1)
shot_freq = image.get_array()
shot_verts = image.get_offsets()
for i,v in enumerate(shot_verts):
    total_freq = shot_freq[i]/max(shot_freq)
    
    hex = RegularPolygon((x_trans+v[0]*scalingx, y_trans+ v[1]*scalingy), numVertices=8, radius=0.1, orientation=np.radians(0),facecolor='#000000', alpha=0.8, edgecolor=None)
    ax.add_patch(hex) 
    
ax.set_xlim([0,100])
ax.set_ylim([-50,50])

plt.grid(False)

plt.show()
# fig.savefig('filename.png', dpi=300)
# # plt.savefig('demo.png', transparent=True)


