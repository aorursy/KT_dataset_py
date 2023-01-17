import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_csv('../input/mm_master_demos.csv')
analyzed_map = 'de_mirage' # It's the best map after all !!!

# Filter by map & type of rounds, we don't want eco rounds as those tend to be more agressive
data = data[(data.map == analyzed_map) & ((data.round_type == 'PISTOL_ROUND') | (data.round_type == 'NORMAL'))]
# Code from billfreeman44 | See his kernel : https://www.kaggle.com/billfreeman44/finding-classic-smokes-by-t-side-on-mirage 
# Used to convert Dataset positions to positions on the radar
# Keep in mind that the startX, endX resX, startY, endY, resY are specific to mirage, coordinates for each map are available in the dataset

def pointx_to_resolutionx(xinput,startX=-3217,endX=1912,resX=1024):
    sizeX=endX-startX
    if startX < 0:
        xinput += startX *(-1.0)
    else:
        xinput += startX
    xoutput = float((xinput / abs(sizeX)) * resX);
    return xoutput

def pointy_to_resolutiony(yinput,startY=-3401,endY=1682,resY=1024):
    sizeY=endY-startY
    if startY < 0:
        yinput += startY *(-1.0)
    else:
        yinput += startY
    youtput = float((yinput / abs(sizeY)) * resY);
    return resY-youtput

# Convert the data to radar positions
data['attacker_mapX'] = data['att_pos_x'].apply(pointx_to_resolutionx)
data['attacker_mapY'] = data['att_pos_y'].apply(pointy_to_resolutiony)
data.head()
ct_data = data[(data.is_bomb_planted == False) & (data.att_side == 'CounterTerrorist') & (data.winner_side == 'CounterTerrorist')]

# Code from billfreeman44 | See his kernel : https://www.kaggle.com/billfreeman44/finding-classic-smokes-by-t-side-on-mirage 
im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(20,20))
t = plt.imshow(im)
t = plt.scatter(ct_data['attacker_mapX'], ct_data['attacker_mapY'],alpha=0.005,c='blue')
ct_data_awp = data[(data.is_bomb_planted == False) & (data.att_side == 'CounterTerrorist') & (data.wp == 'AWP') & (data.winner_side == 'CounterTerrorist')]

# Code from billfreeman44 | See his kernel : https://www.kaggle.com/billfreeman44/finding-classic-smokes-by-t-side-on-mirage 
im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(20,20))
t = plt.imshow(im)
t = plt.scatter(ct_data_awp['attacker_mapX'], ct_data_awp['attacker_mapY'],alpha=0.025,c='blue')
t_data_siteA = data[(data.is_bomb_planted == True) & (data.att_side == 'Terrorist') & (data.bomb_site == 'A') & (data.winner_side == 'Terrorist')]

# Code from billfreeman44 | See his kernel : https://www.kaggle.com/billfreeman44/finding-classic-smokes-by-t-side-on-mirage 
im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(20,20))
t = plt.imshow(im)
t = plt.scatter(t_data_siteA['attacker_mapX'], t_data_siteA['attacker_mapY'],alpha=0.025,c='orange')
t_data_siteB = data[(data.is_bomb_planted == True) & (data.att_side == 'Terrorist') & (data.bomb_site == 'B') & (data.winner_side == 'Terrorist')]

# Code from billfreeman44 | See his kernel : https://www.kaggle.com/billfreeman44/finding-classic-smokes-by-t-side-on-mirage 
im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(20,20))
t = plt.imshow(im)
t = plt.scatter(t_data_siteB['attacker_mapX'], t_data_siteB['attacker_mapY'],alpha=0.025,c='orange')