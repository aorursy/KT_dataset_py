from IPython.display import HTML
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
HTML('<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">WHAT A PLAY. \n Dwayne Harris goes 99 YARDS for a punt return TD! <a href="https://twitter.com/hashtag/RaiderNation?src=hash&amp;ref_src=twsrc%5Etfw">#RaiderNation</a> | <a href="https://twitter.com/hashtag/DENvsOAK?src=hash&amp;ref_src=twsrc%5Etfw">#DENvsOAK</a> <a href="https://t.co/fjEkxMKg0C">pic.twitter.com/fjEkxMKg0C</a></p>&mdash; NFL (@NFL) <a href="https://twitter.com/NFL/status/1077375040507629569?ref_src=twsrc%5Etfw">2018. december 24.</a></blockquote>'
'<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>')
HTML('<video width="560" height="315" controls> <source src="http://a.video.nfl.com//films/vodzilla/153323/Kern_55_yd_punt-iJvv7OOA-20181119_173911500_5000k.mp4 " type="video/mp4"></video>')
HTML('<video width="560" height="315" controls> <source src="https://nfl-vod.cdn.anvato.net/league/5691/18/11/25/284956/284956_12D27120C06E4DB994040750FB43991D_181125_284956_way_punt_3200.mp4" type="video/mp4"></video>')
# Reading in NGS Data...
pre2016 = pd.read_csv('/kaggle/input/NGS-2016-pre.csv')
pre2017 = pd.read_csv('/kaggle/input/NGS-2017-pre.csv')
reg2016_1 = pd.read_csv('/kaggle/input/NGS-2016-reg-wk1-6.csv')
reg2016_2 = pd.read_csv('/kaggle/input/NGS-2016-reg-wk7-12.csv')
reg2016_3 = pd.read_csv('/kaggle/input/NGS-2016-reg-wk13-17.csv')
reg2017_1 = pd.read_csv('/kaggle/input/NGS-2017-reg-wk1-6.csv')
reg2017_2 = pd.read_csv('/kaggle/input/NGS-2017-reg-wk7-12.csv')
reg2017_3 = pd.read_csv('/kaggle/input/NGS-2017-reg-wk13-17.csv')
post2016 = pd.read_csv('/kaggle/input/NGS-2016-post.csv')
post2017 = pd.read_csv('/kaggle/input/NGS-2017-post.csv')

#Catch Location Data:
play_info['catch_x'] = 0
play_info['catch_y'] = 0
#counter = 0
for file in [pre2016,pre2017,reg2016_1,reg2016_2,reg2016_3,reg2017_1,reg2017_2,reg2017_3,post2016,post2017]:
    for game in file.GameKey.unique():
        for playid in file[file.GameKey == game].PlayID.unique():
            try:
                returner_GSISID = PPRD[PPRD.GameKey == game][PPRD.PlayID == playid][PPRD.Role == 'PR'].GSISID.values[0]
            except:
                returner_GSISID = 0
            #print(returner_GSISID)
            #print(playid)
            #print(game)
            #print(file)
            catch_x = -10
            catch_y = -10
            try:
                catch_x = file[file.GameKey == game][file.PlayID == playid][file.GSISID == returner_GSISID][file.Event == 'punt_received'].x.values[0]
                catch_y = file[file.GameKey == game][file.PlayID == playid][file.GSISID == returner_GSISID][file.Event == 'punt_received'].y.values[0]
            except:
                catch_x = file[file.GameKey == game][file.PlayID == playid][file.GSISID == returner_GSISID][file.Event == 'punt_received'].x.values
                catch_y = file[file.GameKey == game][file.PlayID == playid][file.GSISID == returner_GSISID][file.Event == 'punt_received'].y.values
            if catch_x == []:
                catch_x = -10
            if catch_y == []:
                catch_y = -10
            try:
                play_info.loc[(play_info.GameKey == game) & (play_info.PlayID == playid), 'catch_x'] = catch_x
                play_info.loc[(play_info.GameKey == game) & (play_info.PlayID == playid), 'catch_y'] = catch_y
            except:
                play_info.loc[(play_info.GameKey == game) & (play_info.PlayID == playid), 'catch_x'] = -10
                play_info.loc[(play_info.GameKey == game) & (play_info.PlayID == playid), 'catch_y'] = -10
                
# Making Measurements from NGS Data...

#Add approprate columns to Play Info Here:
play_info['avg_avg_speed'] = 0
play_info['avg_max_speed'] = 0
play_info['avg_dist_covered'] = 0
play_info['total_collisions'] = 0
play_info['hang_time'] = 0
play_info['catch_x'] = 0
play_info['catch_y'] = 0
maxspd = []
dist = []
avspd = []
b = 0
# Iterate through all players and plays and compute values...
for file in [reg2017_1,reg2017_2,reg2017_3]:
    for game in file.GameKey.unique():
        for playid in file[file.GameKey == game].PlayID.unique():
            #Play-level values here
            avspeeds = [] #
            maxspeeds = [] #
            distances_covered = [] #
            collisions = []
            hang_time = 0 #
            catch_x = 0 #
            catch_y = 0 #
            for player in file[file.GameKey == game][file.PlayID == playid].GSISID.unique():
                xvals = file[file.GameKey == game][file.PlayID == playid][file.GSISID == player].sort_values(by=['Time']).x.values
                yvals = file[file.GameKey == game][file.PlayID == playid][file.GSISID == player].sort_values(by=['Time']).y.values
                disvals = file[file.GameKey == game][file.PlayID == playid][file.GSISID == player].sort_values(by=['Time']).dis.values
                ovals = file[file.GameKey == game][file.PlayID == playid][file.GSISID == player].sort_values(by=['Time']).o.values
                dirvals = file[file.GameKey == game][file.PlayID == playid][file.GSISID == player].sort_values(by=['Time']).dir.values
                events = file[file.GameKey == game][file.PlayID == playid][file.GSISID == player].sort_values(by=['Time']).Event.values
                
                # Hang-Time and Catch Location
                if PPRD[PPRD.GameKey == game][PPRD.PlayID == playid][PPRD.GSISID == player].Role.values == 'PR':
                    recieved_index = 0
                    for i in range(len(events)):
                        if events[i] == 'punt':
                            punt_index = i
                        if events[i] == 'punt_recieved':
                            recieved_index = i
                    hang_time = (recieved_index-punt_index)/10
                    catch_x = xvals[recieved_index]
                    catch_y = yvals[recieved_index]
                    
                # Distance, Speed, and Collisions
                pos_change = []
                speed = []
                play_started = False
                play_not_finished = True
                i = 0
                play_going = True
                while (i < len(xvals)) & play_going:
                    if i == 0:
                        dP = 0
                    else:
                        dP = (((xvals[i] - xvals[i-1])**2)+((yvals[i] - yvals[i-1])**2))**(.5)
                    if events[i] == 'ball_snap':
                        play_started = True
                    if events[i] == 'tackle':
                        play_going = False
                    pos_change.append(dP)
                    speed.append(dP*10) # 1/10th of second timesteps
                    i = i+1
                avspeeds.append(np.mean(speed))
                if len(speed) == 0:
                    maxspeeds.append(-1)
                else:
                    maxspeeds.append(np.max(speed))
                distances_covered.append(np.sum(pos_change))
                #print('Did a player')
            collisions = [0] #placeholder value
            play_info.loc[(play_info.GameKey == game) & (play_info.PlayID == playid), 'avg_avg_speed'] = np.mean(avspeeds)
            play_info.loc[(play_info.GameKey == game) & (play_info.PlayID == playid), 'avg_max_speed'] = np.mean(maxspeeds)
            play_info.loc[(play_info.GameKey == game) & (play_info.PlayID == playid), 'avg_dist_covered'] = np.mean(distances_covered)
            maxspd.append(np.mean(maxspeeds))
            avspd.append(np.mean(avspeeds))
            dist.append(np.mean(distances_covered))
            
play_info.to_csv('play_info_with_measurements.csv',index = False)