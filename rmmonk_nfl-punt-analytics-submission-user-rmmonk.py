# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# To view HTML links in this window
from IPython.display import HTML
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pd.set_option('display.max_colwidth', -1)  # makes columns wider

# Load the datasets, comment on lines we will not be using. 
game_data = pd.read_csv('../input/game_data.csv')
# post_2016 = pd.read_csv('../input/NGS-2016-post.csv')
# pre_2016 = pd.read_csv('../input/NGS-2016-pre.csv')
# reg_2016_wk16 = pd.read_csv('../input/NGS-2016-reg-wk1-6.csv')
# reg_2016_wk1317 = pd.read_csv('../input/NGS-2016-reg-wk13-17.csv')
# reg_2016_wk712 = pd.read_csv('../input/NGS-2016-reg-wk7-12.csv')
# post_2017 = pd.read_csv('../input/NGS-2017-post.csv')
# pre_2017 = pd.read_csv('../input/NGS-2017-pre.csv')
# reg_2017_wk16 = pd.read_csv('../input/NGS-2017-reg-wk1-6.csv')
# reg_2017_wk1317 = pd.read_csv('../input/NGS-2017-reg-wk13-17.csv')
# reg_2017_wk712 = pd.read_csv('../input/NGS-2017-reg-wk7-12.csv')
# play_information = pd.read_csv('../input/play_information.csv')
# player_role = pd.read_csv('../input/play_player_role_data.csv')
# player_punt = pd.read_csv('../input/player_punt_data.csv')
video_footage_injury = pd.read_csv('../input/video_footage-injury.csv')
video_footage_control = pd.read_csv('../input/video_footage-control.csv')
video_review = pd.read_csv('../input/video_review.csv')

# set gamekey playid column to GameKey and PlayId (for merging)
video_footage_injury.columns = ['Season_Year', 'Season_Type', 'Week', 'Home_Team', 'Visit_Team', 'Qtr', 'Play_Description', 'GameKey', 'PlayID', 'Preview_Link']

# Set up dataframe to merge info with video review files
video_injury_links = video_footage_injury[['GameKey', 'PlayID', 'Play_Description', 'Preview_Link', 'Week', 'Home_Team', 'Visit_Team', 'Qtr']]

# Create the injury_info dataframe by merging video_review and video_injury_links dataframes
injury_info = pd.merge(video_review, video_injury_links, how='left', on=['GameKey','PlayID'])

# MERGE with game info, selected columns if we want to analyze external game factors against concussions
game_data_merge = game_data[['GameKey', 'Season_Type', 'Game_Site', 'StadiumType', 'Turf', 'GameWeather', 'Game_Day','Start_Time', 'Temperature', 'OutdoorWeather']]

injury_and_game_data = pd.merge(injury_info, game_data_merge, how='left', on='GameKey')
# BUCKET 1 - Blindside Blocks by Returning Team, Blocker going towards own endzone
blindside_blocks = injury_and_game_data.iloc[[4, 7, 9, 10, 17, 20, 21, 23, 30, 33]]
blindside_blocks.head(15)  # To preview merged table uncomment at beginning of line. Links for each identified play and clips are below. 
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153238/Punt_Return_by_Damiere_Byrd-IX9zynRU-20181119_154215217_5000k.mp4"></video>')

HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153240/Punt_by_Thomas_Morstead-eZpDKgMR-20181119_154525222_5000k.mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153243/Punt_by_Brett_Kern-p3udGBnb-20181119_15513915_5000k.mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="https://nfl-vod.cdn.anvato.net/league/5691/18/11/25/284954/284954_75F12432BA90408C92660A696C1A12C8_181125_284954_huber_punt_3200.mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153247/Punt_by_Tress_Way-QsI21aYF-20181119_160141260_5000k.mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153252/44_yard_Punt_by_Justin_Vogel-n7U6IS6I-20181119_161556468_5000k.mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153253/Justin_Vogel_2-uaXi4twT-20181119_161626398_5000k.mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153258/61_yard_Punt_by_Brett_Kern-g8sqyGTz-20181119_162413664_5000k.mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153280/Wing_37_yard_punt-cPHvctKg-20181119_165941654_5000k.mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153321/Lechler_55_yd_punt-lG1K51rf-20181119_173634665_5000k.mp4"></video>')
excessive_tackling = injury_and_game_data.iloc[[18, 22, 28, 29]]
# excessive_tackling.head(10)  # Uncomment at begininng of line to view table
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153249/Punt_by_Brett_Kern-KYTnoH51-20181119_161310312_5000k.mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153257/40_yard_Punt_by_Brad_Nortman-oSbtDlHu-20181119_162303930_5000k.mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153273/King_62_yard_punt-BSOws7nQ-20181119_165306255_5000k.mp4"></video>')

HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153274/Haack_punts_41_yards-SRJMeOc3-20181119_165546590_5000k.mp4"></video>')
## REFERENCES:
"""
1. I did not know how to embed the videos into the kernel. I used the following code from the following kernel:
Source: https://www.kaggle.com/jmbishop/penalize-all-blindside-blocks

Code used:
from IPython.display import HTML
HTML('<video width="800" height="600" controls> <source src="[SPECIFIC GAME LINK]"></video>')
"""